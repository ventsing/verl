# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fenced partial responses in memory or SQLite. One CPU actor owns the DB.

The default trainer uses :memory: with memory journaling and no fsync. An
explicit file path enables FULL rollback journaling (not WAL), requiring working
filesystem locks/fsync. ACK means committed in the selected store; recovery
requires that store to survive. No model/KV tensors are stored.
"""

import fcntl
import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PartialTrajectoryStore:
    def __init__(self, path, max_active=4096, identity=None):
        self.path = str(path)
        self.max_active = max_active
        self.persistent = self.path != ":memory:"
        self.writer_lock = None
        if self.persistent:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.writer_lock = open(f"{path}.writer.lock", "a")
            try:
                fcntl.flock(self.writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self.writer_lock.close()
                raise RuntimeError("Partial response database already has a live writer") from error
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("PRAGMA journal_mode=DELETE" if self.persistent else "PRAGMA journal_mode=MEMORY")
        self.db.execute("PRAGMA synchronous=FULL" if self.persistent else "PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id TEXT PRIMARY KEY, identity TEXT NOT NULL, version INTEGER NOT NULL,
                epoch INTEGER NOT NULL DEFAULT 0, attempt TEXT,
                size INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'running',
                extra TEXT NOT NULL DEFAULT '{}', updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT NOT NULL, offset INTEGER NOT NULL, tokens TEXT NOT NULL,
                logprobs TEXT NOT NULL, PRIMARY KEY(id, offset)
            );
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS trajectory_versions ON trajectories(status, version);
            CREATE INDEX IF NOT EXISTS trajectory_progress ON trajectories(status, updated);
            PRAGMA user_version=1;
        """)
        if identity is not None:
            expected = encode(identity)
            try:
                with self.db:
                    old = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
                    if old is not None and old[0] != expected:
                        raise ValueError("Partial response database belongs to another model/run")
                    self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (expected,))
            except BaseException:
                self.close()
                raise

    def _row(self, request_id):
        self.db.row_factory = sqlite3.Row
        row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        return row

    def create(self, request_id, prompt_ids, sampling_params, budget, version):
        identity = encode({"prompt_ids": prompt_ids, "sampling_params": sampling_params, "budget": budget})
        with self.db:
            row = self.db.execute("SELECT identity, version FROM trajectories WHERE id=?", (request_id,)).fetchone()
            if row is not None:
                if row[0] != identity or row[1] != version:
                    raise ValueError("Request identity or behavior version changed")
            else:
                active = self.db.execute("SELECT count(*) FROM trajectories WHERE status='running'").fetchone()[0]
                if active >= self.max_active:
                    raise RuntimeError("Partial response pool capacity exceeded")
                self.db.execute(
                    "INSERT INTO trajectories(id, identity, version, updated) VALUES (?, ?, ?, ?)",
                    (request_id, identity, version, time.time()),
                )
        return self.get(request_id)

    def get(self, request_id):
        row = self._row(request_id)
        tokens, logprobs = [], []
        for chunk in self.db.execute("SELECT * FROM chunks WHERE id=? ORDER BY offset", (request_id,)):
            if chunk["offset"] != len(tokens):
                raise RuntimeError("Partial response has a noncontiguous prefix")
            tokens.extend(json.loads(chunk["tokens"]))
            logprobs.extend(json.loads(chunk["logprobs"]))
        if len(tokens) != row["size"] or len(tokens) != len(logprobs):
            raise RuntimeError("Partial response metadata does not match chunks")
        return {
            "request_id": request_id,
            "version": row["version"],
            "epoch": row["epoch"],
            "token_ids": tokens,
            "log_probs": logprobs,
            "status": row["status"],
            "extra_fields": json.loads(row["extra"]),
            "updated": row["updated"],
            **json.loads(row["identity"]),
        }

    def lookup(self, request_id):
        try:
            return self.get(request_id)
        except KeyError:
            return None

    def begin_attempt(self, request_id, attempt):
        # The attempt UUID makes a lost response/Ray task retry idempotent.
        with self.db:
            row = self._row(request_id)
            if row["status"] != "running":
                raise ValueError("Cannot restart a completed trajectory")
            if row["attempt"] != attempt:
                self.db.execute(
                    "UPDATE trajectories SET epoch=epoch+1, attempt=?, updated=? WHERE id=?",
                    (attempt, time.time(), request_id),
                )
        row = self._row(request_id)
        return {"epoch": row["epoch"], "offset": row["size"], "version": row["version"]}

    def fence(self, request_id):
        with self.db:
            self.db.execute("UPDATE trajectories SET epoch=epoch+1, attempt=NULL WHERE id=?", (request_id,))
        return self.get(request_id)

    def append(self, request_id, epoch, version, offset, token_ids, log_probs, terminal=False, extra_fields=None):
        if len(token_ids) != len(log_probs) or not all(math.isfinite(v) for v in log_probs):
            raise ValueError("Every token requires a finite behavior log probability")
        if any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError("Invalid token ID")
        payload = encode(token_ids), encode(log_probs)
        with self.db:
            row = self._row(request_id)
            if row["epoch"] != epoch or row["version"] != version:
                raise ValueError("Stale attempt or behavior policy version")
            if offset < row["size"]:
                old = self.db.execute(
                    "SELECT tokens, logprobs FROM chunks WHERE id=? AND offset=?", (request_id, offset)
                ).fetchone()
                if old is None or tuple(old) != payload:
                    raise ValueError("Conflicting duplicate or overlapping chunk")
                # A retry of the exact committed chunk is acknowledged unchanged.
                return {"offset": row["size"], "status": row["status"]}
            if offset != row["size"]:
                raise ValueError("Out-of-order chunk")
            if row["status"] != "running":
                if not token_ids and terminal:
                    return {"offset": row["size"], "status": row["status"]}
                raise ValueError("Cannot append to completed trajectory")
            budget = json.loads(row["identity"])["budget"]
            size = offset + len(token_ids)
            if size > budget:
                raise ValueError("Remaining trajectory budget exceeded")
            if token_ids:
                self.db.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (request_id, offset, *payload))
            self.db.execute(
                "UPDATE trajectories SET size=?, status=?, extra=?, updated=? WHERE id=?",
                (size, "completed" if terminal else "running", encode(extra_fields or {}), time.time(), request_id),
            )
        return {"offset": size, "status": "completed" if terminal else "running"}

    def abandon(self, request_id, reason):
        if not reason:
            raise ValueError("Abandoning a trajectory requires an explicit reason")
        with self.db:
            row = self._row(request_id)
            if row["status"] == "completed":
                raise ValueError("Cannot abandon a completed trajectory")
            self.db.execute(
                "UPDATE trajectories SET status='abandoned', epoch=epoch+1, attempt=NULL, "
                "extra=?, updated=? WHERE id=?",
                (encode({"abandon_reason": reason}), time.time(), request_id),
            )
        return self.get(request_id)

    def incomplete(self, limit=100):
        return [
            dict(zip(("request_id", "version", "tokens", "updated"), row, strict=True))
            for row in self.db.execute(
                "SELECT id, version, size, updated FROM trajectories WHERE status='running' ORDER BY updated LIMIT ?",
                (limit,),
            )
        ]

    def progress(self, request_id):
        row = self._row(request_id)
        return {"updated": row["updated"], "offset": row["size"], "status": row["status"]}

    def versions(self):
        return [r[0] for r in self.db.execute("SELECT DISTINCT version FROM trajectories WHERE status='running'")]

    def status(self):
        return dict(self.db.execute("SELECT status, count(*) FROM trajectories GROUP BY status"))

    def close(self):
        self.db.close()
        if self.writer_lock is not None and not self.writer_lock.closed:
            fcntl.flock(self.writer_lock.fileno(), fcntl.LOCK_UN)
            self.writer_lock.close()


class StreamJournal:
    """Await every store ACK, providing bounded streaming backpressure.

    vLLM is explicitly configured for CUMULATIVE output. We submit only deltas;
    after process loss, unacknowledged suffixes are sampled again from the last
    committed prefix under exactly the same weights.
    """

    def __init__(self, context):
        self.store = context["store"]
        self.request_id = context["request_id"]
        self.epoch = context["epoch"]
        self.version = context["version"]
        self.base = context["offset"]
        self.chunk_tokens = context.get("chunk_tokens", 128)
        self.interval = context.get("interval_s", 1.0)
        self.written = 0
        self.last_ack = time.monotonic()
        self.digest = hashlib.sha256()

    def due(self, count, finished=False):
        return (
            finished or count - self.written >= self.chunk_tokens or time.monotonic() - self.last_ack >= self.interval
        )

    async def update(self, token_ids, log_probs, version, finished=False, extra_fields=None, force=False):
        if version != self.version:
            raise ValueError("Streaming behavior policy version changed")
        if len(token_ids) < self.written or len(token_ids) != len(log_probs):
            raise ValueError("Streaming output is not cumulative or lacks behavior logprobs")
        if not force and not self.due(len(token_ids), finished):
            return
        # vLLM CUMULATIVE output aliases its live detokenizer list. Freeze it
        # BEFORE awaiting storage: the engine can append while the ACK is pending.
        token_ids, log_probs = list(token_ids), list(log_probs)
        # Detect rewriting an acknowledged prefix, without retaining a second copy.
        digest = hashlib.sha256(encode([token_ids[: self.written], log_probs[: self.written]]).encode()).digest()
        if self.written and digest != self.digest.digest():
            raise ValueError("Streaming backend rewrote an acknowledged prefix")
        delta, probs = list(token_ids[self.written :]), list(log_probs[self.written :])
        await self.store.append.remote(
            self.request_id,
            self.epoch,
            version,
            self.base + self.written,
            delta,
            probs,
            terminal=finished,
            extra_fields=extra_fields,
        )
        self.written = len(token_ids)
        self.digest = hashlib.sha256(encode([list(token_ids), list(log_probs)]).encode())
        self.last_ack = time.monotonic()
