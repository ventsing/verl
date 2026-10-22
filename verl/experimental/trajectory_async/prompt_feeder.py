# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded prompt production independent of synchronous actor updates."""

import threading


class PromptFeeder:
    def __init__(self, count_pending, submit_batch, batch_size, capacity, lock, poll_interval=0.2):
        if batch_size < 1 or capacity < batch_size or poll_interval <= 0:
            raise ValueError("Prompt capacity must hold a positive batch and polling must be positive")
        self.count_pending = count_pending
        self.submit_batch = submit_batch
        self.batch_size = batch_size
        self.capacity = capacity
        self.lock = lock
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.batches = 0

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="laminar-prompt-feeder", daemon=True)
            self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if self.count_pending() + self.batch_size <= self.capacity:
                        self.submit_batch()
                        self.batches += 1
                        continue
                self.stop_event.wait(self.poll_interval)
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("Laminar prompt producer failed") from self.error

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Laminar prompt producer did not stop within 10 seconds")
