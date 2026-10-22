# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded CPU checksums over immutable snapshots, without copying payloads."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PARALLEL_MIN_BYTES = 64 << 20


def validate_hash_workers(workers):
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("Snapshot hash_workers must be an integer between 1 and 32")
    return workers


def checksum_arrays(arrays, workers=4):
    """Keep tensor order and SHA256 semantics; at most workers CPU threads.

    Each tensor is read once. Tiny snapshots stay serial to avoid pool startup.
    The caller must own/pin buffers until this function has returned.
    """
    validate_hash_workers(workers)
    for array in arrays:
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or not array.flags.c_contiguous:
            raise ValueError("Checksums require contiguous uint8 CPU snapshots")

    def digest(array):
        return hashlib.sha256(memoryview(array.reshape(-1))).hexdigest()

    if workers == 1 or len(arrays) < 2 or sum(a.nbytes for a in arrays) < PARALLEL_MIN_BYTES:
        return [digest(a) for a in arrays]
    with ThreadPoolExecutor(max_workers=min(workers, len(arrays)), thread_name_prefix="relay-sha256") as executor:
        return list(executor.map(digest, arrays))
