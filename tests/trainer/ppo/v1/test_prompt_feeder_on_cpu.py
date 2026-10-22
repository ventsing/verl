# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prompt production continues independently and respects experience backpressure."""

import threading
import time

import pytest

from verl.experimental.trajectory_async.prompt_feeder import PromptFeeder


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "Producer did not make expected progress"
        time.sleep(0.001)


def test_produces_while_consumer_waits_and_stops_at_backpressure():
    lock = threading.RLock()
    backlog = []
    submitted = []

    def submit():
        backlog.extend([object(), object()])
        submitted.append(threading.get_ident())

    feeder = PromptFeeder(lambda: len(backlog), submit, batch_size=2, capacity=4, lock=lock, poll_interval=0.001)
    feeder.start()
    try:
        wait_for(lambda: len(backlog) == 4)
        assert all(thread_id != threading.get_ident() for thread_id in submitted)
        with lock:
            backlog.pop()
            backlog.pop()
        wait_for(lambda: len(submitted) == 3)
        with lock:
            assert len(backlog) == 4
        feeder.check()
    finally:
        feeder.stop()
    assert not feeder.thread.is_alive()


def test_producer_errors_surface_to_consumer():
    def fail():
        raise ValueError("data unavailable")

    feeder = PromptFeeder(lambda: 0, fail, batch_size=1, capacity=2, lock=threading.RLock())
    feeder.start()
    wait_for(lambda: feeder.error is not None)
    with pytest.raises(RuntimeError, match="producer failed") as error:
        feeder.check()
    assert isinstance(error.value.__cause__, ValueError)
    feeder.stop()
