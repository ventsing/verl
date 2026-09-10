# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the bounded row-generation retry policy (stdlib-only)."""

import asyncio
import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.row_retry import generate_row_with_retry


class FlakyGenerate:
    """Fails the first ``fail_times`` calls, then returns a result."""

    def __init__(self, fail_times: int, result: str = "ok"):
        self.fail_times = fail_times
        self.calls = 0
        self.result = result

    async def __call__(self, row):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"transient failure #{self.calls}")
        return self.result


class Recorder:
    """Captures deliveries: (payload, attempts, version)."""

    def __init__(self):
        self.deliveries: list[tuple[object, int, int]] = []

    async def __call__(self, result, attempts, version):
        self.deliveries.append((result, attempts, version))


def run(coro):
    return asyncio.run(coro)


class TestRowRetry(unittest.TestCase):
    def test_success_first_try(self):
        gen, rec = FlakyGenerate(0), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 7,
                max_attempts=2,
            )
        )
        self.assertTrue(ok)
        self.assertEqual(gen.calls, 1)
        self.assertEqual(rec.deliveries, [("ok", 1, 7)])

    def test_retry_then_success(self):
        gen, rec = FlakyGenerate(1), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 7,
                max_attempts=2,
            )
        )
        self.assertTrue(ok)
        self.assertEqual(gen.calls, 2)
        self.assertEqual(rec.deliveries, [("ok", 2, 7)])  # attempts counted from 1

    def test_budget_exhausted_delivers_failed_sentinel(self):
        gen, rec = FlakyGenerate(99), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 7,
                max_attempts=3,
            )
        )
        self.assertFalse(ok)
        self.assertEqual(gen.calls, 3)  # bounded: exactly max_attempts
        self.assertEqual(rec.deliveries, [("FAILED", 3, 7)])  # sentinel delivered

    def test_single_shot_is_legacy_behavior(self):
        """max_attempts=1 = today's single-shot semantics: first failure terminal."""
        gen, rec = FlakyGenerate(1), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 7,
                max_attempts=1,
            )
        )
        self.assertFalse(ok)
        self.assertEqual(gen.calls, 1)
        self.assertEqual(rec.deliveries, [("FAILED", 1, 7)])

    def test_max_attempts_floor_is_one(self):
        gen, rec = FlakyGenerate(0), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 7,
                max_attempts=0,  # clamped to 1
            )
        )
        self.assertTrue(ok)
        self.assertEqual(rec.deliveries, [("ok", 1, 7)])

    def test_no_silent_drop_on_deliver_error(self):
        """A delivery failure must propagate (never silently lose a row)."""

        async def bad_deliver(result, attempts, version):
            raise RuntimeError("queue exploded")

        gen = FlakyGenerate(0)
        with self.assertRaises(RuntimeError):
            run(
                generate_row_with_retry(
                    "row",
                    generate_fn=gen,
                    deliver_fn=bad_deliver,
                    failed_row_fn=lambda: "FAILED",
                    version_for_attempt=lambda a: 7,
                )
            )

    def test_failure_then_success_versions_observed(self):
        """Deliveries carry the version of the ATTEMPT that produced them
        (the producer re-stamps retried rows with the current fleet
        version — attempt 1 the group's submission snapshot)."""
        versions = {1: 7, 2: 9}  # fleet pulled between the two attempts

        gen, rec = FlakyGenerate(1), Recorder()
        ok = run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=rec,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: versions[a],
                max_attempts=2,
            )
        )
        self.assertTrue(ok)
        # the successful retry is stamped with the NEWER fleet version
        self.assertEqual(rec.deliveries, [("ok", 2, 9)])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
