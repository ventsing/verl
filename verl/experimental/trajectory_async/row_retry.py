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
"""Bounded row-generation retry — the cheapest long-tail mitigation.

Without retry, one transient failure of a single response's generation
terminates that trajectory and (under strict group semantics) evicts its
whole prompt group: the wasted compute is proportional to the group size,
and the waste concentrates on long generations (timeouts / KV pressure
correlate with length) — a length-biased dropout of the training
distribution, the same failure mode the Laminar paper's Appendix C
attributes to partial-rollout mixing (a milder cousin).

Policy (:func:`generate_row_with_retry`):

* up to ``max_attempts`` generation attempts per row (1 = today's
  single-shot behavior);
* a successful attempt is delivered with its ``attempts`` count and the
  model version it generated under — retried attempts re-stamp the
  version via ``version_for_attempt`` (a retry may run under NEWER
  weights after a batch-boundary pull; the honest stamp is the version at
  the retry's start, and the resulting mixed-version group is exactly
  what version_span + the loss-side staleness correction handle);
* a terminal failure (budget exhausted) is delivered as the producer's
  FAILED sentinel with the attempts consumed — the trainer-side group
  policy (strict eviction vs. survivor delivery) then decides;
* with ``pool_consult_fn`` (the partial response pool, ``partial_pool.py``),
  retries consult it for a same-version partial — a hit resumes the
  trajectory from its saved progress instead of restarting from the
  prompt (paper §3.1/§3.3); a miss restarts clean.

The function is stdlib-pure (injected async callables) so the retry
policy is CPU-testable; ``rollout_producer.py`` wires it to the real
generation path.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

GenerateFn = Callable[[Any, Any], Awaitable[Any]]  # (row, resume_hint | None)
DeliverFn = Callable[[Any, int, int], Awaitable[None]]  # (row_result, attempts, version)
FailedRowFn = Callable[[], Any]
VersionForAttemptFn = Callable[[int], int]
# partial-pool consult hook (attempt > 1): returns a resume hint (e.g.
# partial tokens) for this attempt's version, or None for a clean restart
PoolConsultFn = Callable[[int, int], Awaitable[Any]]  # (attempt, version) -> hint | None


async def generate_row_with_retry(
    row: Any,
    *,
    generate_fn: GenerateFn,
    deliver_fn: DeliverFn,
    failed_row_fn: FailedRowFn,
    version_for_attempt: VersionForAttemptFn,
    pool_consult_fn: PoolConsultFn | None = None,
    max_attempts: int = 2,
    label: str = "",
) -> bool:
    """Generate one row with a bounded attempt budget.

    Args:
        row: the 1-row generation input.
        generate_fn: runs one generation attempt, ``(row, resume_hint)``;
            raises on failure. ``resume_hint`` is None on the first attempt
            and whatever ``pool_consult_fn`` returned on retries (the
            producer's closure decides how to attach it — e.g. a
            ``resume_tokens`` field on the row).
        deliver_fn: async ``(result, attempts, version)`` — delivers a
            row (successful or FAILED sentinel) with its attempts count
            and the model version ``version_for_attempt(attempts)`` says
            it generated under.
        failed_row_fn: builds the FAILED sentinel row (attempt-exhausted).
        version_for_attempt: the model version to stamp for attempt ``k``
            (attempt 1 typically the group's submission snapshot; retries
            the CURRENT fleet version — see module docstring).
        pool_consult_fn: optional partial-pool hook — called before every
            RETRY (attempt >= 2) with ``(attempt, version_for_this_attempt)``;
            a truthy return value is a resume hint (same-version partial
            progress) the caller attaches to the generation input, turning
            the retry into a resume instead of a restart. None/absent =
            clean restart (the documented no-pool behavior).
        max_attempts: total attempts budget (>= 1; 1 = single-shot).
        label: logging context (uid[idx]).

    Returns:
        True if the row eventually succeeded, False if the budget was
        exhausted (the FAILED sentinel is delivered either way — callers
        never see a silent drop).
    """
    max_attempts = max(1, int(max_attempts))
    attempts = 0
    hint = None  # resume hint for the NEXT attempt (None on the first)
    while True:
        attempts += 1
        try:
            result = await generate_fn(row, hint)
        except Exception:  # noqa: BLE001 — any attempt failure is retryable
            if attempts >= max_attempts:
                logger.exception(
                    "trajectory %s generation failed (attempt %d/%d); giving up",
                    label,
                    attempts,
                    max_attempts,
                )
                await deliver_fn(failed_row_fn(), attempts, version_for_attempt(attempts))
                return False
            logger.warning(
                "trajectory %s generation failed (attempt %d/%d); retrying",
                label,
                attempts,
                max_attempts,
                exc_info=True,
            )
            # partial-pool consult: a same-version partial turns the retry
            # into a resume (the hint flows to generate_fn; the producer's
            # closure decides what it means for its row format)
            hint = None
            if pool_consult_fn is not None:
                try:
                    hint = await pool_consult_fn(attempts + 1, version_for_attempt(attempts + 1))
                except Exception:  # noqa: BLE001 — the pool must never break retries
                    hint = None
            if hint:
                logger.info(
                    "trajectory %s retry %d resumes from pooled partial progress",
                    label,
                    attempts + 1,
                )
            continue
        await deliver_fn(result, attempts, version_for_attempt(attempts))
        return True
