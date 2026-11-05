# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Final-answer-only mathematical reward for thinking-enabled DeepScaleR runs.

Only the final boxed answer after </think> is graded. Symbolic verification
runs in bounded CPU subprocesses, because Math-Verify uses main-thread signals.
Other datasets retain their existing scorer. Raw correctness is never smoothed.
"""

import asyncio
import atexit
import multiprocessing
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from weakref import WeakKeyDictionary

from verl.experimental.trajectory_async.math_choice_answer import resolve_choice_answer
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

_pool = None
_lock = threading.Lock()
_gates = WeakKeyDictionary()
MATH_SOURCES = {"lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"}


def normalize_dataset_answer(answer):
    """MATH uses percentage points and textual lists of alternative functions."""
    answer = answer.strip().rstrip(".").replace("\\left", "").replace("\\right", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:\\%|%)", answer):
        answer = re.sub(r"\s*(?:\\%|%)$", "", answer)
    parts = re.split(r"\\text\s*\{\s*(?:or|and)\s*\}", answer)
    if len(parts) > 1:
        parts = [part.strip() for part in parts]
        equations = [re.fullmatch(r"([a-zA-Z](?:\([^()]*\))?)\s*=\s*(.+)", part) for part in parts]
        if all(equations) and len({match[1] for match in equations}) == 1:
            answer = r"\{" + ",".join(match[2] for match in equations) + r"\}"
    return answer


def extract_final_answer(solution, require_think_close=True, max_answer_chars=4096):
    """The opening thinking tag can already be in the prompt, not the response."""
    if not isinstance(solution, str):
        raise TypeError("Expected decoded response text")
    closes = solution.count("</think>")
    if closes > 1 or (require_think_close and closes != 1):
        return None, "unfinished_thinking" if closes == 0 else "invalid_thinking_format"
    final = solution.split("</think>", 1)[-1] if closes else solution
    if "<think>" in final:
        return None, "invalid_thinking_format"
    boxed = last_boxed_only_string(final)
    if boxed is None:
        return None, "missing_final_answer"
    answer = remove_boxed(boxed).strip()
    if not answer or len(answer) > max_answer_chars:
        return None, "invalid_answer_length"
    return answer, "ok"


@lru_cache(maxsize=4096)
def _parse_gold(ground_truth, timeout_s):
    from math_verify import parse
    from math_verify.parser import LatexExtractionConfig

    return parse(
        "\\boxed{" + ground_truth + "}",
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
        parsing_timeout=timeout_s,
        raise_on_error=True,
    )


def verify_answer(answer, ground_truth, timeout_s=2):
    """Executed on a subprocess main thread; never parse a complete reasoning trace."""
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from math_verify.parser import LatexExtractionConfig

    answer, ground_truth = normalize_dataset_answer(answer), normalize_dataset_answer(ground_truth)
    # Literal equality also supports valid dataset answers outside the symbolic
    # parser's grammar, without accepting an arbitrary matching subexpression.
    if re.sub(r"\s+", "", answer) == re.sub(r"\s+", "", ground_truth):
        return {"acc": 1.0, "unparsed_gold": 0, "unparsed_answer": 0, "verifier_timeout": 0}
    stage = "gold"
    try:
        gold = _parse_gold(ground_truth, timeout_s)
        stage = "answer"
        pred = parse(
            "\\boxed{" + answer + "}",
            extraction_config=[LatexExtractionConfig()],
            fallback_mode="no_fallback",
            parsing_timeout=timeout_s,
            raise_on_error=True,
        )
        if not gold:
            return {"acc": 0.0, "unparsed_gold": 1, "unparsed_answer": int(not pred), "verifier_timeout": 0}
        if not pred:
            return {"acc": 0.0, "unparsed_gold": 0, "unparsed_answer": 1, "verifier_timeout": 0}
        correct = verify(gold, pred, strict=True, timeout_seconds=timeout_s, raise_on_error=True)
        return {"acc": float(correct), "unparsed_gold": 0, "unparsed_answer": 0, "verifier_timeout": 0}
    except TimeoutException:
        return {"acc": 0.0, "unparsed_gold": 0, "unparsed_answer": 0, "verifier_timeout": 1}
    except Exception:
        # A malformed generated answer is an ungradable sample, not an actor
        # failure. Dependency imports remain outside this handler and fail loudly.
        return {
            "acc": 0.0,
            "unparsed_gold": int(stage == "gold"),
            "unparsed_answer": int(stage != "gold"),
            "verifier_timeout": 0,
        }


def _shutdown_pool():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _get_pool():
    global _pool
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
            atexit.register(_shutdown_pool)
    return _pool


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    require_think_close=True,
    verification_timeout_s=2,
    max_answer_chars=4096,
    **kwargs,
):
    """Dataset-specific callback supported by the V1 asynchronous reward manager."""
    if data_source not in MATH_SOURCES:
        from verl.utils.reward_score import default_compute_score

        return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("MATH requires a nonempty string ground truth")
    if len(ground_truth) > max_answer_chars:
        raise ValueError("Ground truth exceeds configured mathematical answer limit")
    if not isinstance(verification_timeout_s, int) or verification_timeout_s < 1:
        raise ValueError("verification_timeout_s must be a positive integer")
    answer, reason = extract_final_answer(solution_str, require_think_close, max_answer_chars)
    result = {
        "score": 0.0,
        "acc": 0.0,
        "format_ok": int(answer is not None),
        "unfinished_thinking": int(reason == "unfinished_thinking"),
        "unparsed_gold": 0,
        "unparsed_answer": 0,
        "verifier_timeout": 0,
    }
    if answer is None:
        return result
    problem = extra_info.get("problem") if isinstance(extra_info, dict) else None
    answer = resolve_choice_answer(answer, problem)
    ground_truth = resolve_choice_answer(ground_truth, problem)
    # Canonical package import also makes the worker pickleable if this callback
    # is loaded through an external-module path by an older launcher.
    from verl.experimental.trajectory_async.reasoning_math_reward import verify_answer as worker

    loop = asyncio.get_running_loop()
    gate = _gates.setdefault(loop, asyncio.Semaphore(2))
    async with gate:
        future = loop.run_in_executor(_get_pool(), worker, answer, ground_truth, verification_timeout_s)
        try:
            # A spawned Ray/verl interpreter can take tens of seconds to import
            # torch on this host. Parsing/verification still have their own short
            # main-thread timeouts; the outer watchdog includes cold startup.
            verified = await asyncio.wait_for(future, timeout=90 + 3 * verification_timeout_s)
        except TimeoutError:
            result["verifier_timeout"] = 1
            return result
    result.update(verified)
    result["score"] = result["acc"]
    return result
