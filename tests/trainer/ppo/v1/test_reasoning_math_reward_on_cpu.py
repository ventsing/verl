"""Regression cases from real DeepScaleR rollouts and incorrect near matches."""

import asyncio

import pytest

from verl.experimental.trajectory_async.reasoning_math_reward import compute_score, extract_final_answer, verify_answer


@pytest.mark.parametrize(
    "prediction,gold,correct",
    [
        (r"[-1/3, 1]", r"[-\tfrac13, 1]", True),
        (r"(-1/3, 1]", r"[-\tfrac13, 1]", False),
        (r"[-1/3, 1)", r"[-\tfrac13, 1]", False),
        ("1/2", r"\frac12", True),
        ("2/4", "0.5", True),
        ("50", "0.5", False),
        ("-1/3", r"-\frac{1}{3}", True),
        ("1/3", r"-\frac{1}{3}", False),
        (r"\sqrt{8}/2", r"\sqrt{2}", True),
        ("x+x", "2x", True),
        (r"\begin{pmatrix}4\\0\end{pmatrix}", r"\begin{pmatrix}4\\0\end{pmatrix}", True),
        (r"\begin{pmatrix}0\\4\end{pmatrix}", r"\begin{pmatrix}4\\0\end{pmatrix}", False),
        ("60\\%", "60", True),
        ("24.14\\%", "24.14\\%", True),
        ("0.6", "60", False),
        (r"f(x)=x-1 \text{ and } f(x)=-x-1", r"f(x)=x-1 \text{ or } f(x)=-x-1.", True),
        (r"a=-1 \text{ or } a=-2", "-1", False),
        (r"a=-1 \text{ or } a=-2", r"\{-2,-1\}", True),
    ],
)
def test_equivalent_math_and_incorrect_near_matches(prediction, gold, correct):
    assert bool(verify_answer(prediction, gold)["acc"]) == correct


def test_only_last_final_box_is_graded_and_reasoning_cannot_win_reward():
    trace = r"Candidate \boxed{42}." + "</think>" + r"Final \boxed{41}"
    assert extract_final_answer(trace)[0] == "41"
    assert asyncio.run(compute_score("lighteval/MATH", trace, "42"))["score"] == 0


@pytest.mark.parametrize(
    "trace", [r"Thinking \boxed{42}", "Still checking.", "</think>No answer", "</think></think>" + r"\boxed{42}"]
)
def test_incomplete_or_invalid_format_does_not_receive_correctness(trace):
    result = asyncio.run(compute_score("lighteval/MATH", trace, "42"))
    assert result["score"] == 0
    assert not result["format_ok"]


def test_callback_subprocess_works_from_async_reward_thread():
    result = asyncio.run(compute_score("lighteval/MATH", "</think>" + r"\boxed{[-1/3,1]}", r"[-\tfrac13,1]"))
    assert result["score"] == result["acc"] == 1
    assert result["verifier_timeout"] == 0


def test_non_thinking_format_is_explicit_and_other_datasets_keep_existing_reward():
    assert extract_final_answer(r"\boxed{42}", require_think_close=False) == ("42", "ok")
    assert asyncio.run(compute_score("openai/gsm8k", "#### 42", "42")) == 1


def test_oversized_answers_and_empty_gold_are_not_silently_accepted():
    assert extract_final_answer("</think>\\boxed{" + "x" * 4097 + "}")[0] is None
    with pytest.raises(ValueError, match="ground truth"):
        asyncio.run(compute_score("lighteval/MATH", "</think>\\boxed{1}", ""))


def test_audited_dataset_reference_is_corrected_without_mutating_original():
    from prepare_reasoning_math_data import PROBLEM_SHA, correct_reference

    row = {"extra_info": {"problem_sha256": PROBLEM_SHA}, "reward_model": {"ground_truth": "-1"}}
    corrected, changed = correct_reference(row)
    assert changed
    assert corrected["reward_model"]["ground_truth"] == r"\{-2,-1\}"
    assert row["reward_model"]["ground_truth"] == "-1"
    assert not correct_reference({"extra_info": {}, "reward_model": {"ground_truth": "-1"}})[1]


@pytest.mark.parametrize("answer", [r"\frac{}{2}", r"\sqrt{", r"\begin{pmatrix}1&\end{pmatrix}"])
def test_malformed_latex_is_incorrect_without_killing_reward_worker(answer):
    assert verify_answer(answer, "918273645")["acc"] == 0
