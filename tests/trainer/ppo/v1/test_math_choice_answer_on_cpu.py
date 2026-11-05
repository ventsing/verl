"""Actual DeepScaleR option aliases and cases that must remain unresolved."""

import asyncio

import pytest

from verl.experimental.trajectory_async.math_choice_answer import question_choices, resolve_choice_answer
from verl.experimental.trajectory_async.reasoning_math_reward import compute_score, extract_final_answer, verify_answer

BOWLING = (
    "In how many orders can five bowlers receive the prizes?\n"
    r"$\textbf{(A)}\ 10\qquad \textbf{(B)}\ 16\qquad \textbf{(C)}\ 24"
    r"\qquad \textbf{(D)}\ 120\qquad \textbf{(E)}\ \text{none of these}$"
)
COINS = (
    "What is the probability that Amelia wins?\n"
    "A) $\\frac{5}{14}$\nB) $\\frac{9}{14}$\nC) $\\frac{11}{14}$\n"
    "D) $\\frac{13}{14}$\nE) $\\frac{1}{2}$"
)


@pytest.mark.parametrize("answer", ["B", "(B)", r"\text{B}", r"\mathrm{(B)}"])
def test_correct_option_letter_is_equivalent_to_numeric_reference(answer):
    assert resolve_choice_answer(answer, BOWLING) == "16"
    assert verify_answer(resolve_choice_answer(answer, BOWLING), "16")["acc"] == 1


def test_plain_line_fraction_choice_and_incorrect_option():
    assert verify_answer(resolve_choice_answer("B", COINS), r"\frac{9}{14}")["acc"] == 1
    assert verify_answer(resolve_choice_answer("A", COINS), r"\frac{9}{14}")["acc"] == 0


def test_numeric_answers_and_letter_references_remain_supported():
    assert resolve_choice_answer("16", BOWLING) == "16"
    assert resolve_choice_answer(r"\text{E}", "Enter the letter of the graph.") == r"\text{E}"
    assert resolve_choice_answer("B", COINS) == resolve_choice_answer(r"\text{B}", COINS)


@pytest.mark.parametrize(
    "problem",
    [
        "Find the length B of triangle ABC.",
        "Question\nA) 10\nB) 16\nB) 24",
        "Question\nB) 16\nC) 24",
        "Question\nA) 10\nC) 24",
        "Question\nA) 10\nB. 16",
        "Question\nA) 10\nExplain the options.\nB) 16",
        "Question\nA) 10\nB) 16\nNow answer a different question.",
        r"Question $\textbf{(A)} 10\qquad \textbf{(B)} 16}$",
        "Question\nA) 10\nB) 16\n" + r"\textbf{(C)} 24",
        None,
    ],
)
def test_ambiguous_or_missing_option_lists_do_not_create_aliases(problem):
    assert not question_choices(problem)
    assert resolve_choice_answer("B", problem) == "B"


@pytest.mark.parametrize("answer", ["b", "F", "B or C", "B=16", r"B\text{ because }16"])
def test_only_an_entire_unambiguous_option_letter_is_resolved(answer):
    assert resolve_choice_answer(answer, BOWLING) == answer


def test_reasoning_guess_cannot_supply_a_final_option_answer():
    answer, reason = extract_final_answer(r"Thinking \boxed{B}")
    assert answer is None and reason == "unfinished_thinking"
    assert resolve_choice_answer(answer, BOWLING) is None


def test_oversized_source_option_cannot_bypass_the_symbolic_answer_limit():
    assert resolve_choice_answer("B", "Question\nA) 10\nB) " + "x" * 4097) == "B"


@pytest.mark.parametrize(
    "solution,gold,problem,expected",
    [
        (r"</think>\boxed{B}", "16", BOWLING, 1),
        (r"</think>\boxed{A}", "16", BOWLING, 0),
        (r"</think>\boxed{16}", "B", BOWLING, 1),
        (r"</think>\boxed{B}", "16", None, 0),
        (r"Thinking \boxed{B}", "16", BOWLING, 0),
        (r"</think>\boxed{B}", r"\frac{9}{14}", COINS, 1),
    ],
)
def test_production_callback_grades_choice_aliases_without_rewarding_wrong_or_unfinished_answers(
    solution, gold, problem, expected
):
    extra_info = {"problem": problem} if problem is not None else None
    result = asyncio.run(compute_score("lighteval/MATH", solution, gold, extra_info=extra_info))
    assert result["score"] == result["acc"] == expected
    assert result["verifier_timeout"] == 0
