# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Resolve a final option letter only against an explicit source-question list."""

import re

_PLAIN_OPTION = re.compile(r"^[ \t]*([A-E])([).])[ \t]+([^\n]+)$", re.MULTILINE)
_LATEX_OPTION = re.compile(r"\\textbf\{\(([A-E])\)\s*\}")
_SPACING = r"(?:\s|~|\\(?:qquad|quad|[,;:! ]))"


def _valid_labels(labels):
    return 2 <= len(labels) <= 5 and labels == list("ABCDE"[: len(labels)])


def _clean_value(value):
    value = value.strip().strip("$").strip()
    value = re.sub(rf"^{_SPACING}+|{_SPACING}+$", "", value).strip()
    if len(value) > 4096:
        return None
    # Unsupported nested formatting stays unresolved rather than becoming a
    # guessed alias. In particular, do not extract a numeric substring here.
    balance = 0
    for char in value:
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
        if balance < 0:
            return None
    return value if value and balance == 0 else None


def question_choices(problem):
    """Read a complete trailing plain-line or LaTeX-bold option block.

    Only source-question text is accepted. Unrecognized, duplicate, reordered,
    incomplete, or mixed-style lists return no aliases.
    """
    if not isinstance(problem, str):
        return {}
    plain, latex = list(_PLAIN_OPTION.finditer(problem)), list(_LATEX_OPTION.finditer(problem))
    if plain and latex:
        return {}
    if plain:
        if not _valid_labels([match[1] for match in plain]) or len({match[2] for match in plain}) != 1:
            return {}
        if problem[plain[-1].end() :].strip():
            return {}
        if any(problem[left.end() : right.start()].strip() for left, right in zip(plain[:-1], plain[1:], strict=True)):
            return {}
        values = [_clean_value(match[3]) for match in plain]
        labels = [match[1] for match in plain]
    elif latex:
        labels = [match[1] for match in latex]
        if not _valid_labels(labels):
            return {}
        values = [
            _clean_value(problem[match.end() : latex[i + 1].start() if i + 1 < len(latex) else len(problem)])
            for i, match in enumerate(latex)
        ]
    else:
        return {}
    return dict(zip(labels, values, strict=True)) if all(values) else {}


def resolve_choice_answer(answer, problem):
    """Leave mathematical answers unchanged unless they are a listed letter."""
    if not isinstance(answer, str):
        return answer
    letter = answer.strip()
    wrapped = re.fullmatch(r"\\(?:text|mathrm|mathbf|textbf)\{\s*(.*?)\s*\}", letter)
    if wrapped:
        letter = wrapped[1].strip()
    if re.fullmatch(r"\([A-E]\)", letter):
        letter = letter[1]
    if not re.fullmatch(r"[A-E]", letter):
        return answer
    return question_choices(problem).get(letter, answer)
