from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer: str
    expected_citations: list[str]


def load_evaluation_cases(path: Path | str) -> list[EvaluationCase]:
    eval_path = Path(path)
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation JSON on line {line_number}: {exc.msg}") from exc
        cases.append(_case_from_data(data, line_number))
    return cases


def _case_from_data(data: object, line_number: int) -> EvaluationCase:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid evaluation case on line {line_number}: expected object")
    question = data.get("question")
    expected_answer = data.get("expected_answer")
    expected_citations = data.get("expected_citations", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Invalid evaluation case on line {line_number}: question is required")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        raise ValueError(
            f"Invalid evaluation case on line {line_number}: expected_answer is required"
        )
    if not isinstance(expected_citations, list) or not all(
        isinstance(value, str) for value in expected_citations
    ):
        raise ValueError(
            f"Invalid evaluation case on line {line_number}: expected_citations must be strings"
        )
    return EvaluationCase(
        question=question,
        expected_answer=expected_answer,
        expected_citations=expected_citations,
    )
