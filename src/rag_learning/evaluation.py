from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from rag_learning.rag_pipeline import RagPipeline


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer: str
    expected_citations: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    retrieval_hit: bool | None
    answer_contains_expected: bool
    expected_citations: list[str]
    retrieved_citations: list[str]
    top_citation: str | None
    top_score: float
    retrieval_ms: float
    answer_ms: float


@dataclass(frozen=True)
class EvaluationSummary:
    case_count: int
    retrieval_hit_rate: float
    answer_contains_expected_rate: float
    avg_top_score: float
    avg_retrieval_ms: float
    avg_answer_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    summary: EvaluationSummary
    results: list[EvaluationResult]


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


def evaluate_cases(
    pipeline: RagPipeline,
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> EvaluationReport:
    results: list[EvaluationResult] = []
    for case in cases:
        retrieval_started = perf_counter()
        retrieved = pipeline.search(case.question, top_k=top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        answer_started = perf_counter()
        answer = pipeline.llm.generate(case.question, retrieved)
        answer_ms = (perf_counter() - answer_started) * 1000

        retrieved_citations = [result.chunk.citation for result in retrieved]
        retrieval_hit = None
        if case.expected_citations:
            retrieval_hit = any(
                citation in retrieved_citations for citation in case.expected_citations
            )

        top_result = retrieved[0] if retrieved else None
        results.append(
            EvaluationResult(
                case=case,
                retrieval_hit=retrieval_hit,
                answer_contains_expected=_contains_expected(answer, case.expected_answer),
                expected_citations=case.expected_citations,
                retrieved_citations=retrieved_citations,
                top_citation=top_result.chunk.citation if top_result else None,
                top_score=top_result.score if top_result else 0.0,
                retrieval_ms=retrieval_ms,
                answer_ms=answer_ms,
            )
        )

    return EvaluationReport(summary=_summarize(results), results=results)


def _contains_expected(answer: str, expected_answer: str) -> bool:
    return _normalize_text(expected_answer) in _normalize_text(answer)


def _normalize_text(value: str) -> str:
    return "".join(value.split()).lower()


def _summarize(results: list[EvaluationResult]) -> EvaluationSummary:
    hit_results = [result for result in results if result.retrieval_hit is not None]
    return EvaluationSummary(
        case_count=len(results),
        retrieval_hit_rate=_rate(
            sum(1 for result in hit_results if result.retrieval_hit), len(hit_results)
        ),
        answer_contains_expected_rate=_rate(
            sum(1 for result in results if result.answer_contains_expected), len(results)
        ),
        avg_top_score=_average([result.top_score for result in results]),
        avg_retrieval_ms=_average([result.retrieval_ms for result in results]),
        avg_answer_ms=_average([result.answer_ms for result in results]),
    )


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
