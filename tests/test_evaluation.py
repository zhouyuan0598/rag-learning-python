from pathlib import Path

import pytest

from rag_learning.evaluation import EvaluationCase, load_evaluation_cases


def test_load_evaluation_cases_reads_jsonl(tmp_path: Path) -> None:
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"什么是 RAG？","expected_answer":"检索增强生成",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(eval_file)

    assert cases == [
        EvaluationCase(
            question="什么是 RAG？",
            expected_answer="检索增强生成",
            expected_citations=["rag-intro.md#chunk-0"],
        )
    ]


def test_load_evaluation_cases_reports_invalid_json_line(tmp_path: Path) -> None:
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"ok","expected_answer":"ok"}\n'
        '{"question": "broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_evaluation_cases(eval_file)
