from pathlib import Path

import pytest

from rag_learning.embeddings import HashEmbeddingModel
from rag_learning.evaluation import EvaluationCase, evaluate_cases, load_evaluation_cases
from rag_learning.llm_client import OfflineContextLLM
from rag_learning.models import Chunk
from rag_learning.rag_pipeline import RagPipeline
from rag_learning.vector_store import InMemoryVectorStore


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


def test_evaluate_cases_scores_retrieval_and_answer_quality() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add(
        [
            Chunk(
                source=Path("rag-intro.md"),
                index=0,
                text="RAG 是检索增强生成，也就是 retrieval augmented generation.",
            )
        ]
    )
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="什么是 RAG？",
                expected_answer="检索增强生成",
                expected_citations=["rag-intro.md#chunk-0"],
            )
        ],
        top_k=1,
    )

    assert report.summary.case_count == 1
    assert report.summary.retrieval_hit_rate == 1.0
    assert report.summary.answer_contains_expected_rate == 1.0
    assert report.summary.avg_top_score > 0
    assert report.results[0].retrieval_hit is True
    assert report.results[0].answer_contains_expected is True
    assert report.results[0].top_citation == "rag-intro.md#chunk-0"


def test_evaluate_cases_reports_recall_precision_and_mrr() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add(
        [
            Chunk(source=Path("rag.md"), index=0, text="unrelated filler"),
            Chunk(source=Path("rag.md"), index=1, text="BM25 关键词检索"),
            Chunk(source=Path("rag.md"), index=2, text="Hybrid Search 混合检索"),
        ]
    )
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="BM25",
                expected_answer="BM25",
                expected_citations=["rag.md#chunk-1", "rag.md#chunk-2"],
            )
        ],
        top_k=2,
    )

    assert report.results[0].recall_at_k == 0.5
    assert report.results[0].precision_at_k == 0.5
    assert report.results[0].first_relevant_rank == 1
    assert report.results[0].reciprocal_rank == 1.0
    assert report.summary.recall_at_k == 0.5
    assert report.summary.precision_at_k == 0.5
    assert report.summary.mrr == 1.0


def test_evaluate_cases_excludes_rows_without_expected_citations_from_hit_rate() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add([Chunk(source=Path("rag-intro.md"), index=0, text="RAG 是检索增强生成。")])
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="什么是 RAG？",
                expected_answer="检索增强生成",
                expected_citations=[],
            )
        ],
        top_k=1,
    )

    assert report.summary.retrieval_hit_rate == 0.0
    assert report.summary.recall_at_k == 0.0
    assert report.summary.precision_at_k == 0.0
    assert report.summary.mrr == 0.0
    assert report.results[0].retrieval_hit is None
