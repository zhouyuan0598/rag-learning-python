from pathlib import Path

from rag_learning.document_loader import load_documents
from rag_learning.evaluation import load_evaluation_cases
from rag_learning.splitter import split_documents

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT / "data" / "knowledge"
EVAL_PATH = ROOT / "data" / "eval" / "questions.jsonl"


def test_sample_knowledge_base_is_large_enough_for_retrieval_experiments() -> None:
    documents = load_documents(KNOWLEDGE_PATH)
    chunks = split_documents(documents)

    assert len(documents) >= 7
    assert len(chunks) >= 25
    assert len({chunk.citation for chunk in chunks}) == len(chunks)


def test_sample_eval_cases_are_large_and_reference_existing_chunks() -> None:
    chunks = split_documents(load_documents(KNOWLEDGE_PATH))
    existing_citations = {chunk.citation for chunk in chunks}
    cases = load_evaluation_cases(EVAL_PATH)

    assert len(cases) >= 60
    assert len({case.question for case in cases}) == len(cases)
    assert any(len(case.expected_citations) > 1 for case in cases)

    for case in cases:
        assert case.expected_citations
        assert set(case.expected_citations) <= existing_citations
