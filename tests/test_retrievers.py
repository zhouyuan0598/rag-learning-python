from pathlib import Path

import pytest

from rag_learning.models import Chunk
from rag_learning.retrievers import BM25Retriever


def test_bm25_ranks_keyword_match_above_unrelated_chunk() -> None:
    retriever = BM25Retriever(
        [
            Chunk(source=Path("rag.md"), index=0, text="BM25 适合精确术语和关键词检索。"),
            Chunk(source=Path("rag.md"), index=1, text="向量检索适合语义相似和自然语言改写。"),
        ]
    )

    results = retriever.search("BM25 关键词", top_k=2)

    assert [result.chunk.citation for result in results] == [
        "rag.md#chunk-0",
        "rag.md#chunk-1",
    ]
    assert results[0].score > results[1].score


def test_bm25_rejects_non_positive_top_k() -> None:
    retriever = BM25Retriever([Chunk(source=Path("rag.md"), index=0, text="BM25")])

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("BM25", top_k=0)
