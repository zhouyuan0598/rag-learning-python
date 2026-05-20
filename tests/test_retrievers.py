from pathlib import Path

import pytest

from rag_learning.models import Chunk
from rag_learning.models import RetrievedChunk
from rag_learning.retrievers import BM25Retriever, HybridRetriever


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


class StaticRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.results[:top_k]


def test_hybrid_retriever_fuses_rankings_with_rrf() -> None:
    chunk_a = Chunk(source=Path("rag.md"), index=0, text="BM25 keyword match")
    chunk_b = Chunk(source=Path("rag.md"), index=1, text="Vector semantic match")
    chunk_c = Chunk(source=Path("rag.md"), index=2, text="Shared strong match")

    hybrid = HybridRetriever(
        [
            StaticRetriever(
                [
                    RetrievedChunk(chunk=chunk_c, score=9.0),
                    RetrievedChunk(chunk=chunk_a, score=8.0),
                ]
            ),
            StaticRetriever(
                [
                    RetrievedChunk(chunk=chunk_b, score=0.9),
                    RetrievedChunk(chunk=chunk_c, score=0.8),
                ]
            ),
        ],
        rrf_k=60,
    )

    results = hybrid.search("query", top_k=2)

    assert [result.chunk.citation for result in results] == [
        "rag.md#chunk-2",
        "rag.md#chunk-1",
    ]
    assert results[0].score > results[1].score
