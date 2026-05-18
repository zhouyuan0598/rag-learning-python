from pathlib import Path

from rag_learning.embeddings import HashEmbeddingModel
from rag_learning.llm_client import OfflineContextLLM
from rag_learning.models import Chunk
from rag_learning.rag_pipeline import RagPipeline
from rag_learning.vector_store import InMemoryVectorStore


def test_rag_pipeline_builds_answer_with_citations() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add(
        [
            Chunk(
                source=Path("rag.md"),
                index=0,
                text="RAG means retrieval augmented generation.",
            )
        ]
    )
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    answer = pipeline.answer("What is RAG?", top_k=1)

    assert "rag.md#chunk-0" in answer
    assert "RAG means retrieval augmented generation." in answer
