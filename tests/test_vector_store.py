from pathlib import Path

from rag_learning.embeddings import HashEmbeddingModel
from rag_learning.models import Chunk
from rag_learning.vector_store import InMemoryVectorStore


def test_in_memory_vector_store_returns_most_similar_chunks_first() -> None:
    embedding_model = HashEmbeddingModel(dimension=32)
    store = InMemoryVectorStore(embedding_model)
    chunks = [
        Chunk(source=Path("rag.md"), index=0, text="RAG uses retrieval with generation."),
        Chunk(source=Path("java.md"), index=0, text="Spring Boot starts Java services."),
    ]

    store.add(chunks)
    results = store.search("retrieval augmented generation", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source == Path("rag.md")
    assert results[0].score > 0
