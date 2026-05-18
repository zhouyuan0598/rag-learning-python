from __future__ import annotations

from pathlib import Path

from rag_learning.document_loader import load_documents
from rag_learning.llm_client import LLMClient
from rag_learning.models import RetrievedChunk
from rag_learning.splitter import split_documents
from rag_learning.vector_store import ChromaVectorStore, InMemoryVectorStore, JsonVectorStore

VectorStore = InMemoryVectorStore | JsonVectorStore | ChromaVectorStore


class RagPipeline:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> None:
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_path(self, path: Path | str) -> int:
        documents = load_documents(path)
        chunks = split_documents(documents, chunk_size=self.chunk_size, overlap=self.overlap)
        self.store.add(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.store.search(query, top_k=top_k)

    def answer(self, question: str, top_k: int = 5) -> str:
        contexts = self.search(question, top_k=top_k)
        return self.llm.generate(question, contexts)
