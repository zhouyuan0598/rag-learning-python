from __future__ import annotations

import json
import math
from pathlib import Path

from rag_learning.embeddings import EmbeddingModel
from rag_learning.models import Chunk, RetrievedChunk


class InMemoryVectorStore:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._records: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk]) -> None:
        embeddings = self.embedding_model.embed_documents([chunk.text for chunk in chunks])
        self._records.extend(zip(chunks, embeddings, strict=True))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_embedding = self.embedding_model.embed_query(query)
        ranked = [
            RetrievedChunk(chunk=chunk, score=_cosine_similarity(query_embedding, embedding))
            for chunk, embedding in self._records
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]


class JsonVectorStore(InMemoryVectorStore):
    def __init__(self, path: Path | str, embedding_model: EmbeddingModel) -> None:
        super().__init__(embedding_model)
        self.path = Path(path)
        self._load()

    def add(self, chunks: list[Chunk]) -> None:
        self._records = []
        super().add(chunks)
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return

        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = [
            (
                Chunk(
                    source=Path(item["source"]),
                    index=int(item["index"]),
                    text=str(item["text"]),
                ),
                [float(value) for value in item["embedding"]],
            )
            for item in data
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "source": str(chunk.source),
                "index": chunk.index,
                "text": chunk.text,
                "embedding": embedding,
            }
            for chunk, embedding in self._records
        ]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class ChromaVectorStore:
    def __init__(
        self,
        path: Path | str,
        collection_name: str,
        embedding_model: EmbeddingModel,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Run: uv sync --extra local --group dev"
            ) from exc

        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embeddings = self.embedding_model.embed_documents([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {"source": str(chunk.source), "index": chunk.index, "citation": chunk.citation}
                for chunk in chunks
            ],
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        result = self.collection.query(
            query_embeddings=[self.embedding_model.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=True):
            chunk = Chunk(
                source=Path(str(metadata["source"])),
                index=int(metadata["index"]),
                text=str(document),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source}:{chunk.index}"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
