from __future__ import annotations

import json
from pathlib import Path

from rag_learning.models import Chunk, RetrievedChunk
from rag_learning.retrievers import BM25Retriever

INDEX_VERSION = 1


class JsonBM25Store:
    def __init__(
        self,
        path: Path | str,
        require_existing: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.path = Path(path)
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._retriever = BM25Retriever([], k1=self.k1, b=self.b)
        self._load(require_existing= require_existing)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._retriever = BM25Retriever(self._chunks, k1=self.k1, b=self.b)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._retriever.search(query, top_k=top_k)

    def _load(self, require_existing: bool) -> None:
        if not self.path.exists():
            if require_existing:
                raise FileNotFoundError(
                    f"BM25 index does not exist: {self.path}. Run: uv run rag ingest data/knowledge"
                )
            return

        data = json.loads(self.path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != INDEX_VERSION:
            raise ValueError(f"Unsupported BM25 index version: {data.get('version')}")
        self.k1 = float(data.get("k1", self.k1))
        self.b = float(data.get("b", self.b))
        self._chunks = [
            Chunk(
                source=Path(item["source"]),
                index=int(item["index"]),
                text=str(item["text"]),
            )
            for item in data.get("chunks", [])
        ]
        self._retriever = BM25Retriever(self._chunks, k1=self.k1, b=self.b)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": INDEX_VERSION,
            "k1": self.k1,
            "b": self.b,
            "chunks": [
                {
                    "source": str(chunk.source),
                    "index": chunk.index,
                    "text": chunk.text,
                }
                for chunk in self._chunks
            ],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
