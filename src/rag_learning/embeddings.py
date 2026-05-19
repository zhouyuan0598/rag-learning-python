from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Protocol

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingModel:
    """Deterministic lightweight embeddings for learning and tests."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            index = _stable_index(token, self.dimension)
            vector[index] += 1.0
        return _normalize(vector)


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-zh-v1.5"


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str | None = None) -> None:
        _load_dotenv_if_available()
        self.model_name = (
            model_name or os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        )
        self._model = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: uv sync --extra local --group dev"
                ) from exc
            self._model = SentenceTransformer(self.model_name)

        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _stable_index(token: str, dimension: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
