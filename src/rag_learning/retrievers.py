from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from rag_learning.models import Chunk, RetrievedChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._document_tokens = [_tokens(chunk.text) for chunk in self.chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._document_tokens]
        self._document_lengths = [len(tokens) for tokens in self._document_tokens]
        self._average_document_length = _average(self._document_lengths)
        self._document_frequencies = self._build_document_frequencies()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_terms = _tokens(query)
        ranked = [
            RetrievedChunk(chunk=chunk, score=self._score(query_terms, document_index))
            for document_index, chunk in enumerate(self.chunks)
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]

    def _build_document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self._document_tokens:
            frequencies.update(set(tokens))
        return frequencies

    def _score(self, query_terms: list[str], document_index: int) -> float:
        score = 0.0
        term_frequency = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        for term in query_terms:
            frequency = term_frequency[term]
            if frequency == 0:
                continue
            idf = self._idf(term)
            denominator = frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / max(self._average_document_length, 1.0)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator
        return score

    def _idf(self, term: str) -> float:
        document_count = len(self.chunks)
        document_frequency = self._document_frequencies[term]
        return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))


class HybridRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
    ) -> None:
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_count = max(top_k * 4, top_k)
        chunks_by_citation: dict[str, Chunk] = {}
        scores_by_citation: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, result in enumerate(retriever.search(query, top_k=candidate_count), 1):
                citation = result.chunk.citation
                chunks_by_citation.setdefault(citation, result.chunk)
                scores_by_citation[citation] = scores_by_citation.get(citation, 0.0) + (
                    1 / (self.rrf_k + rank)
                )

        ranked = [
            RetrievedChunk(chunk=chunks_by_citation[citation], score=score)
            for citation, score in scores_by_citation.items()
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
