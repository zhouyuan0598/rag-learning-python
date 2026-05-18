from __future__ import annotations

from collections.abc import Iterable

from rag_learning.models import Chunk, Document


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, overlap)):
            chunks.append(Chunk(source=document.source, index=index, text=text))
    return chunks
