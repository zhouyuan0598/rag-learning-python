from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: Path
    text: str


@dataclass(frozen=True)
class Chunk:
    source: Path
    index: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source.name}#chunk-{self.index}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
