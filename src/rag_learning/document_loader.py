from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rag_learning.models import Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


def load_documents(path: Path | str) -> list[Document]:
    root = Path(path)
    files = _iter_supported_files(root)
    return [Document(source=file, text=file.read_text(encoding="utf-8")) for file in files]


def _iter_supported_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []

    if not path.exists():
        raise FileNotFoundError(f"Knowledge path does not exist: {path}")

    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in file.parts)
    )
