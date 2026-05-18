from pathlib import Path

from rag_learning.models import Document
from rag_learning.splitter import split_documents, split_text


def test_split_text_uses_overlap_between_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text(text, chunk_size=10, overlap=3)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_split_documents_keeps_source_metadata() -> None:
    document = Document(source=Path("guide.md"), text="abcdefghijklmnopqrstuvwxyz")

    chunks = split_documents([document], chunk_size=10, overlap=3)

    assert chunks[0].source == Path("guide.md")
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[1].text == "hijklmnopq"
