from pathlib import Path

from rag_learning.document_loader import load_documents


def test_load_documents_reads_markdown_and_text_files(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# RAG\n\n检索增强生成", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("chunking matters", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_text("not supported in v1", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [doc.source.name for doc in documents] == ["guide.md", "notes.txt"]
    assert documents[0].text == "# RAG\n\n检索增强生成"
    assert documents[1].text == "chunking matters"
