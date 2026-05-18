from pathlib import Path

from typer.testing import CliRunner

from rag_learning.cli import app

runner = CliRunner()


def test_ask_command_preserves_source_citation_markup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag.md").write_text(
        "RAG means retrieval augmented generation.",
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    ask_result = runner.invoke(app, ["ask", "What is RAG?", "--top-k", "1"])

    assert ask_result.exit_code == 0
    assert "[rag.md#chunk-0]" in ask_result.output
