from pathlib import Path

from typer.testing import CliRunner

from rag_learning.cli import (
    _create_embedding_model,
    _resolve_collection_name,
    _resolve_storage_path,
    app,
)

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


def test_resolve_storage_path_uses_directory_default_for_chroma() -> None:
    assert _resolve_storage_path("json", None) == Path("data/storage/hash-store.json")
    assert _resolve_storage_path("chroma", None) == Path("data/storage/chroma")


def test_create_sentence_transformer_embedding_uses_cli_model_name() -> None:
    embedding = _create_embedding_model("sentence-transformer", "sentence-transformers/test-model")

    assert embedding.model_name == "sentence-transformers/test-model"


def test_resolve_collection_name_separates_embedding_backends() -> None:
    assert _resolve_collection_name(None, "json", "hash", None) == "rag_learning"
    assert _resolve_collection_name(None, "chroma", "hash", None) == "rag_learning_hash"
    assert (
        _resolve_collection_name(
            None,
            "chroma",
            "sentence-transformer",
            "models/bge-small-zh-v1.5",
        )
        == "rag_learning_sentence_transformer_models_bge_small_zh_v1_5"
    )
    assert (
        _resolve_collection_name("custom_collection", "chroma", "sentence-transformer", None)
        == "custom_collection"
    )
