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


def test_ask_command_can_show_prompt_without_calling_llm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag.md").write_text(
        "RAG means retrieval augmented generation.",
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    ask_result = runner.invoke(app, ["ask", "What is RAG?", "--top-k", "1", "--show-prompt"])

    assert ask_result.exit_code == 0
    assert "System prompt:" in ask_result.output
    assert "User prompt:" in ask_result.output
    assert "Question:\nWhat is RAG?" in ask_result.output
    assert "[rag.md#chunk-0]" in ask_result.output
    assert "RAG means retrieval augmented generation." in ask_result.output


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


def test_eval_command_prints_summary_and_details(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag-intro.md").write_text(
        "RAG 是检索增强生成，也就是 retrieval augmented generation.",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "data" / "eval"
    eval_dir.mkdir(parents=True)
    eval_file = eval_dir / "questions.jsonl"
    eval_file.write_text(
        '{"question":"什么是 RAG？","expected_answer":"检索增强生成",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    eval_result = runner.invoke(app, ["eval", str(eval_file), "--top-k", "1"])

    assert eval_result.exit_code == 0
    assert "Evaluation Summary" in eval_result.output
    assert "retrieval_hit_rate" in eval_result.output
    assert "answer_contains_expected_rate" in eval_result.output
    assert "rag-intro.md#chunk-0" in eval_result.output


def test_eval_command_can_use_bm25_retriever_without_ingest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag-intro.md").write_text(
        "BM25 适合精确术语、编号和关键词检索。",
        encoding="utf-8",
    )
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"BM25 适合什么？","expected_answer":"精确术语",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            str(eval_file),
            "--retriever",
            "bm25",
            "--knowledge-path",
            str(knowledge_dir),
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "retriever" in result.output
    assert "bm25" in result.output
    assert "recall_at_k" in result.output
    assert "precision_at_k" in result.output
    assert "mrr" in result.output
    assert "rag-intro.md#chunk-0" in result.output


def test_eval_command_can_use_vector_retriever_explicitly(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag-intro.md").write_text(
        "向量检索适合语义相似和自然语言改写。",
        encoding="utf-8",
    )
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"向量检索适合什么？","expected_answer":"语义相似",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    result = runner.invoke(
        app,
        ["eval", str(eval_file), "--retriever", "vector", "--top-k", "1"],
    )

    assert result.exit_code == 0
    assert "vector" in result.output
    assert "recall_at_k" in result.output


def test_eval_command_can_use_hybrid_retriever(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag-intro.md").write_text(
        "Hybrid Search 使用 RRF 融合 BM25 关键词检索和向量检索。",
        encoding="utf-8",
    )
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"Hybrid Search 使用什么融合？","expected_answer":"RRF",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "eval",
            str(eval_file),
            "--retriever",
            "hybrid",
            "--knowledge-path",
            str(knowledge_dir),
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "hybrid" in result.output
    assert "mrr" in result.output


def test_eval_command_reports_invalid_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text('{"question":"broken"\n', encoding="utf-8")

    result = runner.invoke(app, ["eval", str(eval_file)])

    assert result.exit_code != 0
    assert "line 1" in result.output
