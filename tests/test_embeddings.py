from rag_learning.embeddings import SentenceTransformerEmbedding


def test_sentence_transformer_embedding_reads_model_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "sentence-transformers/test-model")

    embedding = SentenceTransformerEmbedding()

    assert embedding.model_name == "sentence-transformers/test-model"


def test_sentence_transformer_embedding_disables_hf_xet_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    SentenceTransformerEmbedding("sentence-transformers/test-model")

    assert "HF_HUB_DISABLE_XET" in __import__("os").environ
    assert __import__("os").environ["HF_HUB_DISABLE_XET"] == "1"
