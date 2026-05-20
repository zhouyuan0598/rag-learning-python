from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from rag_learning.bm25_store import JsonBM25Store
from rag_learning.document_loader import load_documents
from rag_learning.embeddings import EmbeddingModel, HashEmbeddingModel, SentenceTransformerEmbedding
from rag_learning.evaluation import EvaluationReport, evaluate_cases, load_evaluation_cases
from rag_learning.llm_client import SYSTEM_PROMPT, ClaudeLLM, OfflineContextLLM, build_prompt
from rag_learning.rag_pipeline import RagPipeline
from rag_learning.retrievers import BM25Retriever, HybridRetriever, Retriever
from rag_learning.splitter import split_documents
from rag_learning.vector_store import ChromaVectorStore, JsonVectorStore

app = typer.Typer(help="RAG learning CLI")
console = Console()

EmbeddingBackend = Literal["hash", "sentence-transformer"]
StoreBackend = Literal["json", "chroma"]
LLMBackend = Literal["offline", "claude"]
RetrieverBackend = Literal["vector", "bm25", "hybrid"]
DEFAULT_JSON_STORAGE_PATH = Path("data/storage/hash-store.json")
DEFAULT_CHROMA_STORAGE_PATH = Path("data/storage/chroma")
DEFAULT_BM25_STORAGE_PATH = Path("data/storage/bm25-index.json")
DEFAULT_COLLECTION_NAME = "rag_learning"
DEFAULT_KNOWLEDGE_PATH = Path("data/knowledge")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Knowledge file or directory to ingest")],
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    bm25_storage_path: Annotated[Path, typer.Option("--bm25-storage-path")] = (
        DEFAULT_BM25_STORAGE_PATH
    ),
    collection: Annotated[str | None, typer.Option("--collection")] = None,
    chunk_size: Annotated[int, typer.Option("--chunk-size")] = 800,
    overlap: Annotated[int, typer.Option("--overlap")] = 100,
) -> None:
    chunks = split_documents(
        load_documents(path),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    _create_store(
        store_backend, storage_path, collection, embedding_backend, embedding_model
    ).add(chunks)
    JsonBM25Store(bm25_storage_path).add(chunks)
    console.print(f"Ingested {len(chunks)} chunks from {path}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
) -> None:
    pipeline = RagPipeline(
        store=_create_store(
            store_backend, storage_path, collection, embedding_backend, embedding_model
        ),
        llm=OfflineContextLLM(),
    )
    results = pipeline.search(query, top_k=top_k)
    _print_results(results)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    llm_backend: Annotated[LLMBackend, typer.Option("--llm-backend")] = "offline",
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
    show_prompt: Annotated[bool, typer.Option("--show-prompt")] = False,
) -> None:
    pipeline = RagPipeline(
        store=_create_store(
            store_backend, storage_path, collection, embedding_backend, embedding_model
        ),
        llm=_create_llm(llm_backend),
    )
    if show_prompt:
        contexts = pipeline.search(question, top_k=top_k)
        console.print("System prompt:", style="bold")
        console.print(SYSTEM_PROMPT, markup=False)
        console.print("\nUser prompt:", style="bold")
        console.print(build_prompt(question, contexts), markup=False)
        return

    console.print(pipeline.answer(question, top_k=top_k), markup=False)


@app.command("eval")
def eval_command(
    path: Annotated[Path, typer.Argument(help="JSONL evaluation file")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    retriever: Annotated[RetrieverBackend, typer.Option("--retriever")] = "vector",
    llm_backend: Annotated[LLMBackend, typer.Option("--llm-backend")] = "offline",
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
    knowledge_path: Annotated[Path, typer.Option("--knowledge-path")] = DEFAULT_KNOWLEDGE_PATH,
    chunk_size: Annotated[int, typer.Option("--chunk-size")] = 800,
    overlap: Annotated[int, typer.Option("--overlap")] = 100,
) -> None:
    try:
        cases = load_evaluation_cases(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="path") from exc

    try:
        selected_retriever = _create_eval_retriever(
            retriever,
            store_backend,
            storage_path,
            collection,
            embedding_backend,
            embedding_model,
            knowledge_path,
            chunk_size,
            overlap,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="knowledge_path") from exc

    report = evaluate_cases(
        selected_retriever,
        cases,
        top_k=top_k,
        llm=_create_llm(llm_backend),
    )
    _print_evaluation_report(report, retriever)


def _create_store(
    store_backend: StoreBackend,
    storage_path: Path | None,
    collection: str | None,
    embedding_backend: EmbeddingBackend,
    embedding_model_name: str | None,
) -> JsonVectorStore | ChromaVectorStore:
    embedding_model = _create_embedding_model(embedding_backend, embedding_model_name)
    resolved_storage_path = _resolve_storage_path(store_backend, storage_path)
    resolved_collection = _resolve_collection_name(
        collection, store_backend, embedding_backend, embedding_model_name
    )
    if store_backend == "json":
        return JsonVectorStore(resolved_storage_path, embedding_model)
    return ChromaVectorStore(resolved_storage_path, resolved_collection, embedding_model)


def _resolve_storage_path(store_backend: StoreBackend, storage_path: Path | None) -> Path:
    if storage_path is not None:
        return storage_path
    if store_backend == "json":
        return DEFAULT_JSON_STORAGE_PATH
    return DEFAULT_CHROMA_STORAGE_PATH


def _resolve_collection_name(
    collection: str | None,
    store_backend: StoreBackend,
    embedding_backend: EmbeddingBackend,
    embedding_model_name: str | None,
) -> str:
    if collection:
        return collection
    if store_backend == "json":
        return DEFAULT_COLLECTION_NAME

    if embedding_backend == "hash":
        return f"{DEFAULT_COLLECTION_NAME}_hash"

    model_slug = _slugify_collection_part(embedding_model_name or "default_model")
    return f"{DEFAULT_COLLECTION_NAME}_sentence_transformer_{model_slug}"


def _slugify_collection_part(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _create_embedding_model(
    embedding_backend: EmbeddingBackend,
    embedding_model_name: str | None = None,
) -> EmbeddingModel:
    if embedding_backend == "hash":
        return HashEmbeddingModel()
    return SentenceTransformerEmbedding(embedding_model_name)


def _create_llm(llm_backend: LLMBackend) -> OfflineContextLLM | ClaudeLLM:
    if llm_backend == "offline":
        return OfflineContextLLM()
    return ClaudeLLM()


def _create_eval_retriever(
    retriever: RetrieverBackend,
    store_backend: StoreBackend,
    storage_path: Path | None,
    collection: str | None,
    embedding_backend: EmbeddingBackend,
    embedding_model_name: str | None,
    knowledge_path: Path,
    chunk_size: int,
    overlap: int,
) -> Retriever:
    if retriever == "bm25":
        return _create_bm25_retriever(knowledge_path, chunk_size, overlap)

    vector_retriever = _create_store(
        store_backend, storage_path, collection, embedding_backend, embedding_model_name
    )
    if retriever == "vector":
        return vector_retriever
    return HybridRetriever(
        [
            vector_retriever,
            _create_bm25_retriever(knowledge_path, chunk_size, overlap),
        ]
    )


def _create_bm25_retriever(
    knowledge_path: Path,
    chunk_size: int,
    overlap: int,
) -> BM25Retriever:
    documents = load_documents(knowledge_path)
    chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
    return BM25Retriever(chunks)


def _print_results(results: list) -> None:
    table = Table(title="Search Results")
    table.add_column("Score", justify="right")
    table.add_column("Citation")
    table.add_column("Text")

    for result in results:
        table.add_row(f"{result.score:.4f}", result.chunk.citation, result.chunk.text[:180])

    console.print(table)


def _print_evaluation_report(report: EvaluationReport, retriever: str) -> None:
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary = report.summary
    summary_table.add_row("retriever", retriever)
    summary_table.add_row("case_count", str(summary.case_count))
    summary_table.add_row("retrieval_hit_rate", f"{summary.retrieval_hit_rate:.2%}")
    summary_table.add_row("recall_at_k", f"{summary.recall_at_k:.2%}")
    summary_table.add_row("precision_at_k", f"{summary.precision_at_k:.2%}")
    summary_table.add_row("mrr", f"{summary.mrr:.4f}")
    summary_table.add_row(
        "answer_contains_expected_rate", f"{summary.answer_contains_expected_rate:.2%}"
    )
    summary_table.add_row("avg_top_score", f"{summary.avg_top_score:.4f}")
    summary_table.add_row("avg_retrieval_ms", f"{summary.avg_retrieval_ms:.2f}")
    summary_table.add_row("avg_answer_ms", f"{summary.avg_answer_ms:.2f}")
    console.print(summary_table)

    detail_table = Table(title="Evaluation Details")
    detail_table.add_column("Question", overflow="fold")
    detail_table.add_column("Retrieval")
    detail_table.add_column("Answer")
    detail_table.add_column("Citations", no_wrap=True, overflow="ignore")
    detail_table.add_column("Top score", justify="right")
    detail_table.add_column("Timing ms", justify="right")
    for result in report.results:
        detail_table.add_row(
            result.case.question,
            _format_optional_bool(result.retrieval_hit),
            _format_bool(result.answer_contains_expected),
            _format_citations(result.expected_citations, result.top_citation),
            f"{result.top_score:.4f}",
            f"retrieval {result.retrieval_ms:.2f}\nanswer {result.answer_ms:.2f}",
        )
    console.print(detail_table)


def _format_citations(expected_citations: list[str], top_citation: str | None) -> str:
    expected = ", ".join(expected_citations) or "-"
    top = top_citation or "-"
    return f"expected: {expected}\ntop: {top}"


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return _format_bool(value)


def _format_bool(value: bool) -> str:
    return "pass" if value else "fail"
