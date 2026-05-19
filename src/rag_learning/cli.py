from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.table import Table

from rag_learning.embeddings import EmbeddingModel, HashEmbeddingModel, SentenceTransformerEmbedding
from rag_learning.llm_client import SYSTEM_PROMPT, ClaudeLLM, OfflineContextLLM, build_prompt
from rag_learning.rag_pipeline import RagPipeline
from rag_learning.vector_store import ChromaVectorStore, JsonVectorStore

app = typer.Typer(help="RAG learning CLI")
console = Console()

EmbeddingBackend = Literal["hash", "sentence-transformer"]
StoreBackend = Literal["json", "chroma"]
LLMBackend = Literal["offline", "claude"]
DEFAULT_JSON_STORAGE_PATH = Path("data/storage/hash-store.json")
DEFAULT_CHROMA_STORAGE_PATH = Path("data/storage/chroma")
DEFAULT_COLLECTION_NAME = "rag_learning"


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Knowledge file or directory to ingest")],
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
    chunk_size: Annotated[int, typer.Option("--chunk-size")] = 800,
    overlap: Annotated[int, typer.Option("--overlap")] = 100,
) -> None:
    pipeline = RagPipeline(
        store=_create_store(
            store_backend, storage_path, collection, embedding_backend, embedding_model
        ),
        llm=OfflineContextLLM(),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    count = pipeline.ingest_path(path)
    console.print(f"Ingested {count} chunks from {path}")


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


def _print_results(results: list) -> None:
    table = Table(title="Search Results")
    table.add_column("Score", justify="right")
    table.add_column("Citation")
    table.add_column("Text")

    for result in results:
        table.add_row(f"{result.score:.4f}", result.chunk.citation, result.chunk.text[:180])

    console.print(table)
