# RAG Learning Python Design

## Goal

Build a small Python RAG learning project under `/Users/zhouyuan/dev/java/workspace/rag-learning-python`.
The first version teaches the core RAG flow without hiding it behind a large framework:
load documents, split chunks, embed text, store vectors, retrieve matches, and generate an answer with citations.

## Technical Stack

- Python 3.12 managed by `uv`
- `Typer` and `Rich` for CLI commands and readable terminal output
- A deterministic hash embedding backend for zero-setup learning and tests
- Optional `sentence-transformers` backend for real local embeddings
- Optional `Chroma` backend for persistent local vector storage
- Optional Claude / Anthropic client for LLM generation
- `pytest` for tests and `ruff` for linting and formatting

## Project Shape

The project is CLI-first. The commands are:

- `rag ingest data/knowledge`
- `rag search "什么是 RAG"`
- `rag ask "RAG 和微调有什么区别"`

The implementation keeps each RAG stage in its own module so the learning path stays visible.
The default path works offline with hash embeddings and an offline answer formatter.
The realistic path can be enabled with optional extras for Chroma, sentence-transformers, and Claude APIs.

## Data Flow

1. Load Markdown and TXT files from a knowledge directory.
2. Split document text into overlapping chunks.
3. Embed each chunk.
4. Add chunk vectors to a vector store.
5. Embed the user query.
6. Retrieve the most similar chunks.
7. Build a context prompt and return an answer with source citations.

## Testing

Unit tests cover document loading, chunk splitting, in-memory retrieval, and answer citation assembly.
Heavy optional dependencies are lazy-loaded so tests stay fast and the base project remains easy to run.
