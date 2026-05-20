# Persistent BM25 Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a local BM25 index during `rag ingest` and make BM25/hybrid evaluation load that index by default.

**Architecture:** Add `JsonBM25Store` as a persisted retriever backed by `data/storage/bm25-index.json`. Update CLI ingestion to write both vector and BM25 stores from the same chunks, then update eval retriever creation to load the persisted BM25 store unless an explicit temporary-build flag is passed.

**Tech Stack:** Python 3.12, Typer, JSON storage, existing loader/splitter, existing `BM25Retriever`, pytest, Ruff.

---

### Task 1: JsonBM25Store

**Files:**
- Create: `src/rag_learning/bm25_store.py`
- Create: `tests/test_bm25_store.py`

- [ ] Write a failing test showing chunks can be persisted and searched after reloading the store from disk.
- [ ] Run the test and verify it fails because `JsonBM25Store` does not exist.
- [ ] Implement `JsonBM25Store.add()`, `_load()`, `_save()`, and `search()` by delegating to `BM25Retriever`.
- [ ] Add a failing test for missing index file.
- [ ] Implement clear `FileNotFoundError` handling for required loads.
- [ ] Run `uv run pytest tests/test_bm25_store.py -v`.
- [ ] Commit `feat: persist bm25 index`.

### Task 2: Ingest Writes BM25 Index

**Files:**
- Modify: `src/rag_learning/cli.py`
- Modify: `src/rag_learning/rag_pipeline.py`
- Modify: `tests/test_cli.py`

- [ ] Write a failing CLI test that `rag ingest` creates `data/storage/bm25-index.json`.
- [ ] Refactor ingest so loaded chunks can be written to both vector store and BM25 store without loading documents twice.
- [ ] Add `--bm25-storage-path` with default `data/storage/bm25-index.json`.
- [ ] Run the new CLI test and existing CLI tests.
- [ ] Commit `feat: write bm25 index during ingest`.

### Task 3: Eval Loads Persisted BM25 Index

**Files:**
- Modify: `src/rag_learning/cli.py`
- Modify: `tests/test_cli.py`

- [ ] Write a failing CLI test where `rag eval --retriever bm25` succeeds after ingest even if `--knowledge-path` points at a missing path.
- [ ] Write a failing CLI test where `rag eval --retriever bm25` fails clearly before ingest.
- [ ] Update `_create_eval_retriever()` to load `JsonBM25Store` by default for `bm25` and `hybrid`.
- [ ] Add `--build-bm25-from-knowledge` to preserve the explicit temporary-build path.
- [ ] Run `uv run pytest tests/test_cli.py -v`.
- [ ] Commit `feat: load persisted bm25 index for eval`.

### Task 4: Docs and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/rag-retrieval-theory.md`

- [ ] Document that `rag ingest` writes both vector and BM25 local indexes.
- [ ] Document when `--build-bm25-from-knowledge` is useful.
- [ ] Run `uv run pytest`.
- [ ] Run `uv run ruff check`.
- [ ] Run `uv run rag ingest data/knowledge`.
- [ ] Run `uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5`.
- [ ] Run `uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5`.
- [ ] Commit `docs: document persistent bm25 index`.
- [ ] Push the branch.
