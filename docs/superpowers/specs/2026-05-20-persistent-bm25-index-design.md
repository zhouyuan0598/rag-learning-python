# Persistent BM25 Index Design

## Goal

Make BM25 behave more like a production retrieval component: build the BM25 index during `rag ingest`, persist it locally, and load that persisted index during `rag eval --retriever bm25|hybrid`. Query-time evaluation should no longer scan `data/knowledge` and rebuild BM25 directly from source documents by default.

## Scope

This iteration adds a local JSON BM25 store. It does not introduce Elasticsearch, OpenSearch, PostgreSQL, Vespa, or another external database. The project is still a learning project, so the goal is to teach the production pattern:

```text
ingest time: build and persist index
query time: load persisted index and search
```

## Storage Design

Add:

```text
src/rag_learning/bm25_store.py
```

Default file:

```text
data/storage/bm25-index.json
```

The JSON file stores:

- `version`
- BM25 parameters `k1` and `b`
- chunk source, index, and text

On load, `JsonBM25Store` rebuilds the lightweight in-memory BM25 scoring structures from the persisted chunk records. It does not read the original knowledge documents. This keeps the format simple and stable while preserving the key production shape: source document parsing and chunking happen during ingest, not every query.

## CLI Design

`rag ingest data/knowledge` should write both:

```text
data/storage/hash-store.json
data/storage/bm25-index.json
```

Add an optional `--bm25-storage-path` to `ingest` and `eval`, defaulting to `data/storage/bm25-index.json`.

`rag eval --retriever bm25` should load `data/storage/bm25-index.json` by default. `rag eval --retriever hybrid` should combine the vector store with that persisted BM25 store.

Keep `--knowledge-path`, `--chunk-size`, and `--overlap` only as an explicit fallback path for users who want to build a temporary BM25 retriever during eval. Query-time fallback should be opt-in through a boolean flag:

```bash
uv run rag eval data/eval/questions.jsonl --retriever bm25 --build-bm25-from-knowledge
```

## Error Handling

If `--retriever bm25` or `--retriever hybrid` is used and the BM25 index file does not exist, fail clearly:

```text
BM25 index does not exist: data/storage/bm25-index.json. Run: uv run rag ingest data/knowledge
```

If `--build-bm25-from-knowledge` is passed, keep the existing behavior of loading `--knowledge-path` and building an in-memory `BM25Retriever`.

## Testing

Use TDD.

Tests should cover:

- `JsonBM25Store.add()` persists chunks and `JsonBM25Store.search()` works after reloading from disk.
- Missing BM25 index raises a clear `FileNotFoundError`.
- `rag ingest` writes a BM25 index file.
- `rag eval --retriever bm25` works after ingest without reading `--knowledge-path`.
- `rag eval --retriever bm25` reports a clear error before ingest.
- `rag eval --retriever hybrid` uses the persisted BM25 store.

## Verification

The implementation is complete when:

- `uv run pytest` passes.
- `uv run ruff check` passes.
- `uv run rag ingest data/knowledge` writes `data/storage/bm25-index.json`.
- `uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5` succeeds after ingest.
- `uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5` succeeds after ingest.
