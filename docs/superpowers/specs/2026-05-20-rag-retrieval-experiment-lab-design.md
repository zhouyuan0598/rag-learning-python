# RAG Retrieval Experiment Lab Design

## Goal

Upgrade the current RAG evaluation lab into a retrieval experiment lab that can compare `vector`, `bm25`, and `hybrid` retrieval with standard retrieval metrics. The learning goal is to make retrieval quality visible, not just prove that `rag eval` can run.

## Current State

The project already has:

- `rag ingest`: loads and chunks documents into the configured vector store.
- `rag search`: runs vector search through the configured store.
- `rag ask`: retrieves context and generates an answer.
- `rag eval`: reads JSONL evaluation cases and reports basic hit-rate, answer substring rate, score, and latency.
- `data/eval/questions.jsonl`: currently only 2 cases, which is enough for a smoke test but too small for comparing retrievers.

The current evaluator depends on `RagPipeline.search()`, which only exposes the vector-store path. BM25 and hybrid retrieval need a separate retrieval abstraction so they do not get forced into the vector-store interface.

## Scope

This iteration will add:

- A larger learning knowledge base.
- A larger evaluation dataset.
- Standard retrieval metrics: `recall@k`, `precision@k`, and `mrr`.
- A unified retriever interface.
- `vector`, `bm25`, and `hybrid` retriever implementations.
- CLI support for `rag eval --retriever vector|bm25|hybrid`.

This iteration will not add:

- `rag experiment` matrix runs.
- nDCG.
- LLM-as-judge scoring.
- External BM25 libraries.
- Web UI.
- Learned rerankers.

Those are useful later, but this step should stay focused on the retrieval comparison foundation.

## Dataset Design

The dataset must be large enough to expose differences between retrievers. The target is still small enough to read manually.

### Knowledge Data

Replace the single short `data/knowledge/rag-intro.md` with a richer local knowledge file that covers at least these sections:

- RAG basic definition and pipeline.
- Chunking and overlap.
- BM25 keyword retrieval.
- Vector retrieval and embeddings.
- Hybrid Search and RRF.
- Reranker role and two-stage retrieval.
- Retrieval metrics: `recall@k`, `precision@k`, `MRR`, and `nDCG`.
- RAG vs fine-tuning.
- Common RAG application scenarios.
- Common retrieval failure modes.

The file should be structured with headings and short paragraphs so chunk citations are stable and easy to inspect.

### Evaluation Data

Expand `data/eval/questions.jsonl` to roughly 18 to 24 cases. The exact number may vary if chunk boundaries make some cases redundant, but it should be far more than the current 2.

The dataset should include:

- Exact keyword questions, such as asking what BM25 means.
- Semantic rewrite questions, such as asking why semantic search can find paraphrases.
- Multi-citation questions where more than one chunk is relevant.
- Confusable questions, such as BM25 vs vector retrieval.
- Metric questions, such as what `recall@k` and `precision@k` measure.
- Scenario questions, such as when RAG is better than fine-tuning.

Each row keeps the existing fields:

```json
{"question":"BM25 适合什么场景？","expected_answer":"精确术语","expected_citations":["rag-intro.md#chunk-2"]}
```

Multiple expected citations are allowed:

```json
{"question":"Hybrid Search 为什么更稳？","expected_answer":"关键词 语义","expected_citations":["rag-intro.md#chunk-3","rag-intro.md#chunk-4"]}
```

The evaluator should treat all expected citations as relevant labels for retrieval metrics.

## Retrieval Architecture

Add a new module:

```text
src/rag_learning/retrievers.py
```

It owns retrieval abstractions and retrieval implementations.

### Shared Types

Define a protocol:

```python
class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ...
```

The existing vector-store classes already provide this shape, so they can be used as the vector retriever without wrapping unless a wrapper makes the CLI clearer.

### BM25 Retriever

Implement a local `BM25Retriever` that:

- Accepts a list of `Chunk` objects.
- Tokenizes text with the existing tokenization style used by `HashEmbeddingModel`, so Chinese characters and ASCII terms are both searchable.
- Computes document frequency and average document length at construction time.
- Scores query terms using BM25 with constants `k1=1.5` and `b=0.75`.
- Returns `RetrievedChunk` sorted by score descending.
- Raises `ValueError` when `top_k <= 0`, matching existing vector stores.

The first version should not add a third-party dependency. Implementing BM25 directly is better for learning.

### Hybrid Retriever

Implement `HybridRetriever` with Reciprocal Rank Fusion:

```text
rrf_score = sum(1 / (rrf_k + rank)) across retrievers
```

Design choices:

- It accepts multiple retrievers, initially BM25 and vector.
- It asks each retriever for a candidate pool larger than the final `top_k`, for example `max(top_k * 4, top_k)`.
- It merges duplicate chunks by citation.
- It returns `RetrievedChunk` with the fused RRF score.
- It defaults to `rrf_k=60`.

RRF avoids mixing incompatible BM25 scores and vector similarity scores.

## Evaluation Metrics

Extend `EvaluationSummary` with:

- `recall_at_k`
- `precision_at_k`
- `mrr`

Extend `EvaluationResult` with:

- `recall_at_k`
- `precision_at_k`
- `reciprocal_rank`
- `first_relevant_rank`

Metric rules:

- A retrieved chunk is relevant if its citation is in `expected_citations`.
- `recall@k = relevant_retrieved_count / expected_citation_count`.
- `precision@k = relevant_retrieved_count / retrieved_count`, using retrieved count instead of fixed `k` so short result lists are not unfairly penalized.
- Reciprocal rank is `1 / first_relevant_rank`, or `0.0` if none is found.
- Cases with no `expected_citations` keep `retrieval_hit = None` and are excluded from recall, precision, and MRR denominators.

The existing `retrieval_hit_rate` stays for continuity, but the new metrics become the preferred way to compare retrievers.

## CLI Design

Add a `--retriever` option to `rag eval`:

```bash
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5
```

CLI behavior:

- `vector` uses the existing configured vector store and requires prior `rag ingest`.
- `bm25` loads and chunks the knowledge path at evaluation time.
- `hybrid` combines the existing configured vector store with an in-memory BM25 retriever.
- Add `--knowledge-path`, defaulting to `data/knowledge`, for `bm25` and `hybrid`.
- Add `--chunk-size` and `--overlap` to `rag eval` so BM25 chunking can match ingest settings.
- Keep `--store-backend`, `--embedding-backend`, `--storage-path`, and `--collection` for vector and hybrid.
- Output summary should include the selected retriever and new retrieval metrics.

The `rag search` command can remain vector-only in this iteration. Comparing retrievers through `rag eval` is the learning priority.

## Data Flow

For vector:

```text
JSONL cases -> configured vector store -> evaluate retrieved citations -> answer -> metrics
```

For BM25:

```text
knowledge path -> load documents -> split chunks -> BM25 index
JSONL cases -> BM25 search -> evaluate retrieved citations -> answer -> metrics
```

For hybrid:

```text
configured vector store + BM25 index
JSONL cases -> vector candidates + BM25 candidates -> RRF merge -> answer -> metrics
```

The answer generation path still uses the selected retrieved contexts, so `answer_contains_expected_rate` remains visible.

## Error Handling

- Invalid `--retriever` values should be rejected by Typer through a `Literal`.
- `bm25` and `hybrid` should fail clearly if `--knowledge-path` does not exist.
- `vector` should preserve existing behavior when the vector store has no indexed chunks.
- `top_k <= 0` should raise the same style of `ValueError` already used by vector stores.
- Invalid evaluation JSONL should keep the existing line-numbered errors.

## Testing Strategy

Use TDD.

Tests should cover:

- BM25 returns exact keyword matches above unrelated chunks.
- BM25 rejects invalid `top_k`.
- Hybrid RRF promotes chunks that rank well across multiple retrievers.
- Retrieval metrics for recall, precision, first relevant rank, and MRR.
- Cases without expected citations are excluded from metric denominators.
- CLI `rag eval --retriever bm25` runs without prior ingest.
- CLI `rag eval --retriever vector` still works with the existing JSON vector store after ingest.
- CLI `rag eval --retriever hybrid` uses both vector and BM25 paths.
- Expanded sample dataset runs with all three retrievers.

## Verification

The implementation is complete when:

- `uv run pytest` passes.
- `uv run ruff check` passes.
- `uv run rag ingest data/knowledge` succeeds.
- `uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5` succeeds.
- `uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5` succeeds.
- `uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5` succeeds.

## Follow-up Ideas

After this iteration, the next useful additions are:

- `rag experiment` to run a matrix of retrievers and `top_k` values.
- CSV or JSON run output for comparing results over time.
- nDCG once the evaluation dataset supports graded relevance.
- A reranker stage after the hybrid retriever.
