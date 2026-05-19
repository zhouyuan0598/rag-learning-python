# RAG Evaluation Lab Design

## Goal

Build a first-version RAG quality evaluation lab inside the existing Python learning project. The feature should help compare retrieval and answer quality using repeatable local evaluation datasets, while staying small enough for learning and test-driven iteration.

## Scope

The first version is a CLI workflow centered on a new `rag eval` command. It reads a JSONL evaluation file, runs each question through the existing retrieval and answer pipeline, computes simple deterministic metrics, and prints an aggregate summary plus per-question details.

This version does not include a web UI, LLM-as-judge scoring, background jobs, persistent experiment history, or a new orchestration framework. Those can come after the project has a reliable baseline evaluator.

## User Workflow

The user creates an evaluation dataset such as `data/eval/questions.jsonl`:

```jsonl
{"question":"什么是 RAG？","expected_answer":"检索增强生成","expected_citations":["rag-intro.md#chunk-0"]}
{"question":"RAG 的主要步骤有哪些？","expected_answer":"检索 生成","expected_citations":["rag-intro.md#chunk-0"]}
```

Then the user runs:

```bash
uv run rag eval data/eval/questions.jsonl --top-k 3
```

The command uses the same store, embedding backend, storage path, and collection options as `search` and `ask`, so the user can compare experiments by changing flags:

```bash
uv run rag eval data/eval/questions.jsonl --top-k 3 --embedding-backend hash
uv run rag eval data/eval/questions.jsonl --top-k 5 --embedding-backend sentence-transformer --store-backend chroma
```

## Evaluation Data Model

Each JSONL row represents one evaluation case.

Required fields:

- `question`: the question to retrieve and answer.
- `expected_answer`: a short reference answer or phrase. The first version uses deterministic substring checks instead of semantic judging.

Optional fields:

- `expected_citations`: citation labels expected to appear in the top-k retrieval results.

Rows with invalid JSON, missing required fields, or wrong field types should fail with a clear message that includes the 1-based line number.

## Metrics

The first version computes deterministic metrics only:

- `case_count`: number of evaluated cases.
- `retrieval_hit_rate`: fraction of cases where at least one expected citation appears in top-k.
- `answer_contains_expected_rate`: fraction of cases where the generated answer contains the expected answer text after basic whitespace-insensitive normalization.
- `avg_top_score`: average score of the first retrieved chunk, using `0.0` for cases with no retrieved chunks.
- `avg_retrieval_ms`: average retrieval time in milliseconds.
- `avg_answer_ms`: average answer generation time in milliseconds.

Per-case detail includes:

- question
- whether retrieval hit
- whether answer contained expected text
- expected citations
- top citation
- top score
- retrieval latency
- answer latency

If a case has no `expected_citations`, retrieval hit should be treated as not applicable for that row and excluded from the retrieval hit-rate denominator.

## Architecture

Add a focused `src/rag_learning/evaluation.py` module. It owns evaluation-case parsing, metric calculation, and result dataclasses. It depends on the existing `RagPipeline` interface rather than duplicating retrieval or answer logic.

`src/rag_learning/cli.py` adds an `eval` command that:

1. Creates the vector store with the existing `_create_store()` helper.
2. Creates an LLM using the existing `_create_llm()` helper, defaulting to `offline`.
3. Loads evaluation cases from a JSONL path.
4. Runs the evaluator.
5. Renders summary and detail tables with Rich.

The existing `RagPipeline` can remain unchanged for the first version because it already exposes `search()` and `answer()`.

## Error Handling

The evaluator should raise `ValueError` for invalid evaluation files with actionable messages. The CLI should catch those errors and display them as Typer errors rather than Python tracebacks.

Runtime errors from missing optional dependencies should keep using the existing behavior from `_create_store()`, `_create_embedding_model()`, and `_create_llm()`.

## Testing

Use TDD for implementation.

Test coverage should include:

- valid JSONL parsing into evaluation cases
- invalid JSONL line reporting with line number
- retrieval hit-rate with expected citations
- rows without expected citations being excluded from retrieval-hit denominator
- answer substring scoring
- CLI smoke test for `rag eval` using the default JSON store and offline LLM

The implementation is complete when:

- `uv run pytest` passes
- `uv run ruff check` passes
- README includes a short evaluation workflow
- the sample evaluation file can run against the existing sample knowledge data after ingestion
