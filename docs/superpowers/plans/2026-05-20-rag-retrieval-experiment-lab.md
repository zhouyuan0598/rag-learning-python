# RAG Retrieval Experiment Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BM25, vector, and hybrid retrieval comparison to `rag eval`, with standard retrieval metrics and a larger local learning dataset.

**Architecture:** Create `rag_learning.retrievers` for BM25, RRF hybrid retrieval, and shared retriever typing. Extend `rag_learning.evaluation` so evaluation can run against any retriever while preserving answer generation through an LLM client. Extend `rag eval` with `--retriever`, `--knowledge-path`, `--chunk-size`, and `--overlap`, then expand the sample knowledge and JSONL evaluation dataset.

**Tech Stack:** Python 3.12, Typer, Rich, pytest, Ruff, existing document loader, splitter, vector stores, and offline LLM.

---

### Task 1: BM25 Retriever

**Files:**
- Create: `src/rag_learning/retrievers.py`
- Create: `tests/test_retrievers.py`

- [ ] **Step 1: Write failing BM25 ranking and validation tests**

Create `tests/test_retrievers.py` with tests for exact keyword ranking and invalid `top_k`:

```python
from pathlib import Path

import pytest

from rag_learning.models import Chunk
from rag_learning.retrievers import BM25Retriever


def test_bm25_ranks_keyword_match_above_unrelated_chunk() -> None:
    retriever = BM25Retriever(
        [
            Chunk(source=Path("rag.md"), index=0, text="BM25 适合精确术语和关键词检索。"),
            Chunk(source=Path("rag.md"), index=1, text="向量检索适合语义相似和自然语言改写。"),
        ]
    )

    results = retriever.search("BM25 关键词", top_k=2)

    assert [result.chunk.citation for result in results] == [
        "rag.md#chunk-0",
        "rag.md#chunk-1",
    ]
    assert results[0].score > results[1].score


def test_bm25_rejects_non_positive_top_k() -> None:
    retriever = BM25Retriever([Chunk(source=Path("rag.md"), index=0, text="BM25")])

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("BM25", top_k=0)
```

- [ ] **Step 2: Run BM25 tests to verify they fail**

```bash
uv run pytest tests/test_retrievers.py::test_bm25_ranks_keyword_match_above_unrelated_chunk tests/test_retrievers.py::test_bm25_rejects_non_positive_top_k -v
```

Expected: fail because `rag_learning.retrievers` does not exist.

- [ ] **Step 3: Implement BM25Retriever**

Create `src/rag_learning/retrievers.py` with:

- `Retriever` protocol.
- `_tokens()` tokenizer matching the embedding token pattern.
- `BM25Retriever` with `k1=1.5`, `b=0.75`, document frequency, average document length, and descending score sorting.

- [ ] **Step 4: Run BM25 tests to verify they pass**

```bash
uv run pytest tests/test_retrievers.py -v
```

Expected: pass.

- [ ] **Step 5: Commit BM25 task**

```bash
git add src/rag_learning/retrievers.py tests/test_retrievers.py
git commit -m "feat: add bm25 retriever"
```

### Task 2: Hybrid Retriever with RRF

**Files:**
- Modify: `src/rag_learning/retrievers.py`
- Modify: `tests/test_retrievers.py`

- [ ] **Step 1: Write failing HybridRetriever test**

Append a small static retriever and assert RRF promotes a chunk that appears in both result lists:

```python
from rag_learning.models import RetrievedChunk
from rag_learning.retrievers import HybridRetriever


class StaticRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.results[:top_k]


def test_hybrid_retriever_fuses_rankings_with_rrf() -> None:
    chunk_a = Chunk(source=Path("rag.md"), index=0, text="BM25 keyword match")
    chunk_b = Chunk(source=Path("rag.md"), index=1, text="Vector semantic match")
    chunk_c = Chunk(source=Path("rag.md"), index=2, text="Shared strong match")

    hybrid = HybridRetriever(
        [
            StaticRetriever(
                [
                    RetrievedChunk(chunk=chunk_c, score=9.0),
                    RetrievedChunk(chunk=chunk_a, score=8.0),
                ]
            ),
            StaticRetriever(
                [
                    RetrievedChunk(chunk=chunk_b, score=0.9),
                    RetrievedChunk(chunk=chunk_c, score=0.8),
                ]
            ),
        ],
        rrf_k=60,
    )

    results = hybrid.search("query", top_k=2)

    assert [result.chunk.citation for result in results] == [
        "rag.md#chunk-2",
        "rag.md#chunk-1",
    ]
    assert results[0].score > results[1].score
```

- [ ] **Step 2: Run hybrid test to verify it fails**

```bash
uv run pytest tests/test_retrievers.py::test_hybrid_retriever_fuses_rankings_with_rrf -v
```

Expected: fail because `HybridRetriever` does not exist.

- [ ] **Step 3: Implement HybridRetriever**

Add `HybridRetriever` that searches each child retriever with `max(top_k * 4, top_k)`, merges by citation, sums `1 / (rrf_k + rank)`, and returns top-k fused `RetrievedChunk` results.

- [ ] **Step 4: Run retriever tests**

```bash
uv run pytest tests/test_retrievers.py -v
```

Expected: pass.

- [ ] **Step 5: Commit hybrid task**

```bash
git add src/rag_learning/retrievers.py tests/test_retrievers.py
git commit -m "feat: add hybrid retriever"
```

### Task 3: Standard Retrieval Metrics

**Files:**
- Modify: `src/rag_learning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing metrics test**

Add a test with two expected citations and retrieved results where only one relevant result appears at rank 2:

```python
def test_evaluate_cases_reports_recall_precision_and_mrr() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add(
        [
            Chunk(source=Path("rag.md"), index=0, text="unrelated filler"),
            Chunk(source=Path("rag.md"), index=1, text="BM25 关键词检索"),
            Chunk(source=Path("rag.md"), index=2, text="Hybrid Search 混合检索"),
        ]
    )
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="BM25",
                expected_answer="BM25",
                expected_citations=["rag.md#chunk-1", "rag.md#chunk-2"],
            )
        ],
        top_k=2,
    )

    assert report.results[0].recall_at_k == 0.5
    assert report.results[0].precision_at_k == 0.5
    assert report.results[0].first_relevant_rank == 1
    assert report.results[0].reciprocal_rank == 1.0
    assert report.summary.recall_at_k == 0.5
    assert report.summary.precision_at_k == 0.5
    assert report.summary.mrr == 1.0
```

- [ ] **Step 2: Run metrics test to verify it fails**

```bash
uv run pytest tests/test_evaluation.py::test_evaluate_cases_reports_recall_precision_and_mrr -v
```

Expected: fail because the metric fields do not exist.

- [ ] **Step 3: Extend evaluation dataclasses and summary logic**

Add result fields `recall_at_k`, `precision_at_k`, `reciprocal_rank`, and `first_relevant_rank`. Add summary fields `recall_at_k`, `precision_at_k`, and `mrr`. Compute metrics only for cases with `expected_citations`.

- [ ] **Step 4: Preserve no-citation denominator behavior**

Update existing no-citation test to assert `recall_at_k`, `precision_at_k`, and `mrr` remain `0.0` at summary level when no cases have expected citations.

- [ ] **Step 5: Run evaluation tests**

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: pass.

- [ ] **Step 6: Commit metrics task**

```bash
git add src/rag_learning/evaluation.py tests/test_evaluation.py
git commit -m "feat: add standard retrieval metrics"
```

### Task 4: CLI Retriever Selection

**Files:**
- Modify: `src/rag_learning/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing BM25 CLI smoke test**

Add a CLI test where `rag eval --retriever bm25` runs without prior `rag ingest` and prints `recall_at_k`, `precision_at_k`, and `mrr`.

- [ ] **Step 2: Run BM25 CLI test to verify it fails**

```bash
uv run pytest tests/test_cli.py::test_eval_command_can_use_bm25_retriever_without_ingest -v
```

Expected: fail because `--retriever` does not exist.

- [ ] **Step 3: Implement CLI retriever factory**

Add `RetrieverBackend = Literal["vector", "bm25", "hybrid"]`. Add `_create_eval_retriever()` and `_create_bm25_retriever()` helpers. Extend `eval_command()` with `--retriever`, `--knowledge-path`, `--chunk-size`, and `--overlap`.

- [ ] **Step 4: Write and pass vector and hybrid CLI tests**

Add tests for:

- `rag eval --retriever vector` after ingest.
- `rag eval --retriever hybrid` after ingest.

Run:

```bash
uv run pytest tests/test_cli.py -v
```

Expected: pass.

- [ ] **Step 5: Commit CLI task**

```bash
git add src/rag_learning/cli.py tests/test_cli.py
git commit -m "feat: compare retrievers in rag eval"
```

### Task 5: Expanded Knowledge and Evaluation Dataset

**Files:**
- Modify: `data/knowledge/rag-intro.md`
- Modify: `data/eval/questions.jsonl`
- Modify: `README.md`
- Modify: `docs/rag-retrieval-theory.md`

- [ ] **Step 1: Expand knowledge file**

Rewrite `data/knowledge/rag-intro.md` into multiple short sections covering RAG pipeline, chunking, BM25, vector retrieval, hybrid search, reranking, metrics, fine-tuning comparison, scenarios, and failure modes.

- [ ] **Step 2: Expand evaluation JSONL**

Replace `data/eval/questions.jsonl` with roughly 18-24 cases. Ensure every expected citation refers to a real chunk after default `chunk_size=800` and `overlap=100`.

- [ ] **Step 3: Update README**

Document:

```bash
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5
```

Also explain `recall_at_k`, `precision_at_k`, and `mrr` briefly.

- [ ] **Step 4: Run sample workflows**

```bash
uv run rag ingest data/knowledge
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5
```

Expected: all commands exit 0.

- [ ] **Step 5: Run full verification**

```bash
uv run pytest
uv run ruff check
```

Expected: pass.

- [ ] **Step 6: Commit docs and dataset task**

```bash
git add README.md docs/rag-retrieval-theory.md data/knowledge/rag-intro.md data/eval/questions.jsonl
git commit -m "docs: expand retrieval evaluation dataset"
```

### Task 6: Final Verification and Push

**Files:**
- Review all modified files.

- [ ] **Step 1: Inspect final state**

```bash
git status --short --branch
git log --oneline -10
```

- [ ] **Step 2: Fresh verification**

```bash
uv run pytest
uv run ruff check
uv run rag ingest data/knowledge
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5
```

- [ ] **Step 3: Push branch**

```bash
git push
```
