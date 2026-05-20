# Expanded RAG Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the bundled RAG learning dataset so retrieval experiments run against a realistic multi-file corpus and a larger labeled eval set.

**Architecture:** Keep the retrieval code unchanged. Add more local Markdown knowledge files, replace the JSONL eval set, and add tests that validate dataset size and citation correctness against the existing loader and splitter.

**Tech Stack:** Python 3.12, pytest, Typer CLI, local Markdown and JSONL sample data.

---

### Task 1: Dataset Integrity Test

**Files:**
- Create: `tests/test_sample_dataset.py`

- [x] **Step 1: Write the failing test**

Add tests that load `data/knowledge`, split with default settings, and require at least 7 documents, 25 chunks, 60 eval cases, unique questions, a multi-citation case, and valid citations.

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sample_dataset.py -q`

Expected: FAIL because the current sample has 1 knowledge document and 21 eval cases.

### Task 2: Expand Knowledge Files

**Files:**
- Modify: `data/knowledge/rag-intro.md`
- Create: `data/knowledge/chunking-and-indexing.md`
- Create: `data/knowledge/bm25-and-vector.md`
- Create: `data/knowledge/reranking.md`
- Create: `data/knowledge/evaluation-metrics.md`
- Create: `data/knowledge/production-rag.md`
- Create: `data/knowledge/failure-cases.md`
- Create: `data/knowledge/query-rewrite-and-routing.md`
- Create: `data/knowledge/prompt-and-citations.md`

- [x] **Step 1: Write focused Markdown files**

Each file should explain one retrieval topic with short paragraphs and stable headings.

- [x] **Step 2: Check chunk count**

Run a local loader/splitter check and confirm at least 25 chunks.

### Task 3: Expand Evaluation Cases

**Files:**
- Modify: `data/eval/questions.jsonl`

- [x] **Step 1: Replace JSONL with at least 60 cases**

Use the current schema: `question`, `expected_answer`, `expected_citations`.

- [x] **Step 2: Validate citations**

Run: `uv run pytest tests/test_sample_dataset.py -q`

Expected: PASS.

### Task 4: End-to-End Verification

**Files:**
- No source files changed.

- [x] **Step 1: Rebuild indexes**

Run: `uv run rag ingest data/knowledge`

- [x] **Step 2: Compare retrievers**

Run:

```bash
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 2
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 2
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 2
```

- [x] **Step 3: Run full checks**

Run:

```bash
uv run pytest
uv run ruff check
```
