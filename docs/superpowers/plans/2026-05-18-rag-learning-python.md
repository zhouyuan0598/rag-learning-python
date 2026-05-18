# RAG Learning Python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a runnable Python RAG learning project with ingestion, retrieval, and answer generation.

**Architecture:** The project exposes a CLI backed by small modules for document loading, splitting, embedding, vector storage, LLM generation, and pipeline orchestration. Optional heavy dependencies are lazy-loaded so the base project can be tested quickly.

**Tech Stack:** Python 3.12, uv, Typer, Rich, pytest, ruff, optional Chroma, optional sentence-transformers, optional Claude / Anthropic LLM client.

---

### Task 1: Core Models and Document Loading

**Files:**
- Create: `src/rag_learning/models.py`
- Create: `src/rag_learning/document_loader.py`
- Test: `tests/test_document_loader.py`

- [x] **Step 1: Write failing tests for loading Markdown and TXT documents**
- [x] **Step 2: Run tests and verify imports fail before implementation**
- [x] **Step 3: Implement document model and loader**
- [x] **Step 4: Run document loader tests and verify they pass**

### Task 2: Chunk Splitting

**Files:**
- Create: `src/rag_learning/splitter.py`
- Test: `tests/test_splitter.py`

- [x] **Step 1: Write failing tests for overlap and source metadata**
- [x] **Step 2: Run tests and verify imports fail before implementation**
- [x] **Step 3: Implement character-based chunking with overlap**
- [x] **Step 4: Run splitter tests and verify they pass**

### Task 3: Embeddings and Vector Store

**Files:**
- Create: `src/rag_learning/embeddings.py`
- Create: `src/rag_learning/vector_store.py`
- Test: `tests/test_vector_store.py`

- [x] **Step 1: Write failing retrieval test**
- [x] **Step 2: Run tests and verify imports fail before implementation**
- [x] **Step 3: Implement hash embeddings and in-memory cosine search**
- [x] **Step 4: Run vector store tests and verify they pass**

### Task 4: RAG Pipeline and CLI

**Files:**
- Create: `src/rag_learning/llm_client.py`
- Create: `src/rag_learning/rag_pipeline.py`
- Create: `src/rag_learning/cli.py`
- Test: `tests/test_rag_pipeline.py`

- [x] **Step 1: Write failing answer citation test**
- [x] **Step 2: Run tests and verify imports fail before implementation**
- [x] **Step 3: Implement offline LLM, RAG pipeline, and CLI commands**
- [x] **Step 4: Run all tests, lint, and a smoke CLI command**
