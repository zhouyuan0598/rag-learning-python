# Expanded RAG Dataset Design

## Goal

Expand the sample RAG knowledge base and evaluation set so retrieval experiments are no longer dominated by a four-chunk corpus. The new data should make `vector`, `bm25`, and `hybrid` comparisons more meaningful while staying readable for learning.

## Scope

This change adds local, original learning material only. It does not introduce external datasets, new retriever logic, rerankers, or new CLI options.

## Knowledge Data

The knowledge base will become a small multi-file RAG handbook under `data/knowledge`:

- `rag-intro.md`: RAG definition, pipeline, and when the pattern is useful.
- `chunking-and-indexing.md`: chunk size, overlap, metadata, and indexing lifecycle.
- `bm25-and-vector.md`: BM25, vector search, hybrid search, and RRF.
- `reranking.md`: rerankers, candidate generation, and latency trade-offs.
- `evaluation-metrics.md`: recall, precision, MRR, nDCG, and answer quality caveats.
- `production-rag.md`: production storage, update jobs, observability, and safety.
- `failure-cases.md`: retrieval failure modes and practical debugging steps.
- `query-rewrite-and-routing.md`: query rewriting, routing, and multi-intent retrieval.
- `prompt-and-citations.md`: prompt assembly, citations, refusal, and prompt injection.

With the existing default `chunk_size=800` and `overlap=100`, this should produce at least 25 chunks.

## Evaluation Data

`data/eval/questions.jsonl` will grow from 21 cases to at least 60 cases. Cases will include exact keyword questions, semantic paraphrases, multi-citation questions, confusable concepts, production-operation questions, and failure-analysis questions.

Every case must:

- Keep the current `question`, `expected_answer`, and `expected_citations` fields.
- Reference citations that exist after default chunking.
- Include at least one expected citation.

## Validation

Add a dataset integrity test that loads the real sample knowledge and evaluation files, splits them with default settings, and verifies:

- At least 7 knowledge documents.
- At least 25 chunks.
- At least 60 evaluation cases.
- Unique questions.
- At least one multi-citation case.
- All expected citations exist.

Then run `rag ingest` and compare `vector`, `bm25`, and `hybrid` at `--top-k 2`.
