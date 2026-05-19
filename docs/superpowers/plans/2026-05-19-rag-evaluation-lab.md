# RAG Evaluation Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `rag eval` CLI workflow that scores retrieval and offline answer quality from a JSONL evaluation dataset.

**Architecture:** Add a focused `rag_learning.evaluation` module for parsing evaluation cases, running the existing `RagPipeline`, and aggregating deterministic metrics. Extend the Typer CLI with an `eval` command that reuses the existing store, embedding, and LLM factory helpers and renders Rich tables.

**Tech Stack:** Python 3.12, Typer, Rich, pytest, Ruff, existing `RagPipeline`, `JsonVectorStore`, and `OfflineContextLLM`.

---

### Task 1: Evaluation Case Parsing

**Files:**
- Create: `src/rag_learning/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing valid JSONL parsing test**

```python
from pathlib import Path

from rag_learning.evaluation import EvaluationCase, load_evaluation_cases


def test_load_evaluation_cases_reads_jsonl(tmp_path: Path) -> None:
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"什么是 RAG？","expected_answer":"检索增强生成",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_cases(eval_file)

    assert cases == [
        EvaluationCase(
            question="什么是 RAG？",
            expected_answer="检索增强生成",
            expected_citations=["rag-intro.md#chunk-0"],
        )
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_load_evaluation_cases_reads_jsonl -v
```

Expected: FAIL because `rag_learning.evaluation` does not exist.

- [ ] **Step 3: Implement minimal parser and dataclass**

Create `src/rag_learning/evaluation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_answer: str
    expected_citations: list[str]


def load_evaluation_cases(path: Path | str) -> list[EvaluationCase]:
    eval_path = Path(path)
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        data = json.loads(raw_line)
        cases.append(
            EvaluationCase(
                question=str(data["question"]),
                expected_answer=str(data["expected_answer"]),
                expected_citations=[str(value) for value in data.get("expected_citations", [])],
            )
        )
    return cases
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_load_evaluation_cases_reads_jsonl -v
```

Expected: PASS.

- [ ] **Step 5: Write failing validation test**

Append to `tests/test_evaluation.py`:

```python
import pytest


def test_load_evaluation_cases_reports_invalid_json_line(tmp_path: Path) -> None:
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text(
        '{"question":"ok","expected_answer":"ok"}\n'
        '{"question": "broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_evaluation_cases(eval_file)
```

- [ ] **Step 6: Run validation test to verify it fails**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_load_evaluation_cases_reports_invalid_json_line -v
```

Expected: FAIL because invalid JSON raises `json.JSONDecodeError`, not `ValueError` with line context.

- [ ] **Step 7: Add parser validation**

Update `load_evaluation_cases()` so invalid JSON and invalid fields raise `ValueError` with the line number:

```python
def load_evaluation_cases(path: Path | str) -> list[EvaluationCase]:
    eval_path = Path(path)
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation JSON on line {line_number}: {exc.msg}") from exc
        cases.append(_case_from_data(data, line_number))
    return cases


def _case_from_data(data: object, line_number: int) -> EvaluationCase:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid evaluation case on line {line_number}: expected object")
    question = data.get("question")
    expected_answer = data.get("expected_answer")
    expected_citations = data.get("expected_citations", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Invalid evaluation case on line {line_number}: question is required")
    if not isinstance(expected_answer, str) or not expected_answer.strip():
        raise ValueError(
            f"Invalid evaluation case on line {line_number}: expected_answer is required"
        )
    if not isinstance(expected_citations, list) or not all(
        isinstance(value, str) for value in expected_citations
    ):
        raise ValueError(
            f"Invalid evaluation case on line {line_number}: expected_citations must be strings"
        )
    return EvaluationCase(
        question=question,
        expected_answer=expected_answer,
        expected_citations=expected_citations,
    )
```

- [ ] **Step 8: Run parser tests**

Run:

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit parser task**

```bash
git add src/rag_learning/evaluation.py tests/test_evaluation.py
git commit -m "feat: parse rag evaluation cases"
```

### Task 2: Evaluation Metrics

**Files:**
- Modify: `src/rag_learning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing metric test for retrieval hits and answer scoring**

Append to `tests/test_evaluation.py`:

```python
from rag_learning.embeddings import HashEmbeddingModel
from rag_learning.llm_client import OfflineContextLLM
from rag_learning.models import Chunk
from rag_learning.rag_pipeline import RagPipeline
from rag_learning.vector_store import InMemoryVectorStore
from rag_learning.evaluation import evaluate_cases


def test_evaluate_cases_scores_retrieval_and_answer_quality() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add(
        [
            Chunk(
                source=Path("rag-intro.md"),
                index=0,
                text="RAG 是检索增强生成，也就是 retrieval augmented generation.",
            )
        ]
    )
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="什么是 RAG？",
                expected_answer="检索增强生成",
                expected_citations=["rag-intro.md#chunk-0"],
            )
        ],
        top_k=1,
    )

    assert report.summary.case_count == 1
    assert report.summary.retrieval_hit_rate == 1.0
    assert report.summary.answer_contains_expected_rate == 1.0
    assert report.summary.avg_top_score > 0
    assert report.results[0].retrieval_hit is True
    assert report.results[0].answer_contains_expected is True
    assert report.results[0].top_citation == "rag-intro.md#chunk-0"
```

- [ ] **Step 2: Run metric test to verify it fails**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_evaluate_cases_scores_retrieval_and_answer_quality -v
```

Expected: FAIL because `evaluate_cases` does not exist.

- [ ] **Step 3: Implement metric dataclasses and evaluator**

Extend `src/rag_learning/evaluation.py`:

```python
from time import perf_counter

from rag_learning.rag_pipeline import RagPipeline


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    retrieval_hit: bool | None
    answer_contains_expected: bool
    expected_citations: list[str]
    retrieved_citations: list[str]
    top_citation: str | None
    top_score: float
    retrieval_ms: float
    answer_ms: float


@dataclass(frozen=True)
class EvaluationSummary:
    case_count: int
    retrieval_hit_rate: float
    answer_contains_expected_rate: float
    avg_top_score: float
    avg_retrieval_ms: float
    avg_answer_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    summary: EvaluationSummary
    results: list[EvaluationResult]


def evaluate_cases(
    pipeline: RagPipeline,
    cases: list[EvaluationCase],
    top_k: int = 5,
) -> EvaluationReport:
    results: list[EvaluationResult] = []
    for case in cases:
        retrieval_started = perf_counter()
        retrieved = pipeline.search(case.question, top_k=top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        answer_started = perf_counter()
        answer = pipeline.llm.generate(case.question, retrieved)
        answer_ms = (perf_counter() - answer_started) * 1000

        retrieved_citations = [result.chunk.citation for result in retrieved]
        retrieval_hit = None
        if case.expected_citations:
            retrieval_hit = any(citation in retrieved_citations for citation in case.expected_citations)

        top_result = retrieved[0] if retrieved else None
        results.append(
            EvaluationResult(
                case=case,
                retrieval_hit=retrieval_hit,
                answer_contains_expected=_contains_expected(answer, case.expected_answer),
                expected_citations=case.expected_citations,
                retrieved_citations=retrieved_citations,
                top_citation=top_result.chunk.citation if top_result else None,
                top_score=top_result.score if top_result else 0.0,
                retrieval_ms=retrieval_ms,
                answer_ms=answer_ms,
            )
        )

    return EvaluationReport(summary=_summarize(results), results=results)


def _contains_expected(answer: str, expected_answer: str) -> bool:
    return _normalize_text(expected_answer) in _normalize_text(answer)


def _normalize_text(value: str) -> str:
    return "".join(value.split()).lower()


def _summarize(results: list[EvaluationResult]) -> EvaluationSummary:
    hit_results = [result for result in results if result.retrieval_hit is not None]
    return EvaluationSummary(
        case_count=len(results),
        retrieval_hit_rate=_rate(
            sum(1 for result in hit_results if result.retrieval_hit), len(hit_results)
        ),
        answer_contains_expected_rate=_rate(
            sum(1 for result in results if result.answer_contains_expected), len(results)
        ),
        avg_top_score=_average([result.top_score for result in results]),
        avg_retrieval_ms=_average([result.retrieval_ms for result in results]),
        avg_answer_ms=_average([result.answer_ms for result in results]),
    )


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
```

- [ ] **Step 4: Run metric test to verify it passes**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_evaluate_cases_scores_retrieval_and_answer_quality -v
```

Expected: PASS.

- [ ] **Step 5: Write failing test for missing expected citations denominator**

Append to `tests/test_evaluation.py`:

```python
def test_evaluate_cases_excludes_rows_without_expected_citations_from_hit_rate() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimension=32))
    store.add([Chunk(source=Path("rag-intro.md"), index=0, text="RAG 是检索增强生成。")])
    pipeline = RagPipeline(store=store, llm=OfflineContextLLM())

    report = evaluate_cases(
        pipeline,
        [
            EvaluationCase(
                question="什么是 RAG？",
                expected_answer="检索增强生成",
                expected_citations=[],
            )
        ],
        top_k=1,
    )

    assert report.summary.retrieval_hit_rate == 0.0
    assert report.results[0].retrieval_hit is None
```

- [ ] **Step 6: Run denominator test**

Run:

```bash
uv run pytest tests/test_evaluation.py::test_evaluate_cases_excludes_rows_without_expected_citations_from_hit_rate -v
```

Expected: PASS if Task 2 Step 3 handled `None`; otherwise fix `_summarize()`.

- [ ] **Step 7: Run all evaluation tests**

Run:

```bash
uv run pytest tests/test_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit metric task**

```bash
git add src/rag_learning/evaluation.py tests/test_evaluation.py
git commit -m "feat: score rag evaluation cases"
```

### Task 3: CLI Eval Command

**Files:**
- Modify: `src/rag_learning/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Append to `tests/test_cli.py`:

```python
def test_eval_command_prints_summary_and_details(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge_dir = tmp_path / "data" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "rag-intro.md").write_text(
        "RAG 是检索增强生成，也就是 retrieval augmented generation.",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "data" / "eval"
    eval_dir.mkdir(parents=True)
    eval_file = eval_dir / "questions.jsonl"
    eval_file.write_text(
        '{"question":"什么是 RAG？","expected_answer":"检索增强生成",'
        '"expected_citations":["rag-intro.md#chunk-0"]}\n',
        encoding="utf-8",
    )

    ingest_result = runner.invoke(app, ["ingest", str(knowledge_dir)])
    assert ingest_result.exit_code == 0

    eval_result = runner.invoke(app, ["eval", str(eval_file), "--top-k", "1"])

    assert eval_result.exit_code == 0
    assert "Evaluation Summary" in eval_result.output
    assert "retrieval_hit_rate" in eval_result.output
    assert "answer_contains_expected_rate" in eval_result.output
    assert "rag-intro.md#chunk-0" in eval_result.output
```

- [ ] **Step 2: Run CLI test to verify it fails**

Run:

```bash
uv run pytest tests/test_cli.py::test_eval_command_prints_summary_and_details -v
```

Expected: FAIL because the `eval` command does not exist.

- [ ] **Step 3: Add CLI imports and command**

Update `src/rag_learning/cli.py` imports:

```python
from rag_learning.evaluation import EvaluationReport, evaluate_cases, load_evaluation_cases
```

Add command after `ask()`:

```python
@app.command("eval")
def eval_command(
    path: Annotated[Path, typer.Argument(help="JSONL evaluation file")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
    llm_backend: Annotated[LLMBackend, typer.Option("--llm-backend")] = "offline",
    store_backend: Annotated[StoreBackend, typer.Option("--store-backend")] = "json",
    embedding_backend: Annotated[EmbeddingBackend, typer.Option("--embedding-backend")] = "hash",
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    storage_path: Annotated[Path | None, typer.Option("--storage-path")] = None,
    collection: Annotated[str | None, typer.Option("--collection")] = None,
) -> None:
    try:
        cases = load_evaluation_cases(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="path") from exc

    pipeline = RagPipeline(
        store=_create_store(
            store_backend, storage_path, collection, embedding_backend, embedding_model
        ),
        llm=_create_llm(llm_backend),
    )
    report = evaluate_cases(pipeline, cases, top_k=top_k)
    _print_evaluation_report(report)
```

Add render helpers near `_print_results()`:

```python
def _print_evaluation_report(report: EvaluationReport) -> None:
    summary_table = Table(title="Evaluation Summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary = report.summary
    summary_table.add_row("case_count", str(summary.case_count))
    summary_table.add_row("retrieval_hit_rate", f"{summary.retrieval_hit_rate:.2%}")
    summary_table.add_row(
        "answer_contains_expected_rate", f"{summary.answer_contains_expected_rate:.2%}"
    )
    summary_table.add_row("avg_top_score", f"{summary.avg_top_score:.4f}")
    summary_table.add_row("avg_retrieval_ms", f"{summary.avg_retrieval_ms:.2f}")
    summary_table.add_row("avg_answer_ms", f"{summary.avg_answer_ms:.2f}")
    console.print(summary_table)

    detail_table = Table(title="Evaluation Details")
    detail_table.add_column("Question")
    detail_table.add_column("Retrieval")
    detail_table.add_column("Answer")
    detail_table.add_column("Expected citations")
    detail_table.add_column("Top citation")
    detail_table.add_column("Top score", justify="right")
    detail_table.add_column("Retrieval ms", justify="right")
    detail_table.add_column("Answer ms", justify="right")
    for result in report.results:
        detail_table.add_row(
            result.case.question,
            _format_optional_bool(result.retrieval_hit),
            _format_bool(result.answer_contains_expected),
            ", ".join(result.expected_citations) or "-",
            result.top_citation or "-",
            f"{result.top_score:.4f}",
            f"{result.retrieval_ms:.2f}",
            f"{result.answer_ms:.2f}",
        )
    console.print(detail_table)


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return _format_bool(value)


def _format_bool(value: bool) -> str:
    return "pass" if value else "fail"
```

- [ ] **Step 4: Run CLI test to verify it passes**

Run:

```bash
uv run pytest tests/test_cli.py::test_eval_command_prints_summary_and_details -v
```

Expected: PASS.

- [ ] **Step 5: Add invalid file CLI test**

Append to `tests/test_cli.py`:

```python
def test_eval_command_reports_invalid_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    eval_file = tmp_path / "questions.jsonl"
    eval_file.write_text('{"question":"broken"\n', encoding="utf-8")

    result = runner.invoke(app, ["eval", str(eval_file)])

    assert result.exit_code != 0
    assert "line 1" in result.output
```

- [ ] **Step 6: Run CLI eval tests**

Run:

```bash
uv run pytest tests/test_cli.py::test_eval_command_prints_summary_and_details tests/test_cli.py::test_eval_command_reports_invalid_jsonl -v
```

Expected: PASS.

- [ ] **Step 7: Commit CLI task**

```bash
git add src/rag_learning/cli.py tests/test_cli.py
git commit -m "feat: add rag eval command"
```

### Task 4: Sample Dataset and README

**Files:**
- Create: `data/eval/questions.jsonl`
- Modify: `README.md`

- [ ] **Step 1: Create sample evaluation file**

Create `data/eval/questions.jsonl`:

```jsonl
{"question":"什么是 RAG？","expected_answer":"检索增强生成","expected_citations":["rag-intro.md#chunk-0"]}
{"question":"RAG 的流程有哪些步骤？","expected_answer":"load documents split chunks embed store retrieve build prompt answer","expected_citations":["rag-intro.md#chunk-0"]}
```

- [ ] **Step 2: Update README with evaluation workflow**

Add after the command examples:

```markdown
## RAG 质量评估

先索引示例知识库：

```bash
uv run rag ingest data/knowledge
```

再运行评测集：

```bash
uv run rag eval data/eval/questions.jsonl --top-k 3
```

评估命令会输出整体指标和逐题明细：

- `retrieval_hit_rate`：期望 citation 是否出现在 top-k 检索结果中。
- `answer_contains_expected_rate`：离线回答是否包含参考答案文本。
- `avg_top_score`：每题最高检索分数的平均值。
- `avg_retrieval_ms` / `avg_answer_ms`：检索与回答生成耗时。

评测集使用 JSONL，每行一个问题：

```json
{"question":"什么是 RAG？","expected_answer":"检索增强生成","expected_citations":["rag-intro.md#chunk-0"]}
```
```

- [ ] **Step 3: Run sample workflow**

Run:

```bash
uv run rag ingest data/knowledge
uv run rag eval data/eval/questions.jsonl --top-k 3
```

Expected: both commands exit 0 and the eval output includes `Evaluation Summary`.

- [ ] **Step 4: Run full verification**

Run:

```bash
uv run pytest
uv run ruff check
```

Expected: PASS for both commands.

- [ ] **Step 5: Commit docs and sample data**

```bash
git add README.md data/eval/questions.jsonl
git commit -m "docs: add rag evaluation workflow"
```

### Task 5: Final Review and Push

**Files:**
- Review all modified files.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: branch is `codex/rag-evaluation-lab`, no unstaged source changes after commits.

- [ ] **Step 2: Fresh verification before completion**

Run:

```bash
uv run pytest
uv run ruff check
```

Expected: PASS for both commands.

- [ ] **Step 3: Push branch**

Run:

```bash
git push -u origin codex/rag-evaluation-lab
```

Expected: branch pushed and tracking `origin/codex/rag-evaluation-lab`.
