# RAG Learning Python

一个用于学习 RAG 的 Python 项目。第一版刻意不用 LangChain / LlamaIndex 封装主流程，而是把每一步拆开：

```text
load documents -> split chunks -> embed -> store -> retrieve -> build prompt -> answer
```

## 技术栈

- Python 3.12
- uv
- Typer + Rich
- pytest + ruff
- 默认零配置 hash embedding
- 可选 sentence-transformers + Chroma
- 可选 OpenAI-compatible LLM

## 快速开始

```bash
uv sync --group dev
uv run pytest
uv run rag --help
```

## 使用本地语义检索依赖

```bash
uv sync --group dev --extra local
```

## 使用 OpenAI-compatible LLM

```bash
cp .env.example .env
uv sync --group dev --extra llm
```

设置 `.env` 后可以接入 OpenAI、DeepSeek、Qwen 或 Ollama 等兼容接口。

## 命令

```bash
uv run rag ingest data/knowledge
uv run rag search "什么是 RAG"
uv run rag ask "RAG 和微调有什么区别"
```

默认命令使用 hash embedding，便于先理解流程。真实语义效果请安装 `local` extra 并使用 sentence-transformers 后端。
