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
- 可选 Claude / Anthropic LLM

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

第一次使用 `--embedding-backend sentence-transformer` 会下载 embedding 模型。默认模型是：

```text
BAAI/bge-small-zh-v1.5
```

如果下载进度长时间卡在 `model.safetensors: 0%`，先停止当前命令，然后重新执行：

```bash
HF_HUB_DISABLE_XET=1 uv run rag ingest data/knowledge \
  --embedding-backend sentence-transformer \
  --embedding-model BAAI/bge-small-zh-v1.5 \
  --store-backend chroma
```

也可以在 `.env` 里设置默认 embedding 模型：

```env
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
HF_HUB_DISABLE_XET=1
```

模型文件约 95.8MB。如果网络速度只有几十 KB/s，首次下载可能需要二十分钟以上；下载完成后会被缓存，后续运行会快很多。

## 使用 Claude LLM

```bash
cp .env.example .env
uv sync --group dev --extra llm
```

设置 `.env` 后可以通过 Anthropic Python SDK 调用 Claude。

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
ANTHROPIC_MAX_TOKENS=1024
```

## 命令

```bash
uv run rag ingest data/knowledge
uv run rag search "什么是 RAG"
uv run rag ask "RAG 和微调有什么区别"
uv run rag ask "RAG 和微调有什么区别" --llm-backend claude
```

默认命令使用 hash embedding，便于先理解流程。真实语义效果请安装 `local` extra 并使用 sentence-transformers 后端。
