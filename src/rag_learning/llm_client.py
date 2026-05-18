from __future__ import annotations

import os
from typing import Protocol

from rag_learning.models import RetrievedChunk


class LLMClient(Protocol):
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        raise NotImplementedError


class OfflineContextLLM:
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "No relevant context was found."

        sections = [
            f"[{result.chunk.citation}] score={result.score:.4f}\n{result.chunk.text}"
            for result in contexts
        ]
        return (
            "Offline answer mode: no external LLM was called.\n\n"
            f"Question: {question}\n\n"
            "Retrieved context:\n" + "\n\n".join(sections)
        )


class ClaudeLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic is not installed. Run: uv sync --extra llm --group dev"
            ) from exc

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        self.max_tokens = max_tokens or int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))

        client_kwargs = {}
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        resolved_base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        if resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = Anthropic(**client_kwargs)

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=(
                "You answer questions using only the provided retrieved context. "
                "Cite sources with the citation labels. If context is insufficient, say so."
            ),
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(question, contexts),
                }
            ],
            temperature=0.2,
        )
        return _message_text(message)


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_text = "\n\n".join(
        f"[{result.chunk.citation}]\n{result.chunk.text}" for result in contexts
    )
    return f"Question:\n{question}\n\nRetrieved context:\n{context_text}"


def _message_text(message) -> str:
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)
