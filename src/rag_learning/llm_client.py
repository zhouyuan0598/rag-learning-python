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


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed. Run: uv sync --extra llm --group dev"
            ) from exc

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided retrieved context. "
                        "Cite sources with the citation labels. If context is insufficient, say so."
                    ),
                },
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_text = "\n\n".join(
        f"[{result.chunk.citation}]\n{result.chunk.text}" for result in contexts
    )
    return f"Question:\n{question}\n\nRetrieved context:\n{context_text}"
