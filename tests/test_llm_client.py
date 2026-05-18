from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from rag_learning.llm_client import ClaudeLLM
from rag_learning.models import Chunk, RetrievedChunk


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Claude answer with "),
                SimpleNamespace(type="text", text="[rag.md#chunk-0]"),
            ]
        )


class FakeAnthropic:
    instances: list[FakeAnthropic] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = FakeMessages()
        self.instances.append(self)


def test_claude_llm_uses_anthropic_env_format(monkeypatch) -> None:
    fake_module = SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "2048")

    llm = ClaudeLLM()
    result = llm.generate(
        "What is RAG?",
        [
            RetrievedChunk(
                chunk=Chunk(
                    source=Path("rag.md"),
                    index=0,
                    text="RAG means retrieval augmented generation.",
                ),
                score=0.9,
            )
        ],
    )

    client = FakeAnthropic.instances[-1]
    call = client.messages.calls[-1]
    assert client.kwargs == {
        "api_key": "test-key",
        "base_url": "https://gateway.example.com",
    }
    assert call["model"] == "claude-test-model"
    assert call["max_tokens"] == 2048
    assert call["system"].startswith("You answer questions")
    assert call["messages"][0]["role"] == "user"
    assert "RAG means retrieval augmented generation." in call["messages"][0]["content"]
    assert result == "Claude answer with [rag.md#chunk-0]"
