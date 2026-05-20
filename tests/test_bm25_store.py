from pathlib import Path

import pytest

from rag_learning.bm25_store import JsonBM25Store
from rag_learning.models import Chunk


def test_json_bm25_store_persists_and_reloads_searchable_index(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25-index.json"
    store = JsonBM25Store(index_path)
    store.add(
        [
            Chunk(source=Path("rag.md"), index=0, text="BM25 适合精确关键词检索。"),
            Chunk(source=Path("rag.md"), index=1, text="向量检索适合语义相似问题。"),
        ]
    )

    reloaded = JsonBM25Store(index_path)
    results = reloaded.search("BM25 关键词", top_k=2)

    assert index_path.exists()
    assert [result.chunk.citation for result in results] == [
        "rag.md#chunk-0",
        "rag.md#chunk-1",
    ]
    assert results[0].score > results[1].score


def test_json_bm25_store_can_require_existing_index(tmp_path: Path) -> None:
    index_path = tmp_path / "missing-bm25-index.json"

    with pytest.raises(FileNotFoundError, match="BM25 index does not exist"):
        JsonBM25Store(index_path, require_existing=True)
