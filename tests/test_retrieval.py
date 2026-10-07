import json

import numpy as np
import pytest

from retrieval.index import VectorIndex, build_index, load_or_build_index
from retrieval.search import cosine_similarity, search


class FakeEmbedder:
    """Deterministic vectors, keyed by text, so tests never download a model."""

    def __init__(self) -> None:
        self.calls = 0
        self.vectors = {
            "revenue": [1.0, 0.0, 0.0],
            "profit": [0.0, 1.0, 0.0],
            "question": [0.9, 0.1, 0.0],
        }

    def embed(self, texts):
        self.calls += 1
        return np.array([self.vectors[text] for text in texts], dtype=np.float32)


def write_chunks(path) -> None:
    rows = [
        {"chunk_id": "report_c0001", "document_id": "report", "text": "revenue", "pages": [4], "metadata": {"page": 4, "source": "report.pdf", "section": "Sales"}},
        {"chunk_id": "report_c0002", "document_id": "report", "text": "profit", "pages": [5], "metadata": {"page_start": 5, "page_end": 6, "source": "report.pdf"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_index_embeds_every_chunk_and_preserves_mapping(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)

    index = build_index(chunks_path, FakeEmbedder())

    assert index.embeddings.shape == (2, 3)
    assert [chunk["chunk_id"] for chunk in index.chunks] == ["report_c0001", "report_c0002"]
    assert index.chunks[0]["metadata"]["section"] == "Sales"


def test_saved_index_is_reused_when_chunks_are_unchanged(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vectors.npz"
    write_chunks(chunks_path)
    embedder = FakeEmbedder()

    load_or_build_index(chunks_path, index_path, embedder)
    loaded = load_or_build_index(chunks_path, index_path, embedder)

    assert embedder.calls == 1
    assert loaded.dimension == 3


def test_cosine_similarity_and_search_ranking_metadata_and_top_k() -> None:
    index = VectorIndex(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        [
            {"chunk_id": "report_c0001", "document_id": "report", "text": "revenue", "pages": [4], "metadata": {"page": 4, "source": "report.pdf", "section": "Sales"}},
            {"chunk_id": "report_c0002", "document_id": "report", "text": "profit", "pages": [5], "metadata": {"page_start": 5, "page_end": 6, "source": "report.pdf"}},
        ],
    )

    assert cosine_similarity(np.array([1.0, 0.0, 0.0]), index.embeddings).tolist() == [1.0, 0.0]
    results = search("question", index, FakeEmbedder(), top_k=2)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["chunk_id"] == "report_c0001"
    assert results[0]["similarity"] > 0
    assert results[0]["similarity"] > results[1]["similarity"]
    assert results[0]["page"] == 4
    assert results[0]["source"] == "report.pdf"
    assert results[0]["metadata"]["section"] == "Sales"


def test_search_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        search("question", VectorIndex(np.empty((0, 3), dtype=np.float32), []), FakeEmbedder(), 0)
