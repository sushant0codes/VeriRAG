import numpy as np
import pytest

from retrieval.reranker import rerank


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs
        return np.array(self.scores)


def candidates():
    return [
        {
            "chunk_id": "a",
            "text": "first",
            "rank": 1,
            "pages": [4],
            "source": "report.pdf",
            "metadata": {"section": "Sales"},
        },
        {
            "chunk_id": "b",
            "text": "second",
            "rank": 2,
            "pages": [5],
            "source": "report.pdf",
            "metadata": {"section": "Income"},
        },
        {"chunk_id": "c", "text": "third", "rank": 3, "pages": [6]},
    ]


def test_rerank_scores_pairs_and_sorts_descending(monkeypatch) -> None:
    model = FakeCrossEncoder([0.2, 0.9, 0.5])
    monkeypatch.setattr("retrieval.reranker._load_model", lambda _: model)

    results = rerank("net income", candidates(), top_k=2)

    assert [result["chunk_id"] for result in results] == ["b", "c"]
    assert [result["rank"] for result in results] == [1, 2]
    assert [result["rerank_score"] for result in results] == [0.9, 0.5]
    assert model.pairs == [
        ("net income", "first"),
        ("net income", "second"),
        ("net income", "third"),
    ]


def test_rerank_preserves_chunk_data_and_does_not_mutate_candidates(monkeypatch) -> None:
    model = FakeCrossEncoder([0.1, 0.9, 0.2])
    monkeypatch.setattr("retrieval.reranker._load_model", lambda _: model)
    original = candidates()

    result = rerank("income", original, top_k=1)[0]

    assert result["chunk_id"] == "b"
    assert result["pages"] == [5]
    assert result["source"] == "report.pdf"
    assert result["metadata"] == {"section": "Income"}
    assert result["hybrid_rank"] == 2
    assert original[1]["rank"] == 2
    assert "rerank_score" not in original[1]


def test_rerank_returns_empty_without_loading_a_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "retrieval.reranker._load_model",
        lambda _: pytest.fail("model should not load for empty candidates"),
    )

    assert rerank("income", []) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_rerank_rejects_invalid_top_k(top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        rerank("income", candidates(), top_k=top_k)
