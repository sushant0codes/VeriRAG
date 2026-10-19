import pytest

from retrieval.hybrid import reciprocal_rank_fusion, search


def result(chunk_id, rank, text, **extra):
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "text": text,
        "pages": [rank],
        "page_start": rank,
        "page_end": rank,
        "source": "report.pdf",
        **extra,
    }


def test_rrf_scores_and_ranking_merge_duplicate_chunks() -> None:
    fused = reciprocal_rank_fusion(
        [result("a", 1, "dense first"), result("b", 2, "dense second")],
        [result("b", 1, "bm25 first"), result("a", 3, "bm25 third")],
    )

    assert [chunk["chunk_id"] for chunk in fused] == ["b", "a"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert [chunk["rank"] for chunk in fused] == [1, 2]


def test_rrf_preserves_chunk_data_and_source_specific_ranks() -> None:
    dense = result(
        "a", 1, "dense text", metadata={"section": "Sales"}, similarity=0.9
    )
    bm25 = result("b", 1, "bm25 text", bm25_score=2.5)

    fused = reciprocal_rank_fusion([dense], [bm25])

    assert fused[0]["metadata"] == {"section": "Sales"}
    assert fused[0]["similarity"] == 0.9
    assert fused[0]["dense_rank"] == 1
    assert "bm25_rank" not in fused[0]
    assert fused[1]["bm25_score"] == 2.5
    assert fused[1]["bm25_rank"] == 1
    assert "dense_rank" not in fused[1]
    assert fused[1]["pages"] == [1]
    assert fused[1]["source"] == "report.pdf"


def test_hybrid_search_passes_top_k_to_both_retrievers(monkeypatch) -> None:
    calls = []

    def fake_dense(query, index, embedder, top_k):
        calls.append(("dense", query, top_k))
        return [result("dense", 1, "dense")]

    def fake_bm25(query, index, top_k):
        calls.append(("bm25", query, top_k))
        return [result("bm25", 1, "bm25")]

    monkeypatch.setattr("retrieval.hybrid.dense_search", fake_dense)
    monkeypatch.setattr("retrieval.hybrid.bm25_search", fake_bm25)

    fused = search("revenue", object(), object(), object(), top_k=1)

    assert calls == [("dense", "revenue", 1), ("bm25", "revenue", 1)]
    assert len(fused) == 1


@pytest.mark.parametrize("top_k", [0, -1])
def test_hybrid_rejects_invalid_top_k(top_k) -> None:
    with pytest.raises(ValueError, match="top_k"):
        search("revenue", object(), object(), object(), top_k=top_k)

    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], [], top_k=top_k)
