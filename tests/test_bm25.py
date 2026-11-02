import json

import pytest

from retrieval.bm25 import build_index, search


def write_chunks(path) -> None:
    rows = [
        {
            "chunk_id": "report_c0001",
            "document_id": "report",
            "text": "Apple iPhone revenue was strong.",
            "pages": [4],
            "metadata": {"page_start": 4, "page_end": 4, "source": "report.pdf", "section": "Sales"},
        },
        {
            "chunk_id": "report_c0002",
            "document_id": "report",
            "text": "Operating expenses increased.",
            "pages": [5, 6],
            "metadata": {"page_start": 5, "page_end": 6, "source": "report.pdf"},
        },
        {
            "chunk_id": "report_c0003",
            "document_id": "report",
            "text": "iPhone revenue revenue revenue rose.",
            "pages": [7],
            "metadata": {"page_start": 7, "page_end": 7, "source": "report.pdf"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_bm25_indexes_all_chunks_and_preserves_metadata(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)

    index = build_index(chunks_path)
    results = search("iPhone revenue", index)

    assert len(index.chunks) == 3
    assert results[0]["chunk_id"] == "report_c0003"
    assert results[0]["metadata"]["source"] == "report.pdf"
    assert results[0]["pages"] == [7]
    assert results[0]["page_start"] == 7
    assert results[0]["page_end"] == 7
    assert results[0]["source"] == "report.pdf"


def test_bm25_search_sorts_scores_and_honors_top_k(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)

    results = search("iPhone revenue", build_index(chunks_path), top_k=2)

    assert len(results) == 2
    assert [result["rank"] for result in results] == [1, 2]
    assert results[0]["bm25_score"] >= results[1]["bm25_score"]


def test_bm25_search_rejects_invalid_top_k(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)

    with pytest.raises(ValueError, match="top_k"):
        search("revenue", build_index(chunks_path), top_k=0)


def test_bm25_prefers_a_focused_sentence_for_a_multi_term_question(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    rows = [
        {
            "chunk_id": "report_c0001",
            "document_id": "report",
            "text": "Fiscal results and year-end reporting are included in this table.",
            "pages": [18],
            "metadata": {"page_start": 18, "page_end": 18, "source": "report.pdf"},
        },
        {
            "chunk_id": "report_c0002",
            "document_id": "report",
            "text": "The fiscal year ends on the last Saturday of September.",
            "pages": [4],
            "metadata": {"page_start": 4, "page_end": 4, "source": "report.pdf"},
        },
    ]
    chunks_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    results = search("What is the fiscal-year end?", build_index(chunks_path), top_k=2)

    assert results[0]["chunk_id"] == "report_c0002"
