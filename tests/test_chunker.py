from ingestion.chunker import chunk_pages
from ingestion.models import PageText


def test_chunker_preserves_overlap_and_cross_page_metadata() -> None:
    pages = [PageText(1, "one two three"), PageText(2, "four five six seven")]
    chunks = chunk_pages(
        pages, document_id="report", source_filename="report.pdf", chunk_size=4, overlap=2
    )

    assert [chunk.chunk_id for chunk in chunks] == ["report_c0001", "report_c0002", "report_c0003"]
    assert chunks[0].text == "one two three four"
    assert chunks[0].pages == [1, 2]
    assert chunks[1].text == "three four five six"
    assert chunks[1].pages == [1, 2]
    assert chunks[2].text == "five six seven"
    assert chunks[2].pages == [2]
    assert all(chunk.metadata["source"] == "report.pdf" for chunk in chunks)
    assert chunks[0].metadata["page_start"] == 1
    assert chunks[0].metadata["page_end"] == 2


def test_chunker_prefers_a_sentence_boundary_near_the_limit() -> None:
    chunks = chunk_pages(
        [PageText(1, "One two three. Four five six. Seven eight nine.")],
        document_id="report",
        source_filename="report.pdf",
        chunk_size=8,
        overlap=2,
    )

    assert chunks[0].text == "One two three. Four five six."
