import json

import pymupdf

from ingestion.pipeline import ingest_pdf


def test_ingest_pdf_extracts_pages_and_persists_chunks(tmp_path) -> None:
    source = tmp_path / "Annual Report.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "First page evidence.")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page evidence.")
    pdf.save(source)
    pdf.close()

    pages, chunks, output = ingest_pdf(source, chunk_size=3, overlap=1)

    assert len(pages) == 2
    assert chunks[0].document_id == "annual_report"
    assert chunks[0].metadata["source"] == "Annual Report.pdf"
    assert output.exists()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["chunk_id"] for row in rows] == [chunk.chunk_id for chunk in chunks]
