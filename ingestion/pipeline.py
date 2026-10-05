"""Command-line orchestration and persistence for Phase 1 ingestion."""

import argparse
import json
from pathlib import Path

from .chunker import chunk_pages
from .cleaner import clean_text
from .loader import load_pdf_pages
from .models import Chunk, PageText


def document_id_from_path(pdf_path: str | Path) -> str:
    """Create a stable filesystem-safe document identifier from the filename."""
    stem = Path(pdf_path).stem.lower()
    normalized = "".join(char if char.isalnum() else "_" for char in stem)
    return normalized.strip("_") or "document"


def persist_chunks(chunks: list[Chunk], output_path: str | Path) -> Path:
    """Write citation-ready chunks as deterministic UTF-8 JSON Lines."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return path


def ingest_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> tuple[list[PageText], list[Chunk], Path]:
    """Extract, clean, chunk, and persist a PDF without losing page metadata."""
    path = Path(pdf_path)
    raw_pages = load_pdf_pages(path)
    clean_pages = [PageText(page.page_number, clean_text(page.text)) for page in raw_pages]
    document_id = document_id_from_path(path)
    chunks = chunk_pages(
        clean_pages,
        document_id=document_id,
        source_filename=path.name,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    target = Path(output_path) if output_path else path.with_suffix(".chunks.jsonl")
    return clean_pages, chunks, persist_chunks(chunks, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into citation-ready chunks.")
    parser.add_argument("pdf", help="Path to the source PDF")
    parser.add_argument("--output", help="Output JSONL path (defaults beside the PDF)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    pages, chunks, output = ingest_pdf(
        args.pdf, args.output, chunk_size=args.chunk_size, overlap=args.overlap
    )
    print(f"Document: {Path(args.pdf).name}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0]
        print("Sample:")
        print(f"Chunk ID: {sample.chunk_id}")
        print(f"Pages: {sample.pages}")
        print(f"Token count: {len(sample.text.split())}")
    print(f"Output: {output}")
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
