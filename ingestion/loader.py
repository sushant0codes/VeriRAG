"""PDF extraction using PyMuPDF."""

from pathlib import Path

import pymupdf

from .models import PageText


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract text from each PDF page, retaining one-indexed page numbers."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    pages: list[PageText] = []
    with pymupdf.open(path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(PageText(page_number=index, text=page.get_text("text")))
    return pages
