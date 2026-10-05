"""Token-window chunking that carries page provenance through boundaries."""

import re
from collections.abc import Iterable

from .models import Chunk, PageText

_TOKEN_PATTERN = re.compile(r"\S+")


def _tokens_by_page(pages: Iterable[PageText]) -> list[tuple[str, int]]:
    return [
        (match.group(0), page.page_number)
        for page in pages
        for match in _TOKEN_PATTERN.finditer(page.text)
    ]


def _choose_window_end(
    token_stream: list[tuple[str, int]], start: int, chunk_size: int
) -> int:
    """Prefer a sentence or page boundary near the configured token limit."""
    hard_end = min(start + chunk_size, len(token_stream))
    if hard_end == len(token_stream):
        return hard_end

    # Avoid sacrificing more than 20% of the requested context just to find a
    # boundary. Page ends are useful fallbacks for dense tables and headings.
    lower_bound = start + max(1, int(chunk_size * 0.8))
    for index in range(hard_end, lower_bound - 1, -1):
        token, page = token_stream[index - 1]
        next_page = token_stream[index][1] if index < len(token_stream) else page
        sentence_end = token.rstrip('"\u201d\u2019)]}').endswith((".", "!", "?"))
        page_end = chunk_size >= 100 and page != next_page
        if sentence_end or page_end:
            return index
    return hard_end


def chunk_pages(
    pages: Iterable[PageText],
    *,
    document_id: str,
    source_filename: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Create deterministic, overlapping chunks from cleaned page text.

    Tokens are whitespace-delimited. The end of each full-size chunk is moved
    to a nearby sentence (or, for normal-sized chunks, page) boundary when one
    is available, without reducing the chunk below 80% of its target size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    token_stream = _tokens_by_page(pages)
    chunks: list[Chunk] = []
    start = 0
    sequence = 1
    while start < len(token_stream):
        end = _choose_window_end(token_stream, start, chunk_size)
        window = token_stream[start:end]
        chunk_pages = list(dict.fromkeys(page for _, page in window))
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}_c{sequence:04d}",
                document_id=document_id,
                text=" ".join(token for token, _ in window),
                pages=chunk_pages,
                metadata={
                    "source": source_filename,
                    "page": chunk_pages[0],
                    "page_start": chunk_pages[0],
                    "page_end": chunk_pages[-1],
                },
            )
        )
        if end >= len(token_stream):
            break
        start = end - overlap
        sequence += 1
    return chunks
