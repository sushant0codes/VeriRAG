"""Data models exchanged between ingestion stages."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PageText:
    """Clean text extracted from a one-indexed PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A citation-ready text chunk and its immutable source metadata."""

    chunk_id: str
    document_id: str
    text: str
    pages: list[int]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
