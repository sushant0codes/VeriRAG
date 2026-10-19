"""BM25 keyword search over the Phase 1 chunk dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
def tokenize(text: str) -> list[str]:
    """Split text into lowercase keyword tokens for BM25."""
    return re.findall(r"\w+", text.lower())
@dataclass(frozen=True)
class BM25Index:
    """A BM25 model and its original Phase 1 chunk records."""

    bm25: BM25Okapi
    chunks: list[dict[str, object]]


def build_index(chunks_path: str | Path) -> BM25Index:
    """Load Phase 1 chunks and build one BM25 document per chunk."""
    source = Path(chunks_path)
    chunks = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return BM25Index(BM25Okapi([tokenize(str(chunk["text"])) for chunk in chunks]), chunks)


def search(query: str, index: BM25Index, top_k: int = 5) -> list[dict[str, object]]:
    """Rank chunks by BM25 score and return the top results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not index.chunks:
        return []

    scores = np.asarray(index.bm25.get_scores(tokenize(query)), dtype=float)
    result_indices = np.argsort(-scores, kind="stable")[:top_k]

    results: list[dict[str, object]] = []
    for rank, chunk_index in enumerate(result_indices, start=1):
        chunk = dict(index.chunks[int(chunk_index)])
        metadata = dict(chunk.get("metadata", {}))
        chunk.update(
            rank=rank,
            bm25_score=float(scores[chunk_index]),
            pages=chunk.get("pages", []),
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
            source=metadata.get("source"),
        )
        results.append(chunk)
    return results
