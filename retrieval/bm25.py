"""BM25 keyword search over the Phase 1 chunk dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi


QUESTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "of", "on", "or", "the", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Split text into lowercase keyword tokens for BM25."""
    return re.findall(r"\w+", text.lower())


def _match_token(token: str) -> str:
    """Normalize the small inflection needed for question-to-sentence matching."""
    return token[:-1] if token.endswith("s") and len(token) > 3 else token


def _sentence_match_key(query: str, chunk: dict[str, object]) -> tuple[int, int]:
    """Return focused-match signals for a chunk without replacing BM25 scoring.

    PDFs often yield broad, overlapping chunks. A sentence that contains the
    meaningful terms of a multi-word question is more useful than an unrelated
    table that merely repeats those words across several pages.
    """
    query_terms = [
        _match_token(token)
        for token in tokenize(query)
        if token not in QUESTION_WORDS and len(token) > 1
    ]
    if len(set(query_terms)) < 3:
        return (0, 0)

    best_coverage = 0
    best_ordered_match = 0
    for sentence in re.split(r"[.!?]+", str(chunk.get("text", ""))):
        sentence_tokens = [_match_token(token) for token in tokenize(sentence)]
        if not sentence_tokens:
            continue
        coverage = len(set(query_terms) & set(sentence_tokens))
        best_coverage = max(best_coverage, coverage)

        for position in range(len(sentence_tokens) - 1):
            first, second = sentence_tokens[position : position + 2]
            if first not in query_terms or second not in query_terms:
                continue
            remaining = sentence_tokens[position + 2 : position + 14]
            other_query_terms = set(query_terms) - {first, second}
            if other_query_terms & set(remaining):
                best_ordered_match = 1
                break

    return (best_ordered_match, best_coverage)
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
    # Keep BM25 as the primary relevance model, then prefer a focused sentence
    # for questions with three or more meaningful terms.  The page tie-breaker
    # makes duplicate passages from overlapping chunks deterministic.
    sentence_keys = [_sentence_match_key(query, chunk) for chunk in index.chunks]
    result_indices = sorted(
        range(len(index.chunks)),
        key=lambda chunk_index: (
            -sentence_keys[chunk_index][0],
            int(dict(index.chunks[chunk_index].get("metadata", {})).get("page_start", 0))
            if sentence_keys[chunk_index][0]
            else 0,
            -float(scores[chunk_index]),
        ),
    )[:top_k]

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
