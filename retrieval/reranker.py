"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load one cross-encoder instance per process."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str, candidates: Iterable[dict[str, object]], top_k: int = 5
) -> list[dict[str, object]]:
    """Score retrieved chunks against a query and return the most relevant ones."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    chunks = list(candidates)
    if not chunks:
        return []

    model = _load_model(CROSS_ENCODER_MODEL)
    scores = np.asarray(
        model.predict([(query, str(chunk["text"])) for chunk in chunks]), dtype=float
    ).reshape(-1)
    if len(scores) != len(chunks):
        raise ValueError("the cross-encoder must return one score per candidate")

    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
    results: list[dict[str, object]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        chunk = dict(chunks[int(index)])
        if "rank" in chunk:
            chunk["hybrid_rank"] = chunk["rank"]
        chunk.update(rank=rank, rerank_score=float(scores[index]))
        results.append(chunk)
    return results
