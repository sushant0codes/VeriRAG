"""Hybrid dense and BM25 retrieval using reciprocal rank fusion."""

from __future__ import annotations

from typing import Iterable

from .bm25 import BM25Index
from .bm25 import search as bm25_search
from .index import VectorIndex
from .search import QueryEmbedder
from .search import search as dense_search

RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: Iterable[dict[str, object]],
    bm25_results: Iterable[dict[str, object]],
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Merge ranked results by chunk ID and score them with RRF."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    merged: dict[str, dict[str, object]] = {}
    for results, rank_key in (
        (dense_results, "dense_rank"),
        (bm25_results, "bm25_rank"),
    ):
        for result in results:
            chunk_id = str(result["chunk_id"])
            result_rank = int(result["rank"])
            if chunk_id not in merged:
                merged[chunk_id] = dict(result)
                merged[chunk_id]["rrf_score"] = 0.0
            chunk = merged[chunk_id]
            for key, value in result.items():
                if key not in chunk or chunk[key] is None:
                    chunk[key] = value
            chunk[rank_key] = result_rank
            chunk["rrf_score"] = float(chunk["rrf_score"]) + 1 / (RRF_K + result_rank)

    ranked = sorted(merged.values(), key=lambda chunk: float(chunk["rrf_score"]), reverse=True)
    for rank, chunk in enumerate(ranked[:top_k], start=1):
        chunk["rank"] = rank
    return ranked[:top_k]


def search(
    query: str,
    dense_index: VectorIndex,
    embedder: QueryEmbedder,
    bm25_index: BM25Index,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Run dense and BM25 retrieval for a query, then fuse their ranks."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return reciprocal_rank_fusion(
        dense_search(query, dense_index, embedder, top_k),
        bm25_search(query, bm25_index, top_k),
        top_k,
    )
