"""Cosine-similarity search over a local VectorIndex."""

from typing import Protocol, Sequence

import numpy as np

from .index import VectorIndex


class QueryEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def cosine_similarity(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity from one query vector to all document vectors."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    documents = np.asarray(vectors, dtype=np.float32)
    if documents.ndim != 2 or documents.shape[1] != query.size:
        raise ValueError("query and document embedding dimensions must match")
    denominator = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
    return np.divide(
        documents @ query,
        denominator,
        out=np.zeros(len(documents), dtype=np.float32),
        where=denominator != 0,
    )


def search(
    query: str, index: VectorIndex, embedder: QueryEmbedder, top_k: int = 5
) -> list[dict[str, object]]:
    """Embed a query, rank chunks by cosine similarity, and return top results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not index.chunks:
        return []

    query_vectors = np.asarray(embedder.embed([query]), dtype=np.float32)
    if query_vectors.shape[0] != 1:
        raise ValueError("the embedder must return one query embedding")
    scores = cosine_similarity(query_vectors[0], index.embeddings)
    result_indices = np.argsort(-scores, kind="stable")[:top_k]

    results: list[dict[str, object]] = []
    for rank, vector_index in enumerate(result_indices, start=1):
        chunk = dict(index.chunks[int(vector_index)])
        metadata = dict(chunk.get("metadata", {}))
        chunk.update(
            rank=rank,
            similarity=float(scores[vector_index]),
            pages=chunk.get("pages", []),
            page=metadata.get("page"),
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
            source=metadata.get("source"),
        )
        results.append(chunk)
    return results
