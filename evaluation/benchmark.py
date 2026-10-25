"""Compact retrieval benchmark for the committed Apple annual-report index.

Run from the repository root with ``python evaluation/benchmark.py``.  It
measures the existing retrieval implementations only; it does not rebuild the
index or call an answer-generation provider.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.bm25 import build_index as build_bm25_index
from retrieval.bm25 import search as bm25_search
from retrieval.embedder import Embedder
from retrieval.hybrid import search as hybrid_search
from retrieval.index import VectorIndex
from retrieval.reranker import rerank
from retrieval.search import search as dense_search

DATASET = ROOT / "evaluation" / "dataset.jsonl"
CHUNKS = ROOT / "data" / "apple_annual_report.chunks.jsonl"
VECTORS = ROOT / "data" / "apple_annual_report.vectors.npz"
TOP_K = 5
RERANK_CANDIDATES = 20


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def score_query(result_ids: list[str], relevant_ids: set[str]) -> tuple[float, float]:
    """Return Recall@5 and reciprocal rank, treating any relevant chunk as a hit."""
    for rank, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def evaluate(name: str, rows: list[dict[str, object]], retrieve: Callable[[str], list[dict[str, object]]]) -> None:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    for row in rows:
        started = time.perf_counter()
        results = retrieve(str(row["question"]))
        latencies_ms.append((time.perf_counter() - started) * 1000)
        recall, reciprocal_rank = score_query(
            [str(result["chunk_id"]) for result in results],
            set(row["relevant_chunk_ids"]),
        )
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
    print(
        f"{name:20} Recall@5={statistics.fmean(recalls):.3f} "
        f"MRR@5={statistics.fmean(reciprocal_ranks):.3f} "
        f"latency_ms_mean={statistics.fmean(latencies_ms):.1f} "
        f"latency_ms_p95={sorted(latencies_ms)[max(0, math.ceil(.95 * len(latencies_ms)) - 1)]:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense, BM25, hybrid, and reranked hybrid retrieval.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.dataset) if row["answerability"] == "answerable"]
    if not rows:
        raise ValueError("The dataset has no answerable rows to score.")

    dense_index = VectorIndex.load(VECTORS)
    bm25_index = build_bm25_index(CHUNKS)
    embedder = Embedder()
    # Warm lazy-loaded models separately so per-query retrieval timing measures
    # retrieval/inference rather than one-time model initialization.
    embedder.embed([str(rows[0]["question"])])
    warm_query = str(rows[0]["question"])
    rerank(
        warm_query,
        hybrid_search(
            warm_query, dense_index, embedder, bm25_index, RERANK_CANDIDATES
        ),
        TOP_K,
    )

    evaluate("dense", rows, lambda query: dense_search(query, dense_index, embedder, TOP_K))
    evaluate("bm25", rows, lambda query: bm25_search(query, bm25_index, TOP_K))
    evaluate("hybrid RRF", rows, lambda query: hybrid_search(query, dense_index, embedder, bm25_index, TOP_K))
    evaluate(
        "hybrid + reranking",
        rows,
        lambda query: rerank(
            query,
            hybrid_search(query, dense_index, embedder, bm25_index, RERANK_CANDIDATES),
            TOP_K,
        ),
    )
    print(f"Scored {len(rows)} answerable draft questions; the unanswerable row is excluded from retrieval metrics.")


if __name__ == "__main__":
    main()
