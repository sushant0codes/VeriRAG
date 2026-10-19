from pathlib import Path

from retrieval.embedder import Embedder
from retrieval.index import build_index as build_dense_index
from retrieval.bm25 import build_index as build_bm25_index
from retrieval.hybrid import search as hybrid_search

CHUNKS_PATH = Path("data/apple_annual_report.chunks.jsonl")

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CHUNKS_PATH}")

print("Building dense index...")
embedder = Embedder()
dense_index = build_dense_index(CHUNKS_PATH, embedder)

print("Building BM25 index...")
bm25_index = build_bm25_index(CHUNKS_PATH)

print(f"\nDense chunks: {len(dense_index.chunks)}")
print(f"BM25 chunks: {len(bm25_index.chunks)}")

queries = [
    "What were Apple's total net sales?",
    "What was Apple's net income?",
    "iPhone revenue",
    "operating expenses",
]

for query in queries:
    print(f"\n{'=' * 65}")
    print(f"QUERY: {query}")
    print("=" * 65)

    results = hybrid_search(
        query=query,
        dense_index=dense_index,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=3,
    )

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Dense Rank: {result.get('dense_rank', 'N/A')}")
        print(f"BM25 Rank: {result.get('bm25_rank', 'N/A')}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Pages: {result.get('pages')}")
        print(f"Source: {result.get('source')}")
        print(f"Text: {result['text'][:1000]}")