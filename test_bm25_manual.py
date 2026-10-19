from pathlib import Path

from retrieval.bm25 import build_index, search

CHUNKS_PATH = Path("data/apple_annual_report.chunks.jsonl")

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"Chunk dataset not found: {CHUNKS_PATH}")

index = build_index(CHUNKS_PATH)

queries = [
    "What were Apple's total net sales?",
    "What was Apple's net income?",
    "iPhone revenue",
    "operating expenses",
]

print(f"Indexed chunks: {len(index.chunks)}")

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print("=" * 60)

    results = search(query, index, top_k=3)

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Pages: {result.get('pages')}")
        print(f"Source: {result.get('source')}")
        print(f"Text: {result['text'][:300]}")