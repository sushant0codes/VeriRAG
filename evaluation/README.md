# Phase 7 evaluation

This is a compact, manually reviewable evaluation for the committed Apple annual-report corpus. Every ID and reference answer in the two JSONL files is a **draft label requiring human review**.

Run it from the repository root:

```powershell
python evaluation/benchmark.py
```

The script uses the existing persisted dense index, existing BM25 implementation, reciprocal-rank fusion, and existing cross-encoder reranker. It scores only the 17 answerable retrieval questions in `dataset.jsonl`; the explicitly unanswerable question is retained for answer-quality review but has no relevant chunk and is excluded from Recall/MRR.

Metrics:

- **Recall@5**: fraction of questions with at least one labeled relevant chunk in the first five results.
- **MRR@5**: mean reciprocal rank of the first labeled relevant chunk, zero if none appears in the first five.
- **Retrieval latency**: mean and p95 wall-clock milliseconds per query, after the embedding and cross-encoder models are warmed. The reranked row includes cross-encoder inference time.

`verification_dataset.jsonl` has three draft examples each for `supported`, `unsupported` (directly contradicted by supplied evidence), and `uncertain` (not established by supplied evidence). The latter is deliberately distinct from contradiction. It is not automatically scored by the benchmark: the production verifier needs a configured LLM for most natural-language claims, so automatic values would be provider- and prompt-dependent. Use it for controlled verifier runs after selecting an LLM, and calculate per-label precision/recall/F1 from its returned statuses.

Interpret retrieval values as a small regression signal, not a claim of general QA quality: the corpus has only one annual report, labels are draft, and judged relevance is limited to the listed chunk IDs. Compare runs on the same machine and model cache; latency is especially environment-dependent.
