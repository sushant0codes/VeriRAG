# VeriRAG evidence workspace

Run the interactive local GUI from the repository root:

```powershell
python frontend/server.py
```

Open `http://127.0.0.1:8001`.

The workspace starts with the committed Apple annual-report chunks, builds a BM25 index for local retrieval, and exposes suggested questions from `evaluation/dataset.jsonl`. Uploading a text-readable PDF ingests it with `ingestion.pipeline` and switches the active corpus for the running process. Uploaded files are kept in a temporary directory and are removed when the server stops.

Set `VERIRAG_LLM_PROVIDER=ollama` (and optionally `VERIRAG_LLM_MODEL`) or configure `OPENAI_API_KEY` to replace the evidence-preview answer with the repository's grounded answer generator.
