"""Local web server for the VeriRAG evidence workspace.

Run from the repository root with ``python frontend/server.py`` and open
http://127.0.0.1:8001. The server keeps uploaded PDFs in a temporary folder
and uses the project's ingestion and BM25 retrieval modules directly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
from threading import Thread
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
# Running ``python frontend/server.py`` puts only ``frontend/`` on the module
# search path. Add the repository root so the ingestion and retrieval packages
# are available regardless of how the server is launched.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND = ROOT / "frontend"
DATASET = ROOT / "evaluation" / "dataset.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "apple_annual_report.chunks.jsonl"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class Workspace:
    """Own the active corpus and optional LLM-backed answer path."""

    def __init__(self) -> None:
        self._temporary_directory = TemporaryDirectory(prefix="verirag-ui-")
        self._chunks_path = DEFAULT_CHUNKS
        self._source_name = "Apple annual report"
        self._bm25_index: Any | None = None

    def _index(self) -> Any:
        if self._bm25_index is None:
            from retrieval.bm25 import build_index

            self._bm25_index = build_index(self._chunks_path)
        return self._bm25_index

    def upload(self, filename: str, payload: bytes) -> dict[str, object]:
        """Persist one local PDF, ingest it, and make it the active corpus."""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Upload a PDF file.")
        if not payload.startswith(b"%PDF"):
            raise ValueError("The uploaded file does not look like a PDF.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise ValueError("PDF uploads are limited to 25 MB.")

        safe_name = Path(filename).name
        pdf_path = Path(self._temporary_directory.name) / safe_name
        pdf_path.write_bytes(payload)
        from ingestion.pipeline import ingest_pdf

        pages, chunks, chunks_path = ingest_pdf(pdf_path)
        if not chunks:
            raise ValueError("No readable text chunks were found in that PDF.")
        self._chunks_path = chunks_path
        self._source_name = safe_name
        self._bm25_index = None
        return {"source": safe_name, "pages": len(pages), "chunks": len(chunks)}

    def query(self, question: str) -> dict[str, object]:
        """Retrieve evidence, with optional grounded generation when configured."""
        clean_question = " ".join(question.split())
        if not clean_question:
            raise ValueError("Write a question before searching.")
        from retrieval.bm25 import search

        evidence = search(clean_question, self._index(), top_k=4)
        answer, mode = self._answer(clean_question, evidence)
        return {
            "question": clean_question,
            "answer": answer,
            "mode": mode,
            "source": self._source_name,
            "evidence": [self._evidence_record(chunk, clean_question) for chunk in evidence],
        }

    def _answer(self, question: str, evidence: list[dict[str, object]]) -> tuple[str, str]:
        """Prefer the configured LLM, but keep the GUI useful without credentials."""
        if os.getenv("OPENAI_API_KEY") or os.getenv("VERIRAG_LLM_PROVIDER") == "ollama":
            try:
                from generation import generate_answer

                return generate_answer(question, evidence), "grounded generation"
            except Exception:
                pass
        if not evidence:
            return "No matching evidence was found in the active document.", "retrieval preview"
        excerpt = " ".join(str(evidence[0]["text"]).split())
        return (
            "LLM generation is not configured. This is the highest-ranked evidence excerpt:\n\n"
            + excerpt[:900]
            + ("…" if len(excerpt) > 900 else ""),
            "retrieval preview",
        )

    @staticmethod
    def _evidence_record(chunk: dict[str, object], question: str) -> dict[str, object]:
        metadata = dict(chunk.get("metadata", {}))
        pages = chunk.get("pages") or [metadata.get("page")]
        text = " ".join(str(chunk.get("text", "")).split())
        ignored_terms = {"what", "which", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an", "of", "for", "to", "in"}
        query_terms = {term.lower() for term in re.findall(r"\w+", question) if len(term) > 1 and term.lower() not in ignored_terms}
        sentences = re.split(r"(?<=[.!?])\s+", text)
        excerpt = max(
            sentences,
            key=lambda sentence: len(query_terms & set(re.findall(r"\w+", sentence.lower()))),
            default=text,
        )
        return {
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source") or metadata.get("source") or chunk.get("document_id"),
            "pages": [page for page in pages if page is not None],
            "score": round(float(chunk.get("bm25_score", 0)), 3),
            "excerpt": text[:650] + ("…" if len(text) > 650 else ""),
            "excerpt": excerpt[:650] + ("…" if len(excerpt) > 650 else ""),
        }


WORKSPACE = Workspace()


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    """An IPv6 loopback companion for browsers that prefer ``::1``."""

    address_family = socket.AF_INET6


def suggested_questions() -> list[str]:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    return [str(row["question"]) for row in rows[:6]]


class Handler(BaseHTTPRequestHandler):
    """Serve static UI assets and a deliberately small local JSON API."""

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/suggestions":
            self._json({"questions": suggested_questions(), "source": WORKSPACE._source_name})
            return
        if path in {"/", "/index.html"}:
            self._file(FRONTEND / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.css":
            self._file(FRONTEND / "app.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._file(FRONTEND / "app.js", "application/javascript; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/query":
                body = self._body()
                self._json(WORKSPACE.query(str(json.loads(body)["question"])))
                return
            if self.path == "/api/upload":
                payload = self._body()
                self._json(WORKSPACE.upload(self.headers.get("X-Filename", "upload.pdf"), payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - protects the browser from server tracebacks
            self._json({"error": f"Unable to process this request: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _body(self) -> bytes:
        size = int(self.headers.get("Content-Length", "0"))
        if size > MAX_UPLOAD_BYTES:
            raise ValueError("Request is too large.")
        return self.rfile.read(size)

    def _file(self, path: Path, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        shutil.copyfileobj(path.open("rb"), self.wfile)

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    servers: list[ThreadingHTTPServer] = []
    for server_type, address in (
        (ThreadingHTTPServer, ("127.0.0.1", 8001)),
        (IPv6ThreadingHTTPServer, ("::1", 8001)),
    ):
        try:
            servers.append(server_type(address, Handler))
        except OSError:
            continue
    if not servers:
        raise RuntimeError("Unable to bind the VeriRAG UI to localhost:8001.")

    for server in servers[1:]:
        Thread(target=server.serve_forever, daemon=True).start()
    print("VeriRAG UI: http://localhost:8001")
    servers[0].serve_forever()
