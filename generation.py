"""Grounded answer generation from reranked retrieval chunks."""

from __future__ import annotations

import json
import os
import re
from typing import Iterable, Protocol
from urllib.request import Request, urlopen

DEFAULT_MODEL = "gpt-4.1-mini"
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
INSUFFICIENT_INFORMATION = "Insufficient information in the provided context."

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = rf"(?:{MONTHS})\s+\d{{1,2}},\s+\d{{4}}"
YEAR_HEADER = re.compile(rf"Years? ended\s+(?P<dates>(?:{DATE}\s*){{3}})")
NUMBER = r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?"
FINANCIAL_ROW = re.compile(
    rf"(?P<label>[A-Z][A-Za-z ,:/&()\-]{{1,80}}?)\s+"
    rf"(?P<values>{NUMBER}\s+{NUMBER}\s+{NUMBER})(?=\s+(?:[A-Z]|See\b)|$)"
)


class LLM(Protocol):
    """The small interface needed by grounded answer generation."""

    def generate(self, instructions: str, input_text: str) -> str: ...


class OpenAIResponsesLLM:
    """OpenAI Responses API client configured from environment variables."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, instructions: str, input_text: str) -> str:
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=input_text
        )
        return response.output_text


class OllamaLLM:
    """Ollama local HTTP API client."""

    def __init__(self, model: str, url: str = OLLAMA_URL) -> None:
        self.model = model
        self.url = url

    def generate(self, instructions: str, input_text: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": instructions,
                "prompt": input_text,
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urlopen(request) as response:
            return str(json.loads(response.read())["response"])


def load_llm_from_env() -> LLM:
    """Build the configured provider without exposing credentials in code."""
    provider = os.getenv("VERIRAG_LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return OllamaLLM(
            os.getenv("VERIRAG_LLM_MODEL", OLLAMA_DEFAULT_MODEL),
            os.getenv("VERIRAG_OLLAMA_URL", OLLAMA_URL),
        )
    if provider != "openai":
        raise ValueError(f"unsupported LLM provider: {provider}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")
    return OpenAIResponsesLLM(
        os.getenv("VERIRAG_LLM_MODEL", DEFAULT_MODEL),
        api_key,
        os.getenv("VERIRAG_LLM_BASE_URL"),
    )


def _citation(chunk: dict[str, object]) -> str:
    metadata = dict(chunk.get("metadata", {}))
    source = chunk.get("source") or metadata.get("source") or chunk.get("document_id")
    pages = chunk.get("pages") or []
    if not pages:
        start = chunk.get("page_start") or metadata.get("page_start") or metadata.get("page")
        end = chunk.get("page_end") or metadata.get("page_end") or start
        pages = [start] if start == end else [start, end]
    page_text = ", ".join(str(page) for page in pages if page is not None)
    return f"{source}, p. {page_text}" if len(pages) == 1 else f"{source}, pp. {page_text}"


def _financial_table_summary(text: str) -> str | None:
    """Summarize reliable three-year financial rows from flattened table text."""
    header = YEAR_HEADER.search(text)
    if not header:
        return None

    dates = re.findall(DATE, header.group("dates"))
    if len(dates) != 3:
        return None

    rows: list[str] = []
    for row in FINANCIAL_ROW.finditer(text, header.end()):
        values = re.findall(NUMBER, row.group("values"))
        if len(values) != 3:
            continue
        label = " ".join(row.group("label").split())
        clean_values = [re.sub(r"\s+", "", value) for value in values]
        pairs = "; ".join(
            f"{date} = {value}" for date, value in zip(dates, clean_values)
        )
        rows.append(f"{label}: {pairs}")
    return "\n".join(rows) or None


def _format_context(chunks: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for chunk in chunks:
        text = str(chunk["text"])
        summary = _financial_table_summary(text)
        if summary:
            formatted.append(
                f"[{_citation(chunk)}]\nOriginal chunk:\n{text}\n\n"
                f"Matched financial table values:\n{summary}"
            )
        else:
            formatted.append(f"[{_citation(chunk)}]\n{text}")
    return "\n\n".join(formatted)


def generate_answer(
    query: str, chunks: Iterable[dict[str, object]], llm: LLM | None = None
) -> str:
    """Answer from reranked chunks only and append their source citations."""
    candidates = list(chunks)
    if not candidates:
        return INSUFFICIENT_INFORMATION

    instructions = (
        "Answer only from the provided context. Do not use outside knowledge. "
        "Financial statements may list values for multiple years in sequence. "
        "Interpret financial tables by matching each row label to its column headers "
        "and dates before selecting a value. If the answer is explicitly present in "
        "the context, answer directly. Do not guess. "
        f"If the context does not support an answer, reply exactly: {INSUFFICIENT_INFORMATION}"
    )
    input_text = (
        f"Question: {query}\n\nContext:\n{_format_context(candidates)}\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = (llm or load_llm_from_env()).generate(instructions, input_text).strip()
    citations = "\n".join(f"- [{_citation(chunk)}]" for chunk in candidates)
    return f"{answer}\n\nSources:\n{citations}"
