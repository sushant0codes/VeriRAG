"""Claim-level verification for answers grounded in retrieved chunks.

This module is deliberately a post-generation stage: it does not alter answer
generation or retrieval.  It asks the configured generation LLM to assess one
claim at a time against the retrieved text, then returns only source metadata
from the supplied chunks (never model-invented citations).
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from generation import LLM, load_llm_from_env

_SOURCES_HEADING = re.compile(r"\n\s*Sources:\s*(?:\n|$)", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE = re.compile(r"\s+")
_STATUSES = {"supported", "unsupported", "uncertain"}
_YEAR_COLUMNS = re.compile(r"\b(?P<years>20\d{2}(?:\s+20\d{2}){2,})\b")
_NUMBER = r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?"
_FINANCIAL_ROW = re.compile(
    rf"(?P<label>[A-Z][A-Za-z ,:/&()\-]{{1,80}}?)\s+"
    rf"(?P<values>{_NUMBER}\s+{_NUMBER}\s+{_NUMBER})(?=\s+(?:[A-Z]|See\b)|$)"
)
_CLAIM_YEAR = re.compile(r"\b(20\d{2})\b")
_CLAIM_AMOUNT = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>millions?|thousands?|billions?)\b",
    re.IGNORECASE,
)


def _answer_body(answer: str) -> str:
    """Remove the source list appended by ``generate_answer``."""
    return _SOURCES_HEADING.split(answer, maxsplit=1)[0].strip()


def _extract_claims(answer: str) -> list[str]:
    """Extract sentence-level factual claims from an answer, excluding Sources."""
    body = _answer_body(answer)
    if not body:
        return []
    return [sentence.strip() for sentence in _SENTENCES.split(body) if sentence.strip()]


def _source_record(chunk: dict[str, object]) -> dict[str, object]:
    """Return citation metadata without exposing the full evidence text again."""
    metadata = dict(chunk.get("metadata", {}))
    pages = chunk.get("pages") or []
    if not pages:
        start = chunk.get("page_start") or metadata.get("page_start") or metadata.get("page")
        end = chunk.get("page_end") or metadata.get("page_end") or start
        pages = [start] if start == end else [start, end]
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "source": chunk.get("source") or metadata.get("source") or chunk.get("document_id"),
        "pages": [page for page in pages if page is not None],
        "metadata": metadata,
    }


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _table_units(context: str, label: str) -> str | None:
    """Find one unambiguous unit that applies to a financial row label."""
    unit_pairs = re.findall(
        r"(?:\(|\band\s+)([A-Za-z][A-Za-z ]*?)\s+in\s+"
        r"(millions?|thousands?|billions?)\b",
        context,
        flags=re.IGNORECASE,
    )
    label_text = _normalise(label.split(":")[-1])
    matching_units = {
        unit.lower().rstrip("s")
        for metric, unit in unit_pairs
        if _normalise(metric) in label_text or label_text in _normalise(metric)
    }
    if len(matching_units) == 1:
        return matching_units.pop()
    all_units = {unit.lower().rstrip("s") for _, unit in unit_pairs}
    if len(all_units) == 1:
        return all_units.pop()
    standalone = re.findall(r"\bin\s+(millions?|thousands?|billions?)\b", context, re.I)
    unique_standalone = {unit.lower().rstrip("s") for unit in standalone}
    return unique_standalone.pop() if len(unique_standalone) == 1 else None


def _financial_table_entries(
    text: str, chunk: dict[str, object]
) -> list[dict[str, object]]:
    """Parse only unambiguous flattened year-column financial rows."""
    entries: list[dict[str, object]] = []
    for header in _YEAR_COLUMNS.finditer(text):
        years = header.group("years").split()
        preceding_text = text[max(0, header.start() - 300) : header.start()]
        table_text = text[header.end() : header.end() + 3000]
        for row in _FINANCIAL_ROW.finditer(table_text):
            values = re.findall(_NUMBER, row.group("values"))
            if len(values) != len(years):
                continue
            label = " ".join(row.group("label").split())
            unit = _table_units(preceding_text, label)
            if unit is None:
                continue
            entries.append(
                {
                    "label": label,
                    "years": years,
                    "values": [_WHITESPACE.sub("", value) for value in values],
                    "unit": unit,
                    "chunk": chunk,
                }
            )
    return entries


def _financial_table_relationships(text: str) -> list[str]:
    """Mechanically expose year-column relationships in flattened financial text.

    The source text remains the evidence.  These lines only make its left-to-
    right header/value relationship explicit for a verifier that cannot see
    table layout in a PDF extraction.
    """
    relationships: list[str] = []
    placeholder = {"chunk_id": ""}
    for entry in _financial_table_entries(text, placeholder):
        pairs = "; ".join(
            f"{year} = {value}"
            for year, value in zip(entry["years"], entry["values"])
        )
        relationship = f"{entry['label']}: {pairs}; units: {entry['unit']}"
        if relationship not in relationships:
            relationships.append(relationship)
    return relationships


def _source_records(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return source records once per chunk, retaining supplied metadata."""
    seen: set[str] = set()
    records: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        key = str(chunk.get("chunk_id", index))
        if key not in seen:
            seen.add(key)
            records.append(_source_record(chunk))
    return records


def _verify_financial_claim(
    claim: str, chunks: list[dict[str, object]]
) -> dict[str, object] | None:
    """Verify a claim deterministically only with an exact table mapping."""
    year_match = _CLAIM_YEAR.search(claim)
    amount_match = _CLAIM_AMOUNT.search(claim)
    if not year_match or not amount_match:
        return None
    year = year_match.group(1)
    claim_value = _WHITESPACE.sub("", amount_match.group("value"))
    claim_unit = amount_match.group("unit").lower().rstrip("s")
    entries = [
        entry
        for chunk in chunks
        for entry in _financial_table_entries(str(chunk.get("text", "")), chunk)
    ]
    if not entries:
        return None

    claim_text = _normalise(claim)
    matching_rows = [
        entry
        for entry in entries
        if _normalise(str(entry["label"]).split(":")[-1]) in claim_text
    ]
    if not matching_rows:
        if year not in {table_year for entry in entries for table_year in entry["years"]}:
            return {
                "status": "uncertain",
                "supporting_sources": _source_records(chunks),
                "reason": f"The financial tables do not contain a {year} column for this claim.",
            }
        return None

    matching_units = [entry for entry in matching_rows if entry["unit"] == claim_unit]
    if not matching_units:
        return {
            "status": "uncertain",
            "supporting_sources": _source_records(
                [entry["chunk"] for entry in matching_rows]
            ),
            "reason": "The matching financial row does not have an unambiguous matching unit.",
        }
    year_entries = [entry for entry in matching_units if year in entry["years"]]
    if not year_entries:
        return {
            "status": "uncertain",
            "supporting_sources": _source_records(
                [entry["chunk"] for entry in matching_units]
            ),
            "reason": f"The matching financial row has no {year} column.",
        }
    values = {entry["values"][entry["years"].index(year)] for entry in year_entries}
    if len(values) != 1:
        return {
            "status": "uncertain",
            "supporting_sources": _source_records(
                [entry["chunk"] for entry in year_entries]
            ),
            "reason": "The matching financial rows provide conflicting values.",
        }
    source_chunks = [entry["chunk"] for entry in year_entries]
    table_value = values.pop()
    status = "supported" if table_value == claim_value else "unsupported"
    return {
        "status": status,
        "supporting_sources": _source_records(source_chunks),
        "reason": (
            f"The {year} value for {year_entries[0]['label']} is {table_value} "
            f"{year_entries[0]['unit']}."
        ),
    }


def _evidence(chunks: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        relationships = _financial_table_relationships(text)
        table_summary = (
            "\nMechanically matched financial table values (headers map left-to-right):\n"
            + "\n".join(relationships)
            if relationships
            else ""
        )
        formatted.append(
            f"Chunk ID: {chunk.get('chunk_id', f'chunk_{index}')}\n"
            f"Document: {chunk.get('document_id', '')}\n"
            f"Source text:\n{text}{table_summary}"
        )
    return "\n\n".join(formatted)


def _parse_verdict(response: str) -> dict[str, object] | None:
    """Extract and validate one complete verdict object from an LLM response."""
    if not isinstance(response, str):
        return None

    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", response):
        try:
            result, _ = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or result.get("status") not in _STATUSES:
            continue
        source_ids = result.get("supporting_chunk_ids", [])
        if isinstance(source_ids, list) and all(
            isinstance(value, str) for value in source_ids
        ):
            return result
    return None


def verify_claims(
    answer: str, chunks: Iterable[dict[str, object]], llm: LLM | None = None
) -> list[dict[str, object]]:
    """Assess every answer claim against retrieved chunk text.

    A result is returned for each sentence-level claim with ``claim``,
    ``status``, ``supporting_sources``, and ``reason``.  Provider failures,
    malformed replies, and missing evidence deliberately produce ``uncertain``;
    they are never interpreted as support.
    """
    claims = _extract_claims(answer)
    candidates = list(chunks)
    if not claims:
        return []
    if not candidates:
        return [
            {
                "claim": claim,
                "status": "uncertain",
                "supporting_sources": [],
                "reason": "No retrieved source chunks were provided for verification.",
            }
            for claim in claims
        ]

    chunks_by_id = {
        str(chunk.get("chunk_id", f"chunk_{index}")): chunk
        for index, chunk in enumerate(candidates)
    }
    response_contract = (
        "Return exactly one JSON object with: status (supported, unsupported, or uncertain), "
        "supporting_chunk_ids (an array of Chunk ID values from the evidence), and reason. "
    )
    support_instructions = (
        "Assess only whether the supplied evidence directly supports the claim. Do not use outside "
        "knowledge. Return supported only for direct support; otherwise return uncertain. "
        + response_contract
    )
    contradiction_instructions = (
        "Assess only whether the supplied evidence directly contradicts the claim. Do not use outside "
        "knowledge. Return unsupported only for direct, incompatible evidence; otherwise return "
        "uncertain. Lack of mention and related but different entities, products, actions, or attributes "
        "are never contradictions. Evaluate negated claims as their actual proposition, not by keyword "
        "overlap. "
        + response_contract
    )
    evidence = _evidence(candidates)
    verifier = llm
    results: list[dict[str, object]] = []
    for claim in claims:
        financial_result = _verify_financial_claim(claim, candidates)
        if financial_result is not None:
            results.append({"claim": claim, **financial_result})
            continue
        if verifier is None:
            verifier = load_llm_from_env()
        try:
            support_response = verifier.generate(
                support_instructions,
                f"Claim: {claim}\n\nEvidence:\n{evidence}",
            )
            contradiction_response = verifier.generate(
                contradiction_instructions,
                f"Claim: {claim}\n\nEvidence:\n{evidence}",
            )
            support_verdict = _parse_verdict(support_response)
            contradiction_verdict = _parse_verdict(contradiction_response)
        except Exception:
            support_verdict = None
            contradiction_verdict = None
        if support_verdict is None or contradiction_verdict is None:
            results.append(
                {
                    "claim": claim,
                    "status": "uncertain",
                    "supporting_sources": [],
                    "reason": "The verifier did not return a valid evidence assessment.",
                }
            )
            continue

        support_ids = [
            source_id
            for source_id in support_verdict["supporting_chunk_ids"]
            if source_id in chunks_by_id
        ]
        contradiction_ids = [
            source_id
            for source_id in contradiction_verdict["supporting_chunk_ids"]
            if source_id in chunks_by_id
        ]
        is_supported = support_verdict["status"] == "supported" and bool(support_ids)
        is_contradicted = (
            contradiction_verdict["status"] == "unsupported"
            and bool(contradiction_ids)
        )
        if is_supported and is_contradicted:
            status = "uncertain"
            source_ids = support_ids + contradiction_ids
            reason = "The verifier reported both direct support and direct contradiction."
        elif is_supported:
            status = "supported"
            source_ids = support_ids
            reason = str(support_verdict.get("reason", "")) or "Direct support found."
        elif is_contradicted:
            status = "unsupported"
            source_ids = contradiction_ids
            reason = (
                str(contradiction_verdict.get("reason", ""))
                or "Direct contradiction found."
            )
        else:
            status = "uncertain"
            source_ids = support_ids + contradiction_ids
            reason = "The verifier found neither direct support nor direct contradiction."
        results.append(
            {
                "claim": claim,
                "status": status,
                "supporting_sources": [
                    _source_record(chunks_by_id[source_id])
                    for source_id in dict.fromkeys(source_ids)
                ],
                "reason": reason,
            }
        )
    return results
