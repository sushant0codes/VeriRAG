"""Persistence and loading for a small, local vector index."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class VectorIndex:
    """Vectors and their original Phase 1 JSON records, in matching order."""

    embeddings: np.ndarray
    chunks: list[dict[str, object]]
    source_digest: str = ""

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if len(self.embeddings) != len(self.chunks):
            raise ValueError("every chunk must have exactly one embedding")

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, path: str | Path) -> Path:
        """Persist vectors and their exact source records in one local file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            embeddings=self.embeddings.astype(np.float32),
            chunks_json=json.dumps(self.chunks, ensure_ascii=False),
            source_digest=self.source_digest,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        with np.load(Path(path), allow_pickle=False) as stored:
            chunks = json.loads(str(stored["chunks_json"].item()))
            digest = str(stored["source_digest"].item())
            return cls(np.asarray(stored["embeddings"], dtype=np.float32), chunks, digest)


def _read_chunks(path: str | Path) -> tuple[list[dict[str, object]], str]:
    source = Path(path)
    payload = source.read_bytes()
    chunks = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    return chunks, hashlib.sha256(payload).hexdigest()


def build_index(chunks_path: str | Path, embedder: EmbeddingProvider) -> VectorIndex:
    """Generate precisely one vector per Phase 1 chunk."""
    chunks, digest = _read_chunks(chunks_path)
    vectors = np.asarray(embedder.embed([str(chunk["text"]) for chunk in chunks]), dtype=np.float32)
    return VectorIndex(vectors, chunks, digest)


def load_or_build_index(
    chunks_path: str | Path, index_path: str | Path, embedder: EmbeddingProvider
) -> VectorIndex:
    """Reuse a matching on-disk index; rebuild only when source chunks changed."""
    _, digest = _read_chunks(chunks_path)
    target = Path(index_path)
    if target.exists():
        loaded = VectorIndex.load(target)
        if loaded.source_digest == digest:
            return loaded
    index = build_index(chunks_path, embedder)
    index.save(target)
    return index
