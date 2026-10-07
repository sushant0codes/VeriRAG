"""Local dense semantic retrieval for Phase 2."""

from .embedder import BGE_SMALL_EN_V15, Embedder
from .index import VectorIndex, load_or_build_index
from .search import search

__all__ = ["BGE_SMALL_EN_V15", "Embedder", "VectorIndex", "load_or_build_index", "search"]
