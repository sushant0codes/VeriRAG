"""Embedding-model wrapper used by the local retriever."""

from functools import lru_cache
from typing import Sequence

import numpy as np

BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load one SentenceTransformer instance per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class Embedder:
    """Create dense vectors with a model that is loaded only once."""

    def __init__(self, model_name: str = BGE_SMALL_EN_V15) -> None:
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return int(_load_model(self.model_name).get_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one float32 embedding for every supplied text."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = _load_model(self.model_name).encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)
