from functools import lru_cache
from typing import Protocol

import numpy as np

from ..config import get_settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns one L2-normalised row per text."""
        ...


class FastEmbedEmbedder:
    """Local ONNX embedding model; no API key, downloads the model on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


@lru_cache
def get_embedder() -> Embedder:
    return FastEmbedEmbedder(get_settings().embedding_model)
