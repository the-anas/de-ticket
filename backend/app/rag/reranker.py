from functools import lru_cache
from typing import Protocol

from ..config import get_settings


class Reranker(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, higher is better."""
        ...


class FastEmbedReranker:
    """Local ONNX cross-encoder: reads the query and a text together and scores
    how well the text answers the query. Downloads the model on first use."""

    def __init__(self, model_name: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


@lru_cache
def get_reranker() -> Reranker | None:
    """The configured reranker, or None when RERANKER_MODEL is empty."""
    name = get_settings().reranker_model
    return FastEmbedReranker(name) if name else None
