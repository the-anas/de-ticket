import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, get_settings
from .embedder import Embedder, get_embedder

INDEX_DIR = DATA_DIR / "index"


@dataclass
class Chunk:
    source: str  # file name
    title: str  # document title
    section: str  # section heading
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    """Brute-force cosine search; fine for a few thousand chunks."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        self.chunks = chunks
        self.vectors = vectors
        self._embedder = embedder

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        q = self._embedder.embed([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"model": model, "chunks": [c.__dict__ for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, model: str) -> "VectorStore":
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if meta["model"] != model:
            raise RuntimeError(
                f"Index was built with {meta['model']} but EMBEDDING_MODEL is {model}; "
                "rebuild it with: uv run python -m app.rag.ingest"
            )
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(directory / "vectors.npy"), embedder)


@lru_cache
def get_store() -> VectorStore:
    """Loads the index, building it on first use if it is missing."""
    model = get_settings().embedding_model
    if not (INDEX_DIR / "chunks.json").exists():
        from .ingest import build

        return build()
    return VectorStore.load(INDEX_DIR, get_embedder(), model)
