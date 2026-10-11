"""Build the vector index from data/docs/*.md.

Run from backend/:  uv run python -m app.rag.ingest
"""

import re
from pathlib import Path

from ..config import DATA_DIR, get_settings
from .embedder import get_embedder
from .store import INDEX_DIR, Chunk, VectorStore

DOCS_DIR = DATA_DIR / "docs"


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """One chunk per `##` section; text before the first section is dropped
    (it holds the title and provenance line, which are stored as metadata)."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    chunks = []
    for part in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append(Chunk(source=source, title=title, section=heading.strip(), text=body))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks += chunk_markdown(path.name, path.read_text(encoding="utf-8"))
    return chunks


def build() -> VectorStore:
    embedder = get_embedder()
    chunks = load_chunks()
    # the heading carries much of the meaning, so embed it with the body
    vectors = embedder.embed([f"{c.title} – {c.section}\n{c.text}" for c in chunks])
    store = VectorStore(chunks, vectors, embedder)
    store.save(INDEX_DIR, get_settings().embedding_model)
    return store


if __name__ == "__main__":
    built = build()
    print(f"Indexed {len(built.chunks)} sections from {DOCS_DIR} into {INDEX_DIR}")
