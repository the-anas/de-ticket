"""Build the search index from data/docs/*.md.

Run from backend/:  uv run python -m app.rag.ingest
"""

import re
from pathlib import Path

from ..config import DATA_DIR, get_settings
from .embedder import Embedder, get_embedder
from .reranker import Reranker, get_reranker
from .store import INDEX_DIR, Chunk, VectorStore

DOCS_DIR = DATA_DIR / "docs"

# The embedding model reads at most 128 tokens and drops the rest silently.
# About 55 words of this text, plus the title and heading that are prepended,
# stays under that; longer sections are indexed in several pieces.
MAX_PIECE_WORDS = 55


# A section may end with a line of German search words, kept out of its text
KEYWORDS_LINE = re.compile(r"^German keywords:[ \t]*(.*)$", flags=re.MULTILINE)


def split_pieces(text: str, max_words: int = MAX_PIECE_WORDS) -> list[str]:
    """Packs whole sentences into pieces of at most max_words words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_markdown(source: str, text: str) -> list[Chunk]:
    """One or more index entries per `##` section. Text before the first section
    is dropped (it holds the title and provenance line, kept as metadata)."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    chunks = []
    for part in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        heading, _, body = part.partition("\n")
        keywords = " ".join(KEYWORDS_LINE.findall(body))
        body = KEYWORDS_LINE.sub("", body).strip()
        if not body:
            continue
        for piece in split_pieces(body):
            chunks.append(
                Chunk(source=source, title=title, section=heading.strip(), text=body, piece=piece, keywords=keywords)
            )
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks += chunk_markdown(path.name, path.read_text(encoding="utf-8"))
    return chunks


def embedding_input(chunk: Chunk) -> str:
    # the heading carries much of the meaning, so it is embedded with each piece
    return f"{chunk.title} – {chunk.section}\n{chunk.indexed_text}"


def make_store(
    embedder: Embedder, chunks: list[Chunk] | None = None, reranker: Reranker | None = None
) -> VectorStore:
    chunks = load_chunks() if chunks is None else chunks
    return VectorStore(chunks, embedder.embed([embedding_input(c) for c in chunks]), embedder, reranker)


def build() -> VectorStore:
    store = make_store(get_embedder(), reranker=get_reranker())
    store.save(INDEX_DIR, get_settings().embedding_model)
    return store


if __name__ == "__main__":
    built = build()
    sections = len({c.key for c in built.chunks})
    print(f"Indexed {sections} sections as {len(built.chunks)} entries from {DOCS_DIR} into {INDEX_DIR}")
