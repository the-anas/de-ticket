import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from py_rust_stemmers import SnowballStemmer

from ..config import DATA_DIR, get_settings
from .embedder import Embedder, get_embedder
from .reranker import Reranker, get_reranker

INDEX_DIR = DATA_DIR / "index"
# bump when the chunk format changes, so an old index is rebuilt instead of misread
INDEX_VERSION = 3

# Reciprocal rank fusion constant; 60 is the conventional value. On the eval
# set, values from 5 to 60 differ by one or two queries, so it is not tuned.
RRF_K = 60
# In the fusion, an entry's keyword rank counts in proportion to how much of
# the query's distinctive wording it matched (KeywordIndex.coverage). Plain
# rank fusion gives a match on "ticket" alone the same credit as a match on a
# place name, which pushed good meaning matches down.

# How many of the best hybrid results the reranker reads. It runs the model
# once per candidate, so this is the cost of a search; sections ranked lower
# than this by the hybrid search cannot be recovered by it.
RERANK_CANDIDATES = 10

STOPWORDS = frozenset(
    "the and for with from that this are was can you your not have has but its into about when what where "
    "which how does will there their them they also only any all one get use via per who why "
    "der die das und mit von für ist nicht ein eine einen einem dem den des auf auch nach bei aus wie was wo "
    "ich kann kannst man mein meine mich mir wir sie ihr sich oder zum zur ins vom noch nur "
    "darf sind wenn dass einer eines meinen meinem dieser diese dieses gibt habe haben wird werden muss "
    "können über unter bis durch ohne sehr mehr mal bitte "
    # as general in a travel question as "the ticket" and "train" are
    "fahren fahre fährt fahrt komme kommen kommt gilt gelten mache machen tun zug züge zügen".split()
)


# Train categories are the only two-letter words worth matching; other short
# words are noise. Tokens with a digit (S1, 63) are kept whatever their length.
SHORT_WORDS = frozenset({"ic", "ec", "re", "rb"})
# "RE 5" and "RE5" are the same line
LINE_NAME = re.compile(r"\b(ire|ice|mex|re|rb|ic|ec|s|u)[ -]?(\d{1,3})\b")
# "you're" must not leave an "re" behind that matches the train category
CONTRACTION = re.compile(r"['’](?:re|s|ll|ve|d|m|t)\b")

_stem_english = SnowballStemmer("english").stem_word
_stem_german = SnowballStemmer("german").stem_word


def _plain(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.replace("ø", "o").replace("ß", "ss").replace("ł", "l"))
    return re.sub(r"[^a-z0-9]", "", word)


def words(text: str) -> list[str]:
    """Lower-case words without accents, filler removed, line names joined
    (Tønder -> tonder, RE 5 -> re5)."""
    text = unicodedata.normalize("NFC", text.lower())
    text = LINE_NAME.sub(r"\1\2", CONTRACTION.sub("", text))
    kept = []
    for raw in re.findall(r"[^\W_]+", text):
        # filler is recognised before accents go: "Fähren" (ferries) must not
        # be dropped as "fahren" (to travel)
        if raw in STOPWORDS:
            continue
        t = _plain(raw)
        has_digit = any(c.isdigit() for c in t)
        if len(t) < 3 and not (t in SHORT_WORDS or (has_digit and len(t) == 2)):
            continue
        kept.append(t)
    return kept


def stem(word: str, german: bool = False) -> str:
    if any(c.isdigit() for c in word):
        return word
    return _stem_german(word) if german else _stem_english(word)


def tokenize(text: str, german: bool = False) -> list[str]:
    """Word stems for keyword matching (trains -> train; with german=True,
    Fahrräder -> fahrrad). The documents are English and stemmed as English,
    which leaves most names alone; their German keywords are stemmed as German."""
    return [stem(w, german) for w in words(text)]


@dataclass
class Chunk:
    source: str  # file name
    title: str  # document title
    section: str  # section heading
    text: str  # the whole section: what the assistant is given to read
    # The part of the section this entry is indexed by. Long sections are split
    # into several entries, because the embedding model only reads the first
    # ~128 tokens of its input; all of them return the same full `text`.
    piece: str = ""
    # German words people use for this section's subject ("Niederlande",
    # "Fahrrad"). Matched by keyword search only; not embedded, not shown.
    keywords: str = ""

    @property
    def indexed_text(self) -> str:
        return self.piece or self.text

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.section)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class KeywordIndex:
    """BM25 over the indexed pieces; the section heading counts double. An
    entry is also indexed by its section's German keywords, and each query word
    is looked up as English and as German, so a German question can match."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        text = [Counter(tokenize(f"{c.section} {c.section} {c.title} {c.indexed_text}")) for c in chunks]
        self._docs = [t + Counter(tokenize(c.keywords, german=True)) for t, c in zip(text, chunks)]
        # length is that of the text alone, so a long keyword list is not held against an entry
        self._lengths = [sum(t.values()) for t in text]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        # compute idf weights depending on frequency of words
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # weight of a query word found in no entry: as rare as a word can be
        self._unseen_idf = math.log(1 + (n + 0.5) / 0.5)

    def _terms(self, query: str) -> tuple[set[str], int]:
        """The query's words as indexed terms, and how many are in no entry."""
        known: set[str] = set()
        unseen = 0
        for word in set(words(query)):
            # as an English word first, since most of the index is English
            term = next((t for t in (stem(word), stem(word, german=True)) if t in self._idf), None)
            if term:
                known.add(term)
            else:
                unseen += 1
        return known, unseen

    def scores(self, query: str) -> list[float]:
        terms, _ = self._terms(query)
        out = []
        for doc, length in zip(self._docs, self._lengths):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self._idf[t] * tf * (self._k1 + 1) / (tf + self._k1 * (1 - self._b + self._b * length / self._avg))
            out.append(s)
        return out

    def coverage(self, query: str) -> list[float]:
        """Per entry, the share (0 to 1) of the query's words it contains, each
        word weighted by its rarity. An entry matching only common words, or a
        query whose rare words appear nowhere, gets a low share."""
        terms, unseen = self._terms(query)
        total = sum(self._idf[t] for t in terms) + unseen * self._unseen_idf
        if not total:
            return [0.0] * len(self._docs)
        return [sum(self._idf[t] for t in terms if t in doc) / total for doc in self._docs]


class VectorStore:
    """Hybrid search over a small corpus: embeddings for meaning, BM25 for exact
    words such as place names. Brute force; fine for a few thousand entries.
    With a reranker, the best hybrid results are re-ordered by a model that
    reads the query and each section together."""

    def __init__(
        self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, reranker: Reranker | None = None
    ):
        self.chunks = chunks
        self.vectors = vectors
        self._embedder = embedder
        self._reranker = reranker
        self._keywords = KeywordIndex(chunks)

    @property
    def default_mode(self) -> str:
        return "rerank" if self._reranker else "hybrid"

    def rank(self, query: str, mode: str | None = None) -> list[Hit]:
        """All sections, best first. mode: "rerank", "hybrid", "embedding" or
        "keyword"; by default "rerank" when a reranker is configured, else "hybrid"."""
        mode = mode or self.default_mode
        if not self.chunks:
            return []
        if mode == "rerank":
            if not self._reranker:
                raise ValueError("No reranker is configured (RERANKER_MODEL is empty)")
            return self._rerank(query, self.rank(query, "hybrid"))
        cosine = self.vectors @ self._embedder.embed([query])[0]
        by_meaning = [int(i) for i in np.argsort(-cosine)]
        keyword = self._keywords.scores(query)
        by_keyword = [i for i in sorted(range(len(keyword)), key=lambda i: -keyword[i]) if keyword[i] > 0]

        if mode == "embedding":
            scored = [(float(cosine[i]), i) for i in by_meaning]
        elif mode == "keyword":
            scored = [(keyword[i], i) for i in by_keyword]
        else:
            # reciprocal rank fusion: an entry scores for its position in each
            # list, the keyword position scaled by how much of the query it matched
            coverage = self._keywords.coverage(query)
            fused = {i: 1 / (RRF_K + rank) for rank, i in enumerate(by_meaning, 1)}
            for rank, i in enumerate(by_keyword, 1):
                fused[i] += coverage[i] / (RRF_K + rank)
            scored = sorted(((s, i) for i, s in fused.items()), reverse=True)

        # several entries can belong to one section; keep its best
        hits: list[Hit] = []
        seen: set[tuple[str, str]] = set()
        for score, i in scored:
            chunk = self.chunks[i]
            if chunk.key not in seen:
                seen.add(chunk.key)
                hits.append(Hit(chunk, float(score)))
        return hits

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Re-orders the first RERANK_CANDIDATES hits by the reranker's score;
        the rest follow in their original order."""
        head, tail = hits[:RERANK_CANDIDATES], hits[RERANK_CANDIDATES:]
        # with the document title and heading, as for the embedding: many sections
        # only say "covered" and rely on them to say by what
        texts = [f"{h.chunk.title} – {h.chunk.section}\n{h.chunk.text}" for h in head]
        scores = self._reranker.scores(query, texts)
        # sorted() is stable, so equal scores keep the hybrid order
        reranked = sorted(zip(scores, head), key=lambda pair: -pair[0])
        return [Hit(h.chunk, float(score)) for score, h in reranked] + tail

    def search(self, query: str, k: int = 5, mode: str | None = None) -> list[Hit]:
        return self.rank(query, mode)[:k]

    def save(self, directory: Path, model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"version": INDEX_VERSION, "model": model, "chunks": [c.__dict__ for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, reranker: Reranker | None = None) -> "VectorStore":
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(directory / "vectors.npy"), embedder, reranker)


def index_is_current(directory: Path, model: str) -> bool:
    try:
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("version") == INDEX_VERSION and meta.get("model") == model


@lru_cache
def get_store() -> VectorStore:
    """Loads the index, rebuilding it if it is missing or was made with another
    embedding model or an older format."""
    if not index_is_current(INDEX_DIR, get_settings().embedding_model):
        from .ingest import build

        return build()
    return VectorStore.load(INDEX_DIR, get_embedder(), get_reranker())
