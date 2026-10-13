import numpy as np

from app.rag.ingest import MAX_PIECE_WORDS, chunk_markdown, load_chunks, split_pieces
from app.rag.store import Chunk, KeywordIndex, VectorStore, tokenize


def test_tokenize_normalises_and_drops_filler():
    assert tokenize("Kann ich nach Tønder fahren? The Gäubahn!") == ["tonder", "gaubahn"]


def test_tokenize_stems_and_keeps_train_categories_and_line_names():
    assert tokenize("Trains and connections") == ["train", "connect"]
    # two-letter train categories and tokens with a digit survive; other short words do not
    assert tokenize("Is an IC or RE ok? It costs 63") == ["ic", "re", "cost", "63"]
    assert tokenize("RE 5, RB-26 and S1") == ["re5", "rb26", "s1"]
    # the "re" of a contraction is not the train category
    assert "re" not in tokenize("You're on the wrong platform")


def test_long_sections_are_split_on_sentences_and_keep_the_full_text():
    body = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))
    chunks = chunk_markdown("x.md", f"# T\n\n## Long\n\n{body}\n")
    assert len(chunks) > 1
    assert all(len(c.piece.split()) <= MAX_PIECE_WORDS for c in chunks)
    assert all(c.text == body and c.section == "Long" for c in chunks)
    assert " ".join(c.piece for c in chunks) == body  # nothing lost, nothing repeated


def test_one_overlong_sentence_is_kept_whole():
    assert split_pieces("word " * 200) == [("word " * 200).strip()]


def test_real_docs_have_no_oversized_pieces():
    chunks = load_chunks()
    assert len({c.key for c in chunks}) == 40
    # only a single sentence longer than the limit may exceed it
    oversized = [c.piece for c in chunks if len(c.piece.split()) > MAX_PIECE_WORDS]
    assert all(len(split_pieces(p, 1)) == 1 for p in oversized)


def test_german_keywords_line_is_indexed_but_kept_out_of_the_text():
    chunks = chunk_markdown("x.md", "# T\n\n## Bikes\n\nBikes need a ticket.\n\nGerman keywords: Fahrrad, Hund\n")
    assert [(c.text, c.piece, c.keywords) for c in chunks] == [("Bikes need a ticket.", "Bikes need a ticket.", "Fahrrad, Hund")]


def test_german_question_matches_through_keywords_in_any_inflection():
    corpus = [
        Chunk("d.md", "Doc", "Bikes and dogs", "Bikes need a ticket.", "Bikes need a ticket.", "Fahrrad, Hund"),
        chunk("Prices", "It costs 63 euros."),
    ]
    index = KeywordIndex(corpus)
    for query in ("Kann ich mein Fahrrad mitnehmen?", "Fahrräder im Zug", "Darf ich meine Hunde mitnehmen?"):
        scores = index.scores(query)
        assert scores[0] > 0 and scores[1] == 0, query
    # "Fähren" (ferries) is not the filler word "fahren" (to travel)
    ferries = KeywordIndex([Chunk("d.md", "Doc", "Ferries", "Boats.", "Boats.", "Fähre"), chunk("Prices", "It costs 63 euros.")])
    assert ferries.scores("Sind Fähren enthalten?")[0] > 0 and ferries.scores("Wohin kann ich fahren?")[0] == 0
    # an English query still matches the English text
    assert index.scores("bike")[0] > 0


def chunk(section, text):
    return Chunk(source="d.md", title="Doc", section=section, text=text, piece=text)


CORPUS = [
    chunk("Austria: Salzburg and Kufstein", "Covered via Freilassing on regional trains."),
    chunk("Using the ticket abroad", "Valid to a short list of border stations outside Germany."),
    chunk("Black Forest", "Freiburg is the gateway to the hills."),
]


def test_keyword_index_finds_the_place_name():
    scores = KeywordIndex(CORPUS).scores("Kann ich mit dem Ticket nach Salzburg fahren?")
    assert scores[0] > 0 and scores[0] == max(scores)
    assert KeywordIndex(CORPUS).scores("xyzzy") == [0.0, 0.0, 0.0]


def test_coverage_is_the_share_of_distinctive_query_words_matched():
    index = KeywordIndex(CORPUS)
    full, partial, none = index.coverage("Salzburg Kufstein"), index.coverage("Salzburg Innsbruck"), index.coverage("xyzzy")
    assert full[0] == 1.0 and full[1] == full[2] == 0.0
    assert 0 < partial[0] < 1  # Innsbruck is in no entry, so half the query is unmatched
    assert none == [0.0, 0.0, 0.0]


def test_hybrid_discounts_a_match_on_common_words_only():
    # entry 0 matches the rare word of the query, entry 2 only a word found in two entries
    corpus = [
        chunk("Denmark: Tonder", "A border station."),
        chunk("Prices", "The ticket costs 63 euros."),
        chunk("Bikes", "A bike needs its own ticket."),
    ]
    store = VectorStore(corpus, np.eye(3, dtype=np.float32), FixedEmbedder())
    coverage = store._keywords.coverage("ticket Denmark")
    assert coverage[0] > coverage[2] > 0
    assert store.search("ticket Denmark")[0].chunk.section == "Denmark: Tonder"


class FixedEmbedder:
    """Says every query is closest to entry 1, then 2, then 0."""

    def embed(self, texts):
        return np.array([[0.1, 1.0, 0.5]] * len(texts), dtype=np.float32)


def test_hybrid_lets_an_exact_word_overrule_a_vague_meaning_match():
    store = VectorStore(CORPUS, np.eye(3, dtype=np.float32), FixedEmbedder())
    query = "Salzburg"
    assert store.search(query, mode="embedding")[0].chunk.section == "Using the ticket abroad"
    assert store.search(query, mode="hybrid")[0].chunk.section == "Austria: Salzburg and Kufstein"
    # without any keyword match, hybrid falls back to the meaning ranking
    assert [h.chunk.section for h in store.rank("xyzzy")] == [h.chunk.section for h in store.rank("xyzzy", "embedding")]


def test_pieces_of_one_section_collapse_to_a_single_hit():
    a1 = Chunk("d.md", "Doc", "A", "full A", "first half")
    a2 = Chunk("d.md", "Doc", "A", "full A", "second half")
    b = chunk("B", "other")
    store = VectorStore([a1, a2, b], np.eye(3, dtype=np.float32), FixedEmbedder())
    hits = store.rank("anything", "embedding")
    assert [h.chunk.section for h in hits] == ["A", "B"] and hits[0].chunk.text == "full A"


class FakeReranker:
    """Prefers texts that contain the word "special"."""

    def __init__(self):
        self.seen: list[str] = []

    def scores(self, query, texts):
        self.seen = texts
        return [1.0 if "special" in t else 0.0 for t in texts]


def test_reranker_reorders_the_best_hybrid_results():
    corpus = [chunk("A", "plain"), chunk("B", "plain"), chunk("C", "the special one")]
    reranker = FakeReranker()
    store = VectorStore(corpus, np.eye(3, dtype=np.float32), FixedEmbedder(), reranker)
    assert [h.chunk.section for h in store.rank("anything", "hybrid")] == ["B", "C", "A"]
    # reranking is the default once a reranker is set; ties keep the hybrid order
    assert [h.chunk.section for h in store.rank("anything")] == ["C", "B", "A"]
    # it reads the document title and heading with the text
    assert reranker.seen[0] == "Doc – B\nplain"


def test_reranker_only_reads_the_first_candidates(monkeypatch):
    monkeypatch.setattr("app.rag.store.RERANK_CANDIDATES", 2)
    corpus = [chunk("A", "the special one"), chunk("B", "plain"), chunk("C", "plain")]
    reranker = FakeReranker()
    store = VectorStore(corpus, np.eye(3, dtype=np.float32), FixedEmbedder(), reranker)
    # A is third in the hybrid order, outside the two candidates, so it stays last
    assert [h.chunk.section for h in store.rank("anything")] == ["B", "C", "A"]
    assert len(reranker.seen) == 2


def test_without_a_reranker_search_is_hybrid_and_rerank_mode_is_an_error():
    store = VectorStore(CORPUS, np.eye(3, dtype=np.float32), FixedEmbedder())
    assert store.default_mode == "hybrid"
    try:
        store.rank("Salzburg", "rerank")
    except ValueError as e:
        assert "RERANKER_MODEL" in str(e)
    else:
        raise AssertionError("expected an error")
