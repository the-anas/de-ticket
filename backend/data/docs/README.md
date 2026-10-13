These Markdown files are the knowledge base the assistant searches.

They are starter content written on 2026-10-11 from public sources and general
travel knowledge. Fares and validity rules change; review and extend them
before relying on the answers. Each `##` section becomes one searchable chunk,
so keep sections self-contained. Open each section with a sentence that names
its subject in full ("The Deutschlandticket is valid to one station in Denmark:
Tønder"), since search reads the section without the rest of the document. A section can end with a line
`German keywords: Wort, Wort, ...`: German words people use for its subject.
They are matched by keyword search only and are not part of the text the
assistant reads. After editing, rebuild the index:

    uv run python -m app.rag.ingest

This README is not indexed.
