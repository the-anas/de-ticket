# D-Ticket Trip Planner

Plan trips that stay inside the Deutschlandticket: regional trains, S-Bahn,
U-Bahn, trams and buses in Germany, plus the few border stations abroad where
the ticket is valid. No ICE, IC or other long-distance services.

- **Map** of Germany with railways highlighted and curated destinations.
- **Start city** box: pick a station and see popular destinations with live
  regional-only journey times; select one to draw the route.
- **Assistant**: a prompt box backed by an LLM that plans trips with the live
  timetable and answers questions from the bundled documents in
  `backend/data/docs/`.

## Run it

Requires Python 3.12+ with [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
cp .env.example .env          # then set LLM_API_KEY (see below)

cd backend
uv sync
uv run python -m app.rag.ingest      # build the document index (downloads a ~220 MB embedding model once)
uv run uvicorn app.main:app --port 8000 --reload --reload-dir app

cd ../frontend
npm install
npm run dev                   # http://localhost:5173
```

The map and route search work without an LLM key; only the assistant needs one.

## Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `anthropic` (default) or `openai` for any OpenAI-compatible endpoint |
| `LLM_MODEL` | Model id, e.g. `claude-opus-5-5`, or `llama3.1` on Ollama |
| `LLM_API_KEY` | API key. For Anthropic it may be left empty if `ANTHROPIC_API_KEY` is set |
| `LLM_BASE_URL` | Only for OpenAI-compatible servers, e.g. `http://localhost:11434/v1` |
| `TRANSIT_BASE_URL` | MOTIS API base URL, default `https://api.transitous.org` |
| `TRANSIT_USER_AGENT` | Sent to the transit API; put a contact address in it |
| `EMBEDDING_MODEL` | fastembed model for document search; rebuild the index after changing it |

## How it works

```
frontend/   React + Vite + MapLibre GL
backend/
  app/transit/   Transitous (MOTIS) client and the Deutschlandticket validity check
  app/routes/    /api/stations, /api/journeys, /api/popular, /api/destinations, /api/chat (SSE)
  app/llm/       provider interface, Anthropic and OpenAI-compatible adapters, tool loop
  app/rag/       Markdown chunking, local embeddings, cosine search
  data/          destinations, border stations, Germany outline, docs
```

**Validity check.** Journeys are requested with regional modes only, then each
leg is checked again by line name and operator (ICE, IC, EC, RJ, FLX, Westbahn
and so on), and stops outside Germany are rejected unless they are listed in
`data/border_stations.json`. This is a heuristic over timetable data, not a
fare engine: treat the result as a strong hint and confirm unusual cases,
especially near borders and for ferries. The few IC sections that do accept the
ticket are never offered.

**Transit data.** [Transitous](https://transitous.org) is a free community
service and rate-limits clients to roughly one request every three seconds
after a small burst. The backend caches responses for 30 minutes and retries on
429, and the popular-routes list fills in one destination at a time for that
reason. For anything beyond personal use, run your own
[MOTIS](https://github.com/motis-project/motis) instance and point
`TRANSIT_BASE_URL` at it.

**Documents.** The files in `backend/data/docs/` are starter content written in
October 2026. Edit or add Markdown files (one `##` section per topic) and rerun
`uv run python -m app.rag.ingest`.

## Maintenance

```bash
cd backend
uv run pytest                                   # unit tests
uv run python scripts/build_destinations.py     # re-resolve the curated destinations
cd ../frontend && npm run build && npm run lint
```

## Data sources

Timetables and geocoding: Transitous / MOTIS. Base map: OpenFreeMap,
OpenMapTiles, © OpenStreetMap contributors. Germany outline: Natural Earth.
