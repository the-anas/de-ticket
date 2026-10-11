from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import chat, journeys, popular, stations

app = FastAPI(title="D-Ticket Trip Planner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(stations.router)
app.include_router(journeys.router)
app.include_router(popular.router)
app.include_router(chat.router)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}
