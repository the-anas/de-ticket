import json
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..deps import get_transit
from ..llm import get_llm
from ..llm.agent import run_agent
from ..llm.tools import ToolRunner

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Origin(BaseModel):
    name: str
    lat: float
    lon: float


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    origin: Origin | None = None


@router.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def events():
        try:
            llm = get_llm()
        except Exception as e:  # bad provider config
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return
        history = [m.model_dump() for m in req.messages]
        origin = req.origin.model_dump() if req.origin else None
        async for event in run_agent(llm, ToolRunner(get_transit()), history, origin):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
