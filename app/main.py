"""Adaptive Reader Agent — FastAPI entrypoint.

Part 2: /api/chat now runs the LangGraph planner pipeline —
classify intent -> route via planner -> run intent handler -> respond —
instead of a plain LLM passthrough.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, rag_index
from .graph import build_graph

app = FastAPI(title="Adaptive Reader Agent")

# Build the graph once at import time — reused across requests.
_agent_graph = build_graph()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
_USERS = {
    u["id"]: u for u in json.loads((config.DATA_DIR / "users.json").read_text())
}

# reader_profiles.json is accessed exclusively through app/tools.py now
# (get_reader_profile / update_reader_profile) — main.py no longer reads
# it directly. Intent handlers and reader_modeling.py own that file.

# ---------------------------------------------------------------------------
# RAG: index the knowledge base once at startup (structure-aware chunking —
# this is the strategy you're shipping, not the naive baseline used only
# for the Part 1 comparison test).
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup_index_knowledge_base():
    rag_index.index_knowledge_base(chunk_fn=rag_index.structure_aware_chunk)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/users")
def users():
    """Reader / Author personas for the UI switcher."""
    return list(_USERS.values())


@app.post("/api/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="", alias="X-User-Id")):
    """
    Runs the full Part 2 pipeline via LangGraph:
      1. classify_intent — which of the 4 intents does this map to?
      2. planner — decide needed sources (Reader Model / RAG / Google Books),
         including the peer-match-first sequencing for get_recommendation
      3. intent handler (app/intents/*.py) executes
      4. reader_modeling_update — shared mechanism, writes if the handler
         produced a durable signal

    Falls back to a plain "user not found" response if x_user_id doesn't
    match a known persona — the graph still needs a user object to route
    author-only intents correctly.
    """
    user = _USERS.get(x_user_id)
    if user is None:
        return {"reply": "I don't recognize that user — please select a persona in the switcher."}

    result = _agent_graph.invoke({
        "message": req.message,
        "user": user,
    })

    return {
        "reply": result.get("response", "Something went wrong — no response was generated."),
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),  # TEMP: remove once classifier confirmed stable
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")