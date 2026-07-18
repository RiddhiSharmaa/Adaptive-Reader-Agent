"""Adaptive Reader Agent — FastAPI entrypoint.

Currently: plain conversation with the LLM, persona switching (Reader vs
Author) works, but no domain logic yet — no recommendations, no reader
modeling, no RAG. That's Part 1/2.

TODO (Part 2): swap the chat() body below for the LangGraph planner
pipeline: classify intent -> route via planner -> run intent handler ->
respond.
"""

import json
from pathlib import Path

import anthropic
from fastapi import FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config

app = FastAPI(title="Adaptive Reader Agent")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
_USERS = {
    u["id"]: u for u in json.loads((config.DATA_DIR / "users.json").read_text())
}

# reader_profiles.json is your one local data file — dict keyed by reader_id.
# TODO: once tools.py is built, main.py shouldn't touch this file directly;
# route through get_reader_profile() / update_reader_profile() instead.
_PROFILES_PATH = config.DATA_DIR / "reader_profiles.json"
_PROFILES = json.loads(_PROFILES_PATH.read_text()) if _PROFILES_PATH.exists() else {}


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def _client():
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Copy .env.example to .env "
            "and fill in your values."
        )
    kwargs = {"api_key": config.ANTHROPIC_API_KEY}
    if config.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = config.ANTHROPIC_BASE_URL
    return anthropic.Anthropic(**kwargs)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/users")
def users():
    """Reader / Author personas for the UI switcher."""
    return list(_USERS.values())


@app.post("/api/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="", alias="X-User-Id")):
    """
    Plain conversation with the LLM — no planner/intents wired in yet.

    TODO (Part 2): replace this body with:
      1. Intent classification (which of your 4 intents does this map to?)
      2. Planner routing (Reader Model / RAG / Google Books, per source
         confidence)
      3. Intent handler execution (app/intents/*.py)
      4. Reader Modeling update if the turn produced a durable signal
    """
    user = _USERS.get(x_user_id)

    who = (
        f"You are talking to {user['full_name']}, acting as a {user['role']}."
        if user
        else "You are talking to a user."
    )

    system = (
        "You are the Adaptive Reader Agent, a reading companion that learns "
        "a reader's preferences over time instead of relying on genre tags. "
        + who
        + " Right now you can only make small talk. You cannot yet give "
        "recommendations, log reading outcomes, share insights, or show "
        "author stats. If asked for any of those, say plainly that you "
        "can't do that yet."
    )

    resp = _client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=500,
        temperature=0.5,
        system=system,
        messages=[{"role": "user", "content": req.message}],
    )
    return {"reply": resp.content[0].text}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")