"""
get_recommendation intent handler.

Reads `needed_sources` (set by planner_node in app/graph.py) to decide
which path to take:

  - "resolve" mode: a peer match was already found. Just resolve that
    known book_id into display metadata and explain the peer-based
    reasoning. No RAG, no discovery search.

  - "discover" mode: no peer match. Interpret the reader's message
    (merged with their stored preferences) into search filters, query RAG
    for domain grounding (pacing/tone/trope guidance relevant to the
    request), run a discovery search, and compose a grounded recommendation.
"""

import json
import os
from anthropic import Anthropic

from app.tools import get_book_by_id, search_books, get_reader_profile
from app.rag_index import knowledge_collection

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Filter extraction (discover mode only)
# ---------------------------------------------------------------------------

def _extract_filters(message: str, stored_preferences: dict) -> dict:
    """
    Merge the reader's current message with their stored preferences into
    a filter dict for search_books().

    Approach: stored preferences are the baseline (what we already believe
    about this reader). The current message can override or add to that —
    e.g. a reader with a stored "measured" pacing preference who explicitly
    asks for "something breakneck today" should get breakneck results, not
    have their one-off request overridden by stale history.
    """
    baseline = {
        key: field["value"]
        for key, field in stored_preferences.items()
        if isinstance(field, dict) and "value" in field
    }

    extraction_prompt = f"""The reader's stored preferences are: {json.dumps(baseline)}

Their current message is: "{message}"

If the message expresses a specific pacing, tone, mood, or trope preference,
use that value (it overrides the stored baseline for this request). If the
message doesn't mention a given dimension, keep the stored baseline value
for it. Omit any dimension with no value from either source.

Respond ONLY with a JSON object mapping dimension names to values, e.g.:
{{"pacing": "breakneck", "tone": "dark"}}
If nothing can be determined, respond with {{}}.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": extraction_prompt}],
    )

    try:
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        return json.loads(raw_text)
    except (json.JSONDecodeError, IndexError):
        return baseline  # fail safe: fall back to stored preferences only


# ---------------------------------------------------------------------------
# RAG grounding (discover mode only)
# ---------------------------------------------------------------------------

def _query_rag(message: str, n_results: int = 3) -> str:
    """
    Retrieve relevant knowledge-base chunks to ground the recommendation's
    reasoning (e.g. pacing taxonomy definitions, trope framing) — NOT used
    to find book titles, since RAG never stores book titles.
    """
    results = knowledge_collection.query(query_texts=[message], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents) if documents else ""


# ---------------------------------------------------------------------------
# Response composition
# ---------------------------------------------------------------------------

def _compose_response(reasoning_context: str, candidates: list[dict], message: str) -> str:
    """
    Compose the final natural-language recommendation, grounded in the
    actual candidate book data — never inventing a book not in `candidates`.
    """
    if not candidates:
        return (
            "I couldn't find a book matching what you're looking for right now. "
            "Could you tell me a bit more about what you're in the mood for?"
        )

    candidates_text = "\n".join(
        f"- {b['title']} by {b['author']}: {b['description']} "
        f"(pacing: {b.get('pacing')}, tone: {b.get('tone')}, trope: {b.get('trope')})"
        for b in candidates
    )

    prompt = f"""The reader asked: "{message}"

{reasoning_context}

Candidate book(s) available to recommend:
{candidates_text}

Write a short, natural recommendation (2-4 sentences) grounded ONLY in the
candidate data above. Do not invent details not listed. Explain briefly why
it fits what the reader is looking for.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handle_get_recommendation(state: dict) -> dict:
    """
    state must contain: message, user (with user_id), needed_sources
    (set by planner_node).

    Returns a dict to merge into graph state: {"handler_result": ..., "response": ...}
    """
    reader_id = state["user"]["id"]
    message = state["message"]
    sources = state["needed_sources"]

    if sources.get("google_books_mode") == "resolve" and sources.get("peer_match_book_id"):
        # --- Peer match path: no RAG, no discovery search ---
        book_id = sources["peer_match_book_id"]
        book = get_book_by_id(book_id)

        if book is None:
            # Defensive fallback — peer match pointed at a book_id that
            # couldn't be resolved. Don't fail silently, fall through to
            # discover mode instead.
            return _discover_path(reader_id, message)

        reasoning_context = (
            "This book was recommended because a reader with a similar "
            "preference profile finished it and rated it highly."
        )
        response_text = _compose_response(reasoning_context, [book], message)
        return {
            "handler_result": {"mode": "resolve", "book": book},
            "response": response_text,
        }

    return _discover_path(reader_id, message)


def _discover_path(reader_id: str, message: str) -> dict:
    profile = get_reader_profile(reader_id)
    filters = _extract_filters(message, profile.get("preferences", {}))
    rag_context = _query_rag(message)
    candidates = search_books(query=message, filters=filters if filters else None)

    reasoning_context = (
        f"Relevant background on pacing/tone/trope terminology:\n{rag_context}"
        if rag_context else ""
    )
    response_text = _compose_response(reasoning_context, candidates[:1], message)
    return {
        "handler_result": {"mode": "discover", "filters": filters, "candidates": candidates},
        "response": response_text,
    }