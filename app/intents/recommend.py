"""
get_recommendation intent handler — Part 3 (Google Books integration).

Reads `needed_sources` (set by planner_node in app/graph.py) to decide which path to take:

  - "resolve" mode: a peer match was already found. Just resolve that known 
    book_id into display metadata and explain the peer-based reasoning. No 
    RAG, no discovery search.

  - "discover" mode: no peer match. Interpret the reader's message (merged 
    with their stored preferences) into filter criteria, construct a targeted 
    Google Books query, retrieve real candidates, rank them against the 
    reader's profile, and compose a grounded recommendation from the best match.

New in Part 3:
- _construct_google_books_query() builds natural-language queries Google 
  Books understands (genre keywords, pacing descriptors, narrative patterns)
- _rank_candidates() uses the LLM to score descriptions against the reader's 
  desired profile (pacing/tone/trope intent)
- Error handling if Google Books returns no results or is unreachable
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
    a filter dict for query construction.

    Returns: dict mapping dimension names (pacing, tone, trope, mood, etc.)
    to desired values.

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
    reasoning (e.g. pacing taxonomy definitions, trope framing, tone/mood
    guidance) — NOT used to find book titles, since RAG never stores book
    titles.

    Returns: concatenated document chunks, or empty string if none found.
    """
    results = knowledge_collection.query(query_texts=[message], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents) if documents else ""


# ---------------------------------------------------------------------------
# NEW: Construct a Google Books query
# ---------------------------------------------------------------------------

def _construct_google_books_query(
    filters: dict,
    rag_context: str,
    message: str,
) -> str:
    """
    Transform extracted preference filters into a natural-language query
    that Google Books can understand and return useful candidates for.

    The agent decides WHAT KIND of book to look for, and this function
    articulates that intent as a concrete search string.

    Args:
        filters: dict of extracted preferences, e.g. 
                 {"pacing": "breakneck", "tone": "dark", "trope": "ensemble"}
        rag_context: relevant domain knowledge from RAG (pacing definitions, etc.)
        message: the reader's original message for context

    Returns:
        str: A natural-language Google Books query, e.g.
             "fast-paced dark thriller with ensemble cast"
             
    The query uses keywords Google Books understands (genre, pacing descriptors
    like "quick" or "leisurely", mood/tone words, narrative patterns) — NOT
    the invented pacing/tone/trope fields, since Google Books doesn't have those.
    """
    filters_text = json.dumps(filters) if filters else "{}"
    rag_snippet = rag_context[:500] if rag_context else "(No RAG context)"

    prompt = f"""You are building a search query for Google Books API to find books 
that match a reader's preferences.

Reader's original request: "{message}"

Extracted preference filters: {filters_text}

Relevant background on pacing/tone/trope concepts:
{rag_snippet}

Translate these preferences into a single, concise Google Books search query 
(3-8 words) that uses keywords Google Books understands:
- For pacing: use "fast-paced", "quick", "slow-burn", "leisurely", "gripping", 
  "page-turner", etc.
- For tone/mood: use "dark", "uplifting", "humorous", "suspenseful", "melancholic", etc.
- For trope/narrative: use "ensemble cast", "slow-burn romance", "unreliable 
  narrator", "found family", etc.
- Include genre keywords if relevant (mystery, thriller, fantasy, science fiction, etc.)

Examples:
- If filters say breakneck + dark + ensemble: "fast-paced dark thriller ensemble cast"
- If filters say measured + hopeful + found_family: "slow-burn hopeful story community"

Respond with ONLY the query string, no explanation or quotes.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    query = response.content[0].text.strip().strip('"')
    return query if query else "fiction"  # fallback to broad query


# ---------------------------------------------------------------------------
# NEW: Rank candidates by description fit
# ---------------------------------------------------------------------------

def _rank_candidates(
    candidates: list[dict],
    filters: dict,
    rag_context: str,
) -> list[dict]:
    """
    Score and rank a small set of candidate books (Google Books results)
    against the reader's desired profile.

    Uses LLM-based description analysis: for each candidate, evaluates how
    well the title/description/categories match the desired 
    pacing/tone/trope profile, then ranks by fit.

    Args:
        candidates: list of normalized book dicts from search_books()
        filters: extracted preference filters (pacing, tone, trope, etc.)
        rag_context: domain knowledge for grounding the evaluation

    Returns:
        list[dict]: same candidates, ranked best-to-worst by fit to profile.
                    Empty list if all candidates are ruled out.
    """
    if not candidates:
        return []

    if not filters:
        # No preference signals to rank against, return as-is
        return candidates

    filters_text = json.dumps(filters)
    rag_snippet = rag_context[:500] if rag_context else ""

    # Build the scoring prompt
    candidates_text = "\n".join(
        f"[{i+1}] {c['title']} by {c['author']}\n"
        f"    Description: {c['description']}\n"
        f"    Categories: {', '.join(c.get('categories', ['N/A']))}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are evaluating books for a reader with specific preferences.

Reader's desired profile: {filters_text}

Relevant pacing/tone/trope definitions:
{rag_snippet}

Candidates to score:
{candidates_text}

For each candidate, rate how well its description and categories match the 
reader's desired pacing, tone, and narrative patterns. Consider:
- Does the description suggest the desired pacing (e.g., "gripping" for 
  breakneck, "meditative" for measured)?
- Does the tone/mood language (dark, hopeful, etc.) match?
- Do the categories/narrative patterns align with desired tropes?

Respond ONLY with a JSON object mapping candidate numbers [1, 2, 3...] to 
scores (0-10, where 10 is perfect fit):
{{"1": 8, "2": 5, "3": 9}}

If you cannot score a candidate, assign it a 0.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        scores = json.loads(raw_text)
    except (json.JSONDecodeError, IndexError):
        # If ranking fails, return candidates in original order
        return candidates

    # Rank candidates by score
    scored_candidates = []
    for i, candidate in enumerate(candidates):
        score = scores.get(str(i + 1), 0)
        scored_candidates.append((score, candidate))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_candidates]


# ---------------------------------------------------------------------------
# Response composition
# ---------------------------------------------------------------------------

def _compose_response(reasoning_context: str, candidates: list[dict], message: str) -> str:
    """
    Compose the final natural-language recommendation, grounded in actual
    candidate book data — never inventing a book not in `candidates`.

    Updated for Part 3: doesn't reference pacing/tone/trope fields
    (those don't exist in Google Books metadata), only uses the real
    metadata Google Books provides: title, author, description, categories.
    """
    if not candidates:
        return (
            "I couldn't find a book matching what you're looking for right now. "
            "Could you tell me a bit more about what you're in the mood for?"
        )

    best_book = candidates[0]
    categories_str = ", ".join(best_book.get("categories", [])) if best_book.get("categories") else "General"

    candidates_text = (
        f"- {best_book['title']} by {best_book['author']}\n"
        f"  {best_book['description']}\n"
        f"  [{categories_str}]"
    )

    prompt = f"""The reader asked: "{message}"

{reasoning_context}

Best matching book available:
{candidates_text}

Write a short, natural recommendation (2-4 sentences) grounded ONLY in the
book data above. Do not invent details not listed. Explain briefly why
it fits what the reader is looking for based on description, categories, 
and your knowledge of their preferences.
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
    """
    Discovery path for when no peer match exists.
    
    Flow (Part 3):
    1. Extract filters from stored preferences + current message
    2. Query RAG for domain grounding (pacing/tone/trope concepts)
    3. Construct a targeted Google Books search query from filters
    4. Search Google Books with that query (returns ~10 candidates)
    5. Rank candidates by description fit to the reader's profile
    6. Compose recommendation from the best-ranked candidate
    7. Return to graph
    """
    profile = get_reader_profile(reader_id)
    filters = _extract_filters(message, profile.get("preferences", {}))
    rag_context = _query_rag(message)

    # NEW: Construct a natural-language query for Google Books
    google_books_query = _construct_google_books_query(filters, rag_context, message)

    # Search Google Books with the constructed query
    candidates = search_books(query=google_books_query, max_results=10)

    if not candidates:
        # Google Books returned nothing (network error, rate limit, or just no match)
        return {
            "handler_result": {"mode": "discover", "filters": filters, "candidates": []},
            "response": (
                "I wasn't able to find books matching your request right now. "
                "This might be a temporary issue with book search. Could you try "
                "describing what you're looking for in different words?"
            ),
        }

    # NEW: Rank candidates against the reader's profile
    ranked_candidates = _rank_candidates(candidates, filters, rag_context)

    reasoning_context = (
        f"Relevant pacing/tone concepts:\n{rag_context}"
        if rag_context else "Based on your preferences:"
    )
    response_text = _compose_response(reasoning_context, ranked_candidates, message)

    return {
        "handler_result": {
            "mode": "discover",
            "filters": filters,
            "candidates": ranked_candidates,
            "google_books_query": google_books_query,
        },
        "response": response_text,
    }