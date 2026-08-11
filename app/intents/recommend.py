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
  desired profile (pacing/tone/trope intent) AND checks page_count
- Error handling if Google Books returns no results or is unreachable
"""

import json
import os
from anthropic import Anthropic
from typing import Optional

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
    to desired values, PLUS any metadata constraints like "length" or "page_count".

    Approach: stored preferences are the baseline (what we already believe
    about this reader). The current message can override or add to that —
    e.g. a reader with a stored "measured" pacing preference who explicitly
    asks for "something breakneck today" should get breakneck results, not
    have their one-off request overridden by stale history.
    
    IMPORTANT: explicitly extract metadata constraints (length, page count,
    series vs standalone, etc.) from the current message. These are NOT in
    stored preferences and must be inferred from words like "short", "long",
    "quick", "quick read", "series", "standalone", etc.
    """
    baseline = {
        key: field["value"]
        for key, field in stored_preferences.items()
        if isinstance(field, dict) and "value" in field
    }

    extraction_prompt = f"""The reader's stored preferences are: {json.dumps(baseline)}

Their current message is: "{message}"

Extract TWO things:

1. Reading preferences (pacing, tone, mood, trope, genre):
   If the message expresses a specific pacing, tone, mood, or trope preference,
   use that value (it overrides the stored baseline for this request). If the
   message doesn't mention a given dimension, keep the stored baseline value
   for it. Omit any dimension with no value from either source.

2. Metadata constraints:
   - "length": if the message says "short", "quick read", "novella" → "short"
     if the message says "long", "epic", "dense", "hefty" → "long"
   - "series": if the message asks for a series → "series"
     if the message asks for standalone → "standalone"
   Omit if not mentioned.

Respond ONLY with a JSON object, e.g.:
{{"pacing": "breakneck", "tone": "dark", "length": "short", "series": "series"}}

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
# Construct a Google Books query
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
# Rank candidates by description fit AND metadata constraints
# ---------------------------------------------------------------------------

# FIX 1: REVERT _rank_candidates in app/intents/recommend.py
# Replace the ENTIRE _rank_candidates function with this (simpler, was working):

def _rank_candidates(
    candidates: list[dict],
    filters: dict,
    rag_context: str,
) -> list[dict]:
    """Score and rank candidates. Soft penalties for constraints."""
    if not candidates or not filters:
        return candidates

    filters_text = json.dumps(filters)
    rag_snippet = rag_context[:300] if rag_context else ""

    candidates_text = "\n".join(
        f"[{i+1}] {c['title']} by {c['author']} ({c.get('page_count', 'unknown')} pages)\n"
        f"    {c['description'][:150]}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""Score books {1}-{len(candidates)} for reader profile {filters_text}.
Respond ONLY: {{"1": 9, "2": 5}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").lstrip("json").strip()
        scores = json.loads(raw_text)
    except (json.JSONDecodeError, IndexError):
        return candidates

    # SOFT penalties only (not hard filters)
    length_constraint = filters.get("length")
    scored = []
    
    for i, candidate in enumerate(candidates):
        base_score = float(scores.get(str(i + 1), 0))
        page_count = candidate.get("page_count", 0)
        
        # Soft penalty (0.8x multiplier, not removal)
        if length_constraint == "short" and page_count and page_count > 300:
            base_score *= 0.8
        elif length_constraint == "long" and page_count and page_count < 200:
            base_score *= 0.8
        
        scored.append((base_score, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ---------------------------------------------------------------------------
# Response composition
# ---------------------------------------------------------------------------

# In app/intents/recommend.py, replace _compose_response guard section:

# def _compose_response(
#     reasoning_context: str, 
#     candidates: list[dict], 
#     message: str,
#     reader_profile: Optional[dict] = None,
# ) -> str:
#     """Recommend from candidates grounded in metadata."""
#     if not candidates:
#         return (
#             "I couldn't find a book matching what you're looking for right now. "
#             "Try describing what you're in the mood for in different words."
#         )

#     best_book = candidates[0]
    
#     # Softer guard: Only refuse if description is COMPLETELY empty AND profile is null
#     description = best_book.get("description", "").strip()
    
#     # If description empty AND no reader profile, ask for help
#     profile_empty = not reader_profile or not reader_profile.get("preferences")
    
#     if not description and profile_empty:
#         return (
#             "I found a book but don't have enough details about it to "
#             "explain why it fits. Try describing what you're looking for?"
#         )
    
#     # If description exists (even short), proceed with recommendation
#     # Don't refuse just because it's thin
    
#     categories_str = ", ".join(best_book.get("categories", [])) if best_book.get("categories") else ""

#     candidates_text = (
#         f"- {best_book['title']} by {best_book['author']}\n"
#         f"  {description if description else '(Limited details available)'}\n"
#         f"  {f'[{categories_str}]' if categories_str else ''}"
#     )

#     prompt = f"""Reader: "{message}"

# {reasoning_context}

# Book to recommend:
# {candidates_text}

# Write 2-3 sentence recommendation grounded ONLY in the data above.
# Do not invent details. Just explain why it might fit their request."""

#     response = client.messages.create(
#         model=MODEL,
#         max_tokens=200,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.content[0].text.strip()
# FIX: Add metadata validation to _compose_response() in app/intents/recommend.py

def _compose_response(
    reasoning_context: str, 
    candidates: list[dict], 
    message: str,
    reader_profile: Optional[dict] = None,
) -> str:
    """Recommend from candidates, validating metadata matches constraints."""
    
    if not candidates:
        return "I couldn't find a book matching what you're looking for."

    # VALIDATION 1: Null profile
    profile_empty = not reader_profile or not reader_profile.get("preferences")
    message_lower = message.lower()
    
    if profile_empty and any(kw in message_lower for kw in ["perfect", "every preference", "match me"]):
        return (
            "I'd need to know your reading preferences to make a confident recommendation. "
            "What kinds of pacing, tone, or story types appeal to you?"
        )
    
    # VALIDATION 2: Find best candidate that respects explicit constraints
    best_book = None
    
    # Extract explicit constraints from message
    wants_slow = any(w in message_lower for w in ["slow", "measured", "leisurely", "atmospheric"])
    wants_short = any(w in message_lower for w in ["short", "quick read", "brief"])
    
    for candidate in candidates:
        # Skip if violates pacing constraint
        if wants_slow:
            pacing = candidate.get("pacing", "").lower()
            if pacing in ["breakneck", "brisk", "fast-paced"]:
                continue  # Skip this one, find slower alternative
        
        if wants_short:
            pages = candidate.get("page_count", 0)
            if pages > 300:
                continue  # Skip this one, find shorter alternative
        
        best_book = candidate
        break
    
    # If all candidates violated constraints, fall back to first but note mismatch
    if not best_book:
        best_book = candidates[0]
        
        # Check if recommending despite mismatch
        description = best_book.get("description", "").strip()
        pacing = best_book.get("pacing", "").lower()
        pages = best_book.get("page_count", 0)
        
        mismatch = False
        if wants_slow and pacing in ["breakneck", "brisk"]:
            if not description:
                return "The available books don't match what you're looking for right now. Try describing differently?"
            mismatch = True
        
        if wants_short and pages > 300:
            if not description:
                return "The available books don't match what you're looking for right now. Try describing differently?"
            mismatch = True
        
        # If mismatch but has description, note it in prompt
        if mismatch:
            caveat = f"\n[Note: This book has {pacing} pacing and {pages} pages, which doesn't perfectly match the 'slow' or 'short' request, but it may still interest you based on other factors.]"
        else:
            caveat = ""
    else:
        caveat = ""
    
    # COMPOSE RECOMMENDATION
    description = best_book.get("description", "").strip()
    if not description:
        return "The available books don't have enough details for a confident recommendation. Try describing differently?"
    
    categories_str = ", ".join(best_book.get("categories", [])) if best_book.get("categories") else ""
    
    candidates_text = (
        f"- {best_book['title']} by {best_book['author']}\n"
        f"  {description}\n"
        f"  {f'[{categories_str}]' if categories_str else ''}"
    )

    prompt = f"""Reader: "{message}"

{reasoning_context}

Book:
{candidates_text}{caveat}

Recommend in 2-3 sentences. Ground ONLY in metadata above. No invented details."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
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
            return _discover_path(reader_id, message)

        reasoning_context = (
            "This book was recommended because a reader with a similar "
            "preference profile finished it and rated it highly."
        )
        reader_profile = get_reader_profile(reader_id)  # <-- ADD THIS LINE
        response_text = _compose_response(reasoning_context, [book], message, reader_profile)  # <-- ADD reader_profile PARAM
        return {
            "handler_result": {"mode": "resolve", "book": book},
            "response": response_text,
        }

    return _discover_path(reader_id, message)


def _discover_path(reader_id: str, message: str) -> dict:
    """
    Discovery path for when no peer match exists.
    
    Flow (Part 3):
    1. Extract filters from stored preferences + current message (including metadata constraints)
    2. Query RAG for domain grounding (pacing/tone/trope concepts)
    3. Construct a targeted Google Books search query from filters
    4. Search Google Books with that query (returns ~10 candidates)
    5. Rank candidates by description fit to the reader's profile + metadata constraints
    6. Compose recommendation from the best-ranked candidate
    7. Return to graph
    """
    profile = get_reader_profile(reader_id)
    filters = _extract_filters(message, profile.get("preferences", {}))
    rag_context = _query_rag(message)

    # Construct a natural-language query for Google Books
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

    # Rank candidates against the reader's profile AND metadata constraints
    ranked_candidates = _rank_candidates(candidates, filters, rag_context)

    reasoning_context = (
        f"Relevant pacing/tone concepts:\n{rag_context}"
        if rag_context else "Based on your preferences:"
    )
    response_text = _compose_response(reasoning_context, ranked_candidates, message, profile)  # <-- ADD profile PARAM

    return {
        "handler_result": {
            "mode": "discover",
            "filters": filters,
            "candidates": ranked_candidates,
            "google_books_query": google_books_query,
        },
        "response": response_text,
    }