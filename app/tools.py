"""
Tool layer for the Adaptive Reader Agent — Part 3 (Google Books API).

Replaces search_books(), get_book_by_id() stubs with real Google Books API calls.
Everything else (reader profile management, log_book) stays the same.

Normalized return shape ensures nothing upstream (planner, recommend handler)
needs to change — Google Books metadata is mapped to a consistent dict schema.
"""

import json
import os
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
READER_PROFILES_PATH = DATA_DIR / "reader_profiles.json"

# Google Books API configuration
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
GOOGLE_BOOKS_API_BASE = "https://www.googleapis.com/books/v1"
GOOGLE_BOOKS_API_TIMEOUT = 10  # seconds

_GOOGLE_BOOKS_DEGRADED = False
_SEARCH_CACHE = {}
_BOOK_BY_ID_CACHE = {}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_profiles() -> dict:
    if not READER_PROFILES_PATH.exists():
        return {}
    with open(READER_PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_profiles(profiles: dict) -> None:
    with open(READER_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_google_book(volume: dict) -> Optional[dict]:
    """
    Transform a Google Books volume object into our normalized shape.
    Returns None if the volume is missing critical fields.

    Normalized shape (kept consistent with Part 2 to avoid breaking upstream):
    {
        "book_id": str,          (Google volumeId)
        "title": str,
        "author": str,           (primary author, comma-separated if multiple)
        "description": str,      (truncated if too long)
        "categories": list,      (genre/category tags)
        "pageCount": int,
        "imageLink": str,        (thumbnail URL or empty string)
        "publishedDate": str,
        "language": str,
    }

    Note: pacing, tone, trope fields are deliberately NOT included — these
    are invented fields specific to the reader model. They will be evaluated
    later by comparing descriptions against the reader's preferences.
    """
    vol_info = volume.get("volumeInfo", {})
    
    # Critical fields that must exist
    title = vol_info.get("title")
    if not title:
        return None

    # Authors
    authors_list = vol_info.get("authors", ["Unknown"])
    author = ", ".join(authors_list) if authors_list else "Unknown"

    # Description (may be missing or very long)
    description = vol_info.get("description", "").strip()
    if len(description) > 500:
        description = description[:497] + "..."

    return {
        "book_id": volume.get("id"),
        "title": title,
        "author": author,
        "description": description if description else "(No description available)",
        "categories": vol_info.get("categories", []),
        "pageCount": vol_info.get("pageCount", 0),
        "imageLink": vol_info.get("imageLinks", {}).get("thumbnail", ""),
        "publishedDate": vol_info.get("publishedDate", ""),
        "language": vol_info.get("language", "en"),
    }

# ---------------------------------------------------------------------------
# Local book database fallback
# ---------------------------------------------------------------------------

_LOCAL_BOOKS_PATH = DATA_DIR / "books.json"

def _load_local_books() -> list[dict]:
    """Load local book database for fallback when Google Books API is unavailable."""
    if not _LOCAL_BOOKS_PATH.exists():
        return []
    with open(_LOCAL_BOOKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _search_local_books(query: str, max_results: int = 10) -> list[dict]:
    """Search local database, return best matches or fallback to all books."""
    books = _load_local_books()
    if not books:
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored = []
    for book in books:
        title_match = sum(1 for w in query_words if w in book.get("title", "").lower())
        desc_match = sum(1 for w in query_words if w in book.get("description", "").lower())
        category_match = sum(1 for w in query_words if w in " ".join(book.get("categories", [])).lower())
        
        score = title_match * 3 + desc_match + category_match
        scored.append((score, book))
    
    # Sort by score, return top matches
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return top results (even if score is 0, return something)
    return [book for _, book in scored[:max_results]]


def _get_local_book_by_id(book_id: str) -> Optional[dict]:
    """Retrieve a book from local database by ID."""
    books = _load_local_books()
    for book in books:
        if book.get("book_id") == book_id or book.get("id") == book_id:
            return book
    return None

# ---------------------------------------------------------------------------
# 1. get_reader_profile
# ---------------------------------------------------------------------------

def get_reader_profile(reader_id: str) -> dict:
    """
    Fetch a reader's full profile: preferences, reading_log, and
    signals_pending_confirmation.

    Returns an empty-but-valid profile shape if the reader has no history
    yet, rather than raising — a brand new reader is a normal case, not
    an error.
    """
    profiles = _load_profiles()
    if reader_id in profiles:
        return profiles[reader_id]

    return {
        "reader_id": reader_id,
        "preferences": {},
        "reading_log": [],
        "signals_pending_confirmation": [],
    }


# ---------------------------------------------------------------------------
# 2. update_reader_profile
# ---------------------------------------------------------------------------

def update_reader_profile(reader_id: str, updates: dict) -> dict:
    """
    Apply updates to a reader's preferences and/or signals_pending_confirmation.

    `updates` is a partial patch, e.g.:
        {
          "preferences": {
            "pacing": {"value": "breakneck", "personal_confidence": 0.8, "peer_boost": 0.0}
          },
          "signals_pending_confirmation": [
            {"signal": "prefers_standalone", "source": "free_text_feedback",
             "confidence": 0.4, "temporary": True, "timestamp": "..."}
          ]
        }

    Preference keys are merged/overwritten individually (open-ended schema —
    new keys are just added). signals_pending_confirmation entries are
    appended, not merged, since each is its own observation.

    Promotion logic (pending -> preferences, requiring BOTH a confidence
    threshold AND explicit reader confirmation) is intentionally NOT done
    here — this function only applies whatever the caller decides. The
    decision of *when* to promote belongs in reader_modeling.py, which is
    the shared mechanism intent handlers call into.
    """
    profiles = _load_profiles()
    profile = profiles.get(reader_id) or {
        "reader_id": reader_id,
        "preferences": {},
        "reading_log": [],
        "signals_pending_confirmation": [],
    }

    if "preferences" in updates:
        profile.setdefault("preferences", {})
        for key, value in updates["preferences"].items():
            profile["preferences"][key] = value

    if "signals_pending_confirmation" in updates:
        profile.setdefault("signals_pending_confirmation", [])
        profile["signals_pending_confirmation"].extend(
            updates["signals_pending_confirmation"]
        )

    profiles[reader_id] = profile
    _save_profiles(profiles)
    return profile


# ---------------------------------------------------------------------------
# 3. log_book
# ---------------------------------------------------------------------------

def log_book(
    reader_id: str,
    book_id: str,
    outcome: str,
    rating: Optional[int] = None,
    dnf_reason: Optional[str] = None,
    signals_extracted: Optional[list] = None,
) -> dict:
    """
    Append a reading outcome to a reader's reading_log.

    `outcome` should be one of: "finished", "dnf", "in_progress" (extend as
    your log_outcome.py handler needs).
    `signals_extracted` records which preference keys this log entry
    contributed to updating — the traceability link described in the
    schema doc. Pass [] if this entry didn't move any preference.
    """
    profiles = _load_profiles()
    profile = profiles.get(reader_id) or {
        "reader_id": reader_id,
        "preferences": {},
        "reading_log": [],
        "signals_pending_confirmation": [],
    }

    profile.setdefault("reading_log", [])
    log_id = f"log_{len(profile['reading_log']) + 1:03d}"

    entry = {
        "log_id": log_id,
        "book_id": book_id,
        "outcome": outcome,
        "rating": rating,
        "dnf_reason": dnf_reason,
        "timestamp": _now_iso(),
        "signals_extracted": signals_extracted or [],
    }
    profile["reading_log"].append(entry)

    profiles[reader_id] = profile
    _save_profiles(profiles)
    return entry


# ---------------------------------------------------------------------------
# 4. search_books — LIVE Google Books API
# ---------------------------------------------------------------------------

def search_books(query: str, filters: Optional[dict] = None, max_results: int = 10) -> list[dict]:
    """
    Search Google Books API for books matching a query.

    Args:
        query (str): Natural-language search query (e.g. "fast-paced dark 
                     mystery"). This is constructed by the agent to be 
                     semantically meaningful to Google Books.
        filters (dict): Accepted for signature compatibility, but NOT used 
                        for filtering Google results (Google Books doesn't 
                        have pacing/tone/trope fields). Filtering against 
                        these happens in recommend.py after results return.
        max_results (int): Max results to return (Google Books API supports 
                          0-40, default 10).

    Returns:
        list[dict]: Normalized book objects, or empty list if no results or fallback used.

    Behavior:
        - Caches successful queries to avoid repeated API calls.
        - Once API rate limit (429) or timeout occurs, switches to local book 
          database fallback for remaining session.
        - If Google Books unavailable, returns matches from local books.json.

    Error handling:
        - API key missing → returns []
        - Query empty → returns []
        - Rate limit/timeout → logs, switches to local fallback
        - Malformed response → logs, switches to local fallback
    """
    
    global _GOOGLE_BOOKS_DEGRADED, _SEARCH_CACHE
    
    # Check cache first
    cache_key = (query, max_results)
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]
    
    # Skip Google Books if already rate limited
    if _GOOGLE_BOOKS_DEGRADED:
        print(f"[GOOGLE BOOKS DEGRADED] Using local fallback for: {query}")
        result = _search_local_books(query, max_results)
        _SEARCH_CACHE[cache_key] = result
        return result

    if not GOOGLE_BOOKS_API_KEY:
        print("ERROR: GOOGLE_BOOKS_API_KEY not set in environment")
        return []

    if not query or not query.strip():
        return []

    query = query.strip()
    max_results = min(max(1, max_results), 40)

    try:
        url = f"{GOOGLE_BOOKS_API_BASE}/volumes"
        params = {
            "q": query,
            "maxResults": max_results,
            "key": GOOGLE_BOOKS_API_KEY,
        }

        response = requests.get(
            url,
            params=params,
            timeout=GOOGLE_BOOKS_API_TIMEOUT,
        )

        if response.status_code == 429:
            print(f"WARNING: Google Books API rate limited (429)")
            _GOOGLE_BOOKS_DEGRADED = True
            # Fall through to local fallback
        elif response.status_code == 503:
            print("WARNING: Google Books API temporarily unavailable (503)")
            _GOOGLE_BOOKS_DEGRADED = True
        elif response.status_code != 200:
            print(f"ERROR: Google Books API returned {response.status_code}")
            _GOOGLE_BOOKS_DEGRADED = True
        else:
            # Success — return and cache
            data = response.json()
            items = data.get("items", [])
            normalized = []
            for volume in items:
                normalized_vol = _normalize_google_book(volume)
                if normalized_vol:
                    normalized.append(normalized_vol)
            _SEARCH_CACHE[cache_key] = normalized
            return normalized

    except requests.Timeout:
        print(f"ERROR: Google Books API timeout")
        _GOOGLE_BOOKS_DEGRADED = True
    except requests.RequestException as e:
        print(f"ERROR: Google Books API network error")
        _GOOGLE_BOOKS_DEGRADED = True
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: Could not parse Google Books response")
        _GOOGLE_BOOKS_DEGRADED = True
    
    # Fall back to local
    print(f"Falling back to local database for: {query}")
    result = _search_local_books(query, max_results)
    _SEARCH_CACHE[cache_key] = result
    return result

    # if not GOOGLE_BOOKS_API_KEY:
    #     print("ERROR: GOOGLE_BOOKS_API_KEY not set in environment")
    #     return []

    # if not query or not query.strip():
    #     return []

    # query = query.strip()
    # max_results = min(max(1, max_results), 40)  # Google API limit

    # try:
    #     url = f"{GOOGLE_BOOKS_API_BASE}/volumes"
    #     params = {
    #         "q": query,
    #         "maxResults": max_results,
    #         "key": GOOGLE_BOOKS_API_KEY,
    #     }

    #     response = requests.get(
    #         url,
    #         params=params,
    #         timeout=GOOGLE_BOOKS_API_TIMEOUT,
    #     )

    #     # Handle HTTP errors
    #     if response.status_code == 429:
    #         print(f"WARNING: Google Books API rate limited (429). Retry after {response.headers.get('Retry-After', 'unknown')} seconds")
    #         return []
    #     elif response.status_code == 503:
    #         print("WARNING: Google Books API temporarily unavailable (503)")
    #         return []
    #     elif response.status_code != 200:
    #         print(f"ERROR: Google Books API returned {response.status_code}: {response.text[:200]}")
    #         return []

    #     data = response.json()
    #     items = data.get("items", [])

    #     # Normalize and filter valid volumes
    #     normalized = []
    #     for volume in items:
    #         normalized_vol = _normalize_google_book(volume)
    #         if normalized_vol:
    #             normalized.append(normalized_vol)

    #     return normalized

    # except requests.Timeout:
    #     print(f"ERROR: Google Books API timeout (>{GOOGLE_BOOKS_API_TIMEOUT}s)")
    # except requests.RequestException as e:
    #     print(f"ERROR: Google Books API network error: {str(e)[:200]}")
    # except (json.JSONDecodeError, KeyError) as e:
    #     print(f"ERROR: Could not parse Google Books response: {str(e)[:200]}")
    
    # # Fall back to local book database
    # print(f"Falling back to local book database for query: {query}")
    # return _search_local_books(query, max_results)


# ---------------------------------------------------------------------------
# 5. get_book_by_id — LIVE Google Books API (resolve mode)
# ---------------------------------------------------------------------------

def get_book_by_id(book_id: str) -> Optional[dict]:
    """
    Resolve a single known book_id into its display metadata.

    Used by the "resolve" path in recommend.py when a peer match was already
    found — cheaper than a discovery search since we already know exactly
    which book we want, we just need its title/author/description.

    Args:
        book_id (str): Google volumeId or local book_id.

    Returns:
        dict: Normalized book object, or None if not found or API error.

    Behavior:
        - Caches successful lookups to avoid repeated API calls.
        - Once API rate limit (429) or timeout occurs, switches to local book 
          database fallback for remaining session.
        - Checks local books.json if Google Books unavailable.

    Error handling:
        - API key missing → checks local database
        - book_id empty/None → returns None
        - 404 (not found) → returns None
        - Rate limit/timeout → logs, switches to local fallback
    """
    global _GOOGLE_BOOKS_DEGRADED, _BOOK_BY_ID_CACHE
    
    # Check cache first
    if book_id in _BOOK_BY_ID_CACHE:
        return _BOOK_BY_ID_CACHE[book_id]
    
    # Skip Google Books if already rate limited
    if _GOOGLE_BOOKS_DEGRADED:
        result = _get_local_book_by_id(book_id)
        _BOOK_BY_ID_CACHE[book_id] = result
        return result

    if not GOOGLE_BOOKS_API_KEY:
        return None

    if not book_id or not book_id.strip():
        return None

    book_id = book_id.strip()

    try:
        url = f"{GOOGLE_BOOKS_API_BASE}/volumes/{book_id}"
        params = {"key": GOOGLE_BOOKS_API_KEY}

        response = requests.get(
            url,
            params=params,
            timeout=GOOGLE_BOOKS_API_TIMEOUT,
        )

        if response.status_code == 404:
            _BOOK_BY_ID_CACHE[book_id] = None
            return None
        elif response.status_code == 429:
            print("WARNING: Google Books API rate limited (429)")
            _GOOGLE_BOOKS_DEGRADED = True
        elif response.status_code != 200:
            print(f"ERROR: Google Books API returned {response.status_code}")
            _GOOGLE_BOOKS_DEGRADED = True
        else:
            # Success — cache and return
            volume = response.json()
            result = _normalize_google_book(volume)
            _BOOK_BY_ID_CACHE[book_id] = result
            return result

    except (requests.Timeout, requests.RequestException, json.JSONDecodeError, KeyError):
        print(f"[ERROR] Checking local database for: {book_id}")
        _GOOGLE_BOOKS_DEGRADED = True
    
    # Fall back to local
    result = _get_local_book_by_id(book_id)
    _BOOK_BY_ID_CACHE[book_id] = result
    return result
    # if not GOOGLE_BOOKS_API_KEY:
    #     print("ERROR: GOOGLE_BOOKS_API_KEY not set in environment")
    #     return None

    # if not book_id or not book_id.strip():
    #     return None

    # book_id = book_id.strip()

    # try:
    #     url = f"{GOOGLE_BOOKS_API_BASE}/volumes/{book_id}"
    #     params = {"key": GOOGLE_BOOKS_API_KEY}

    #     response = requests.get(
    #         url,
    #         params=params,
    #         timeout=GOOGLE_BOOKS_API_TIMEOUT,
    #     )

    #     if response.status_code == 404:
    #         # Book not found — this is expected, not an error to log loudly
    #         return None
    #     elif response.status_code == 429:
    #         print("WARNING: Google Books API rate limited (429)")
    #         return None
    #     elif response.status_code != 200:
    #         print(f"ERROR: Google Books API returned {response.status_code}")
    #         return None

    #     volume = response.json()
    #     return _normalize_google_book(volume)

    # except requests.Timeout:
    #     print(f"ERROR: Google Books API timeout for book_id={book_id}")
    # except requests.RequestException as e:
    #     print(f"ERROR: Google Books API network error for book_id={book_id}: {str(e)[:200]}")
    # except (json.JSONDecodeError, KeyError) as e:
    #     print(f"ERROR: Could not parse Google Books response for book_id={book_id}")
    
    # # Fall back to local book database
    # print(f"Falling back to local book database for book_id={book_id}")
    # return _get_local_book_by_id(book_id)


# ---------------------------------------------------------------------------
# 6. get_author_stats — STUB, authorization enforced in Part 4
# ---------------------------------------------------------------------------

def get_author_stats(author_id: str, requesting_user: dict) -> dict:
    """
    Return anonymized, aggregate reception stats for an author's own book —
    never individual reader data.

    `requesting_user` is the full user record from data/users.json (e.g.
    {"id": "author_01", "role": "author"}). The signature takes this
    now, even though full enforcement (role == "author" AND
    requesting_user matches the author being queried) lands in Part 4 — the
    point is that the boundary is a code-level check on this parameter,
    not a prompt instruction asking the model to behave.

    STUB: returns hardcoded aggregate numbers for now.
    """
    # Minimal boundary check now; Part 4 will make this the real,
    # fully-tested authorization gate.
    if requesting_user.get("role") != "author":
        raise PermissionError("Only author-role users can view reception stats.")
    if requesting_user.get("id") != author_id:
        raise PermissionError("Authors can only view stats for their own book.")

    return {
        "author_id": author_id,
        "total_readers": 214,
        "completion_rate": 0.78,
        "average_rating": 4.3,
        "top_pacing_match": "breakneck",
        "top_tone_match": "dark",
        "dnf_reasons_summary": {
            "too_slow_to_start": 12,
            "unresolved_ending": 5,
        },
    }