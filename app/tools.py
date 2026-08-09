"""
Tool layer for the Adaptive Reader Agent.

This is the seam between the agent's reasoning (LangGraph, intent handlers)
and where data actually lives. Every intent handler should call through
these functions rather than touching data/reader_profiles.json directly.

Why this matters for Part 3 / Part 4:
- search_books() is the deliberate stub that gets swapped for a live Google
  Books API call in Part 3. Nothing above this function should need to change.
- get_author_stats() already takes `requesting_user` even though the real
  authorization check isn't built until Part 4 — the seam for "enforce this
  in code, not just in the prompt" needs to exist now, not be bolted on later.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
READER_PROFILES_PATH = DATA_DIR / "reader_profiles.json"


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
# 4. search_books  — STUB, swapped for live Google Books API in Part 3
# ---------------------------------------------------------------------------

def search_books(query: str, filters: Optional[dict] = None) -> list[dict]:
    """
    Search for books matching a query and optional filters
    (e.g. {"pacing": "breakneck", "tone": "dark"}).

    STUB: returns a small hardcoded catalog for now. This is the deliberate
    seam for Part 3 — when swapped for a real Google Books API call, the
    return shape (list of dicts with these keys) should stay the same so
    nothing upstream (planner, recommend handler) needs to change.
    """
    fake_catalog = [
        {
            "book_id": "b001",
            "title": "The Glass Labyrinth",
            "author": "M. Okafor",
            "pacing": "breakneck",
            "tone": "dark",
            "trope": "found_family",
            "description": "A fugitive engineer and a defected soldier race across "
                            "a collapsing city-state to stop a coup.",
        },
        {
            "book_id": "b002",
            "title": "Winter's Ledger",
            "author": "R. Voss",
            "pacing": "measured",
            "tone": "hopeful",
            "trope": "found_family",
            "description": "A retired accountant uncovers a decades-old fraud in "
                            "her small mountain town.",
        },
        {
            "book_id": "b003",
            "title": "Static Bloom",
            "author": "J. Adeyemi",
            "pacing": "breakneck",
            "tone": "melancholic",
            "trope": "unreliable_narrator",
            "description": "A signal-jamming technician starts hearing a voice "
                            "that shouldn't be on the network.",
        },
    ]

    if not filters:
        return fake_catalog

    def matches(book: dict) -> bool:
        return all(book.get(k) == v for k, v in filters.items())

    return [b for b in fake_catalog if matches(b)]


def get_book_by_id(book_id: str) -> Optional[dict]:
    """
    Resolve a single known book_id into its display metadata.

    STUB: looks up against the same hardcoded fake catalog as search_books().
    This is the "resolve" call the peer-match recommendation path uses —
    cheaper than a discovery search since we already know exactly which
    book we want, we just need its title/author/description to show the
    reader. Swapped for a real Google Books "get by id" call in Part 3.
    """
    for book in search_books(query=""):
        if book["book_id"] == book_id:
            return book
    return None


def find_book_by_title(title: str) -> Optional[dict]:
    """
    Resolve a reader-mentioned title (e.g. from log_reading_outcome) to a
    book record via fuzzy, case-insensitive substring matching on title.

    This is DELIBERATELY separate from search_books(): search_books's
    `query` argument is not used for text matching (it filters purely on
    tags like pacing/tone/trope, which is correct for discovery mode) — so
    reusing it here would silently resolve to the wrong book regardless of
    what title was actually mentioned. This function is Part 3's other
    seam: swapped for a real Google Books title search when live.
    """
    if not title:
        return None
    needle = title.strip().lower()
    for book in search_books(query=""):
        if needle in book["title"].lower() or book["title"].lower() in needle:
            return book
    return None


# ---------------------------------------------------------------------------
# 5. get_author_stats — STUB, authorization enforced in Part 4
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