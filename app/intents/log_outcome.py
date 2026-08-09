"""
log_reading_outcome intent handler.

Merges rating and DNF reporting into one flow. Runs a lightweight
diagnostic pass (which dimension — length/pacing/mood/style/genre — drove
a DNF or strong reaction) before filing anything into Reader Modeling.

Important: this handler NEVER writes directly into `preferences`. It only
writes to reading_log (via tools.log_book) and stages an observation in
signals_pending_confirmation (via reader_modeling.process_feedback_signal).
Promotion into a stated preference still requires the separate confidence +
explicit-confirmation gate — one DNF is a data point, not a stated taste.
"""

import json
import os
from anthropic import Anthropic

from app.tools import find_book_by_title, log_book
from app.reader_modeling import process_feedback_signal

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DIAGNOSTIC_DIMENSIONS = ["length", "pacing", "mood", "style", "genre"]


def _parse_outcome(message: str) -> dict:
    """
    Extract structured outcome data from the reader's free-text message.

    Returns:
        {
          "book_title": str | null,
          "outcome": "finished" | "dnf" | "in_progress",
          "rating": int | null,        # 1-5, only for finished
          "dnf_dimension": one of DIAGNOSTIC_DIMENSIONS | null,
          "dnf_reason": str | null,    # short free-text reason, if given
        }
    """
    prompt = f"""The reader said: "{message}"

Extract the reading outcome as JSON. Fields:
- book_title: the book being discussed, or null if not named
- outcome: "finished", "dnf" (did not finish / gave up), or "in_progress"
- rating: integer 1-5 if the reader gave or implied a rating, else null
- dnf_dimension: if outcome is "dnf", pick the single best-fitting category
  from {DIAGNOSTIC_DIMENSIONS} for WHY they stopped (e.g. "too slow to
  start" -> "pacing", "too long" -> "length", "not my mood right now" ->
  "mood", "writing style didn't click" -> "style", "wrong genre for me" ->
  "genre"). Null if outcome isn't "dnf" or the reason isn't clear.
- dnf_reason: the reader's own short reason in their words, or null

Respond ONLY with the JSON object, no other text.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        print(f"[log_outcome DEBUG] parsed outcome: {parsed}")
        return parsed
    except (json.JSONDecodeError, IndexError) as e:
        print(f"[log_outcome DEBUG] Failed to parse. Raw model output was:\n{raw_text!r}\nError: {e}")
        return {
            "book_title": None,
            "outcome": "finished",
            "rating": None,
            "dnf_dimension": None,
            "dnf_reason": None,
        }


def _resolve_book(book_title: str) -> dict | None:
    """Resolve a mentioned title to a book record via fuzzy title matching."""
    if not book_title:
        return None
    return find_book_by_title(book_title)


def handle_log_reading_outcome(state: dict) -> dict:
    reader_id = state["user"]["id"]
    message = state["message"]

    parsed = _parse_outcome(message)
    book = _resolve_book(parsed.get("book_title"))

    if book is None:
        return {
            "handler_result": {"resolved": False, "parsed": parsed},
            "response": (
                "I couldn't tell which book you meant — could you give me "
                "the title so I can log it correctly?"
            ),
            "profile_updates": None,
        }

    # signals_extracted: which preference dimension(s) this entry speaks to.
    # For a DNF, that's the diagnosed dimension. For a finished book with a
    # rating, we treat pacing/tone/trope as touched if the book's own
    # metadata carries those tags (traceability, not a promotion).
    signals_extracted = []
    if parsed["outcome"] == "dnf" and parsed.get("dnf_dimension"):
        signals_extracted.append(parsed["dnf_dimension"])
    elif parsed["outcome"] == "finished":
        signals_extracted = [k for k in ["pacing", "tone", "trope"] if book.get(k)]

    log_entry = log_book(
        reader_id=reader_id,
        book_id=book["book_id"],
        outcome=parsed["outcome"],
        rating=parsed.get("rating"),
        dnf_reason=parsed.get("dnf_reason"),
        signals_extracted=signals_extracted,
    )

    # Stage an observation for Reader Modeling — NOT a direct preference
    # write. A single DNF or single high rating is one data point; it needs
    # confidence to build and explicit confirmation before it becomes a
    # stated preference (see docs/design/reader_profile_schema.md).
    pending_signal = None
    if parsed["outcome"] == "dnf" and parsed.get("dnf_dimension"):
        dimension = parsed["dnf_dimension"]
        book_value = book.get(dimension)
        signal_name = f"dislikes_{book_value}_{dimension}" if book_value else f"dnf_{dimension}_issue"
        pending_signal = {
            "signal_name": signal_name,
            "confidence": 0.5,  # single data point — moderate, not high
            "source": "log_reading_outcome",
        }
    elif parsed["outcome"] == "finished" and (parsed.get("rating") or 0) >= 4:
        for dimension in ["pacing", "tone", "trope"]:
            if book.get(dimension):
                pending_signal = {
                    "signal_name": f"enjoys_{book[dimension]}_{dimension}",
                    "confidence": 0.5,
                    "source": "log_reading_outcome",
                }
                break  # one signal per turn keeps this from overwhelming pending list

    response_text = (
        f"Logged \"{book['title']}\" as {parsed['outcome']}"
        + (f" ({parsed['rating']}/5)" if parsed.get("rating") else "")
        + (f" — noted that pacing may have been an issue." if parsed.get("dnf_dimension") == "pacing" else "")
        + "."
    )

    return {
        "handler_result": {"resolved": True, "book": book, "parsed": parsed, "log_entry": log_entry},
        "response": response_text,
        "profile_updates": {"pending_signals": [pending_signal]} if pending_signal else None,
    }