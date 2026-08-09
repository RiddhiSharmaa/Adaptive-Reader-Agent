"""
reading_insights intent handler.

Read-only — never writes to the reader profile (no log entry, no pending
signal). Answers questions about the reader's OWN patterns: what they
finish vs DNF, how preferences have trended, why they might be in a slump.

Resolves book_ids in the reading_log back to titles via get_book_by_id
(since the profile only ever stores book_id, never metadata) so the
response can reference actual titles instead of opaque ids.
"""

import os
from anthropic import Anthropic

from app.tools import get_reader_profile, get_book_by_id
from app.rag_index import knowledge_collection

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _query_rag(message: str, n_results: int = 3) -> str:
    """
    Ground the response in the knowledge base when the question touches
    concepts like reading slumps, pacing mismatches, etc. — not used for
    book data, only reading-pattern concepts.
    """
    results = knowledge_collection.query(query_texts=[message], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents) if documents else ""


def _build_log_summary(reading_log: list[dict]) -> str:
    """Resolve book_ids to titles and format the log as readable lines."""
    lines = []
    for entry in reading_log:
        book = get_book_by_id(entry["book_id"])
        title = book["title"] if book else entry["book_id"]
        detail = f"rated {entry['rating']}/5" if entry.get("rating") else entry.get("dnf_reason", "")
        lines.append(f"- {title}: {entry['outcome']} ({detail})")
    return "\n".join(lines) if lines else "(no reading history yet)"


def handle_reading_insights(state: dict) -> dict:
    reader_id = state["user"]["id"]
    message = state["message"]

    profile = get_reader_profile(reader_id)
    log_summary = _build_log_summary(profile.get("reading_log", []))
    preferences_summary = ", ".join(
        f"{k}: {v['value']} (confidence {v['personal_confidence']})"
        for k, v in profile.get("preferences", {}).items()
    ) or "(no established preferences yet)"

    rag_context = _query_rag(message)

    prompt = f"""The reader asked: "{message}"

Their reading history:
{log_summary}

Their current established preferences:
{preferences_summary}

{"Relevant background: " + rag_context if rag_context else ""}

Answer the reader's question about their OWN reading patterns, grounded
only in the history and preferences above. Do not recommend a new book —
that's a different feature. If the history is too thin to answer
confidently, say so plainly rather than overgeneralizing from one data point.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "handler_result": {"log_summary": log_summary, "preferences_summary": preferences_summary},
        "response": response.content[0].text.strip(),
        "profile_updates": None,  # read-only intent — never writes
    }