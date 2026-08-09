"""
view_aggregate_reception intent handler.

Author-only. Never touches individual reader data or the reader profile —
only the anonymized aggregate returned by tools.get_author_stats(), which
enforces the authorization boundary in code (see tools.py docstring).

This handler does not catch-and-hide the PermissionError from
get_author_stats — if a non-author or wrong-author user somehow reaches
this node (shouldn't happen given intent classification + persona
switching, but defense in depth matters), the refusal should be explicit
and visible, not silently swallowed into a vague response.
"""

import os
from anthropic import Anthropic

from app.tools import get_author_stats

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def handle_view_aggregate_reception(state: dict) -> dict:
    user = state["user"]
    author_id = user["id"]

    try:
        stats = get_author_stats(author_id=author_id, requesting_user=user)
    except PermissionError as e:
        return {
            "handler_result": {"error": str(e)},
            "response": (
                "I can't show reception stats here — this view is only "
                "available to the author of the book, and only for their "
                "own work."
            ),
            "profile_updates": None,
        }

    prompt = f"""Aggregate reception stats for this author's book:
{stats}

Write a short, natural summary (2-4 sentences) of how the book is being
received, grounded only in the numbers above. Do not invent details not
present in the data.
"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "handler_result": {"stats": stats},
        "response": response.content[0].text.strip(),
        "profile_updates": None,  # aggregate stats never write to a reader profile
    }