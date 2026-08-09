"""
Reader Modeling — shared mechanism for the Adaptive Reader Agent.

This is NOT an intent. It's called by intent handlers (get_recommendation,
log_reading_outcome, reading_insights) to do three things:

1. find_peer_match()       — peer-based recommendation support
2. process_feedback_signal() — turn free-text feedback into a structured,
                                confidence-scored signal
3. try_promote_signals()   — decide whether pending signals graduate into
                              stated preferences

Design rule this module enforces: a signal only becomes a stated preference
when BOTH (a) its confidence crosses a threshold AND (b) the reader has
explicitly confirmed it. Confidence alone doesn't promote anything —
confirmation alone doesn't either. See docs/design/reader_profile_schema.md
for the full rationale.
"""

from typing import Optional
from app.tools import get_reader_profile, update_reader_profile, _load_profiles

PROMOTION_CONFIDENCE_THRESHOLD = 0.6

# The preference fields we compare across readers for similarity.
# Open-ended in the schema, but this is the fixed set we currently know
# how to compare numerically — extend as new signal types get added.
COMPARABLE_PREFERENCE_KEYS = ["pacing", "tone", "trope", "mood"]


# ---------------------------------------------------------------------------
# 1. Peer similarity / peer-based recommendation support
# ---------------------------------------------------------------------------

def _preference_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Compare two readers' preferences and return a similarity score in [0, 1].

    Simple weighted-overlap approach: for each preference key both readers
    have, score 1.0 if the values match, 0.0 if they don't, weighted by
    each reader's personal_confidence in that field (low-confidence
    preferences shouldn't count as strongly toward "these readers are
    alike"). Total is normalized by the number of shared keys.

    This is intentionally simple for now — swap in a real vector/cosine
    approach later if the demo needs finer-grained matching, but a
    defensible, explainable metric beats a black-box one for a capstone.
    """
    prefs_a = profile_a.get("preferences", {})
    prefs_b = profile_b.get("preferences", {})

    shared_keys = [k for k in COMPARABLE_PREFERENCE_KEYS if k in prefs_a and k in prefs_b]
    if not shared_keys:
        return 0.0

    total_weight = 0.0
    total_score = 0.0

    for key in shared_keys:
        conf_a = prefs_a[key].get("personal_confidence", 0.0)
        conf_b = prefs_b[key].get("personal_confidence", 0.0)
        weight = (conf_a + conf_b) / 2
        match = 1.0 if prefs_a[key].get("value") == prefs_b[key].get("value") else 0.0

        total_score += match * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def find_peer_match(reader_id: str, similarity_threshold: float = 0.6) -> Optional[str]:
    """
    Look across all OTHER readers for the most similar profile to
    `reader_id`. If that peer has a reading_log entry with a proven
    positive outcome (finished, rating >= 4), return that book_id.

    Returns None if no peer clears the similarity threshold, or if the
    best-matching peer has no proven positive outcome to recommend from.
    """
    current_profile = get_reader_profile(reader_id)
    all_profiles = _load_profiles()

    best_score = -1.0
    best_peer_id = None

    for other_id, other_profile in all_profiles.items():
        if other_id == reader_id:
            continue
        score = _preference_similarity(current_profile, other_profile)
        if score > best_score:
            best_score = score
            best_peer_id = other_id

    if best_peer_id is None or best_score < similarity_threshold:
        return None

    peer_profile = all_profiles[best_peer_id]
    proven_positive = [
        entry for entry in peer_profile.get("reading_log", [])
        if entry.get("outcome") == "finished" and (entry.get("rating") or 0) >= 4
    ]

    if not proven_positive:
        return None

    # Most recent proven-positive read from the best-matching peer.
    proven_positive.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return proven_positive[0]["book_id"]


# ---------------------------------------------------------------------------
# 2. Free-text feedback -> structured signal
# ---------------------------------------------------------------------------

def process_feedback_signal(
    reader_id: str,
    signal_name: str,
    confidence: float,
    source: str,
    temporary: bool = True,
) -> dict:
    """
    File a new observation into signals_pending_confirmation. This does NOT
    promote it into preferences — that only happens via try_promote_signals()
    once the reader has explicitly confirmed it.

    `signal_name` should describe what was observed (e.g. "prefers_standalone",
    "dislikes_slow_starts"). `source` records where it came from
    (e.g. "free_text_feedback", "recommendation_feedback") for traceability.
    """
    import datetime
    signal_entry = {
        "signal": signal_name,
        "source": source,
        "confidence": confidence,
        "temporary": temporary,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    update_reader_profile(reader_id, {"signals_pending_confirmation": [signal_entry]})
    return signal_entry


# ---------------------------------------------------------------------------
# 3. Promotion: pending signal -> stated preference
# ---------------------------------------------------------------------------

def try_promote_signal(
    reader_id: str,
    signal_name: str,
    reader_confirmed: bool,
    preference_key: Optional[str] = None,
    preference_value: Optional[str] = None,
) -> bool:
    """
    Attempt to promote a pending signal into `preferences`.

    Promotes ONLY if both hold:
      (a) the signal's stored confidence >= PROMOTION_CONFIDENCE_THRESHOLD
      (b) reader_confirmed is True (the reader explicitly agreed this is a
          real, durable preference — not a one-off comment)

    If promoted, `preference_key`/`preference_value` determine what gets
    written into preferences (e.g. signal "prefers_standalone" ->
    preference_key="trope", preference_value="standalone"). The caller
    (intent handler) decides this mapping since it knows the conversation
    context; this function only enforces the promotion gate.

    Returns True if promotion happened, False otherwise. Does not raise or
    remove the pending signal if promotion doesn't happen — an unconfirmed
    or low-confidence signal just stays pending for next time.
    """
    profile = get_reader_profile(reader_id)
    pending = profile.get("signals_pending_confirmation", [])

    matching = next((s for s in pending if s["signal"] == signal_name), None)
    if matching is None:
        return False

    if matching["confidence"] < PROMOTION_CONFIDENCE_THRESHOLD or not reader_confirmed:
        return False

    if preference_key and preference_value:
        update_reader_profile(reader_id, {
            "preferences": {
                preference_key: {
                    "value": preference_value,
                    "personal_confidence": matching["confidence"],
                    "peer_boost": 0.0,
                }
            }
        })

    # Remove the promoted signal from the pending list.
    remaining = [s for s in pending if s["signal"] != signal_name]
    all_profiles = _load_profiles()
    all_profiles[reader_id]["signals_pending_confirmation"] = remaining
    from app.tools import _save_profiles
    _save_profiles(all_profiles)

    return True