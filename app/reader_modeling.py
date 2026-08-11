"""
Reader Modeling — shared mechanism for the Adaptive Reader Agent.

This is NOT an intent. It's called by intent handlers
(get_recommendation, log_reading_outcome, reading_insights) to do three things:

1. find_peer_match()          — peer-based recommendation support
2. process_feedback_signal() — turn free-text feedback into a structured,
                                confidence-scored signal
3. try_promote_signal()       — decide whether pending signals graduate into
                                stated preferences

Design rule:
A signal only becomes a stated preference when BOTH:
(a) its confidence crosses a threshold AND
(b) the reader has explicitly confirmed it.

Confidence alone does not promote anything.
Confirmation alone does not promote anything.

See docs/design/reader_profile_schema.md for the full rationale.
"""

from typing import Optional

from app.tools import (
    get_reader_profile,
    update_reader_profile,
    _load_profiles,
    _save_profiles,
)


PROMOTION_CONFIDENCE_THRESHOLD = 0.6

# The preference fields we currently compare across readers.
# The schema remains open-ended, but peer similarity only compares
# dimensions for which we have an explicit comparison strategy.

COMPARABLE_PREFERENCE_KEYS = [
    "pacing",
    "tone",
    "trope",
    "mood",
]


# ---------------------------------------------------------------------------
# 1. Peer similarity / peer-based recommendation support
# ---------------------------------------------------------------------------

def _preference_similarity(profile_a: dict, profile_b: dict) -> float:
    """
    Compare two readers' preferences and return a similarity score in [0, 1].

    For every preference key both readers have:
      - score 1.0 if the values match
      - score 0.0 if they differ

    The contribution is weighted by the average personal confidence of the
    two readers for that preference.

    The score is normalized by the total confidence weight of the shared
    dimensions.

    This is intentionally simple and explainable for the capstone.
    """

    prefs_a = profile_a.get("preferences", {})
    prefs_b = profile_b.get("preferences", {})

    shared_keys = [
        key
        for key in COMPARABLE_PREFERENCE_KEYS
        if key in prefs_a and key in prefs_b
    ]

    if not shared_keys:
        return 0.0

    total_weight = 0.0
    total_score = 0.0

    for key in shared_keys:
        pref_a = prefs_a.get(key, {})
        pref_b = prefs_b.get(key, {})

        conf_a = float(pref_a.get("personal_confidence", 0.0) or 0.0)
        conf_b = float(pref_b.get("personal_confidence", 0.0) or 0.0)

        # Defensive clamping so malformed confidence values cannot
        # distort the similarity calculation.
        conf_a = max(0.0, min(1.0, conf_a))
        conf_b = max(0.0, min(1.0, conf_b))

        weight = (conf_a + conf_b) / 2.0

        value_a = pref_a.get("value")
        value_b = pref_b.get("value")

        match = 1.0 if value_a == value_b else 0.0

        total_score += match * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_score / total_weight


def find_peer_match(
    reader_id: str,
    similarity_threshold: float = 0.6,
) -> Optional[str]:
    """
    Look across all OTHER readers for the most similar profile.

    If the best-matching peer clears the similarity threshold and has at
    least one proven positive reading outcome (finished + rating >= 4),
    return the book_id of their most recent proven-positive read.

    Returns None if:
      - the reader does not exist,
      - no other reader is sufficiently similar,
      - or the best matching reader has no proven positive outcome.
    """

    current_profile = get_reader_profile(reader_id)
    all_profiles = _load_profiles()

    best_score = -1.0
    best_peer_id = None

    for other_id, other_profile in all_profiles.items():
        if other_id == reader_id:
            continue

        score = _preference_similarity(
            current_profile,
            other_profile,
        )

        if score > best_score:
            best_score = score
            best_peer_id = other_id

    if best_peer_id is None:
        return None

    if best_score < similarity_threshold:
        return None

    peer_profile = all_profiles.get(best_peer_id, {})

    proven_positive = [
        entry
        for entry in peer_profile.get("reading_log", [])
        if (
            entry.get("outcome") == "finished"
            and (entry.get("rating") or 0) >= 4
            and entry.get("book_id")
        )
    ]

    if not proven_positive:
        return None

    # Most recent proven-positive read from the best-matching peer.
    proven_positive.sort(
        key=lambda entry: entry.get("timestamp", ""),
        reverse=True,
    )

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
    File a new observation into signals_pending_confirmation.

    This does NOT promote the observation into preferences.

    Promotion only happens through try_promote_signal() after the reader
    explicitly confirms the signal and its confidence meets the threshold.

    Args:
        reader_id: Reader whose profile receives the observation.
        signal_name: Structured observation, e.g. "dnf_due_to_pacing".
        confidence: Confidence score for this observation.
        source: Origin of the observation.
        temporary: Whether the observation remains provisional.
    """

    import datetime

    confidence = max(0.0, min(1.0, float(confidence)))

    signal_entry = {
        "signal": signal_name,
        "source": source,
        "confidence": confidence,
        "temporary": temporary,
        "timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    update_reader_profile(
        reader_id,
        {
            "signals_pending_confirmation": [signal_entry]
        },
    )

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
    Attempt to promote a pending signal into preferences.

    Promotion happens ONLY when both conditions hold:

      (a) stored confidence >= PROMOTION_CONFIDENCE_THRESHOLD
      (b) reader_confirmed is True

    If promoted:
      - the preference is written to the reader profile
      - the promoted signal is removed from pending confirmation

    If either condition fails:
      - nothing is promoted
      - the pending signal remains available for future confirmation

    Returns:
        True if promotion happened, otherwise False.
    """

    profile = get_reader_profile(reader_id)
    pending = profile.get("signals_pending_confirmation", [])

    # Be defensive about malformed entries in persisted data.
    matching = next(
        (
            signal
            for signal in pending
            if isinstance(signal, dict)
            and signal.get("signal") == signal_name
        ),
        None,
    )

    if matching is None:
        return False

    try:
        confidence = float(matching.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < PROMOTION_CONFIDENCE_THRESHOLD:
        return False

    if not reader_confirmed:
        return False

    # Only write a preference when the caller supplied an explicit mapping.
    # This preserves the separation between observed signals and durable
    # reader preferences.
    if preference_key and preference_value:
        update_reader_profile(
            reader_id,
            {
                "preferences": {
                    preference_key: {
                        "value": preference_value,
                        "personal_confidence": confidence,
                        "peer_boost": 0.0,
                    }
                }
            },
        )

    # Remove only the matching signal rather than accidentally deleting
    # unrelated observations.
    all_profiles = _load_profiles()

    if reader_id not in all_profiles:
        return False

    current_pending = all_profiles[reader_id].get(
        "signals_pending_confirmation",
        [],
    )

    removed = False
    remaining = []

    for signal in current_pending:
        if (
            not removed
            and isinstance(signal, dict)
            and signal.get("signal") == signal_name
        ):
            removed = True
            continue

        remaining.append(signal)

    if not removed:
        return False

    all_profiles[reader_id]["signals_pending_confirmation"] = remaining
    _save_profiles(all_profiles)

    return True