"""
Handler for reading_insights intent.

Core rule: Use RAG internally to diagnose patterns,
but respond conversationally to the user.
Do NOT expose backend reasoning.
"""

from app.tools import get_reader_profile, get_book_by_id
from app.rag_index import knowledge_collection
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _query_rag_for_insights(user_question: str) -> str:
    """Query RAG for relevant guidance."""
    try:
        results = knowledge_collection.query(
            query_texts=[user_question],
            n_results=3
        )
        
        if results and results.get("documents"):
            docs = results["documents"][0]
            return "\n".join(docs) if docs else ""
    except Exception as e:
        print(f"[RAG ERROR] {e}")
    
    return ""


def _diagnose_pattern(profile: dict) -> str:
    """
    Analyze reading history to detect patterns.
    Returns: plain English diagnosis (for internal use).
    """
    reading_log = profile.get("reading_log", [])
    
    if not reading_log:
        return "no_history"
    
    # Count DNFs vs finishes
    dnf_count = sum(1 for log in reading_log if log.get("outcome") == "dnf")
    finish_count = sum(1 for log in reading_log if log.get("outcome") == "finished")
    
    # ⭐ FIX: Check if LAST read was long ago (dormant period)
    last_read = reading_log[-1].get("timestamp", "")
    
    # Simple heuristic: if last read timestamp is from 2025 or earlier, 
    # and we're now in 2026, that's a long gap
    if last_read and "2025" in last_read:
        return "slump_dormant"
    
    # Check for DNF patterns
    dnf_reasons = [
        log.get("dnf_reason", "")
        for log in reading_log
        if log.get("outcome") == "dnf" and log.get("dnf_reason")
    ]
    
    if dnf_reasons and all("too slow" in reason for reason in dnf_reasons):
        return "pacing_mismatch"
    
    if dnf_count > finish_count:
        return "general_slump"
    
    return "normal"


def handle_reading_insights(state: dict) -> dict:
    """
    Analyze reader's pattern and respond conversationally.
    
    Steps:
    1. Load profile + reading log
    2. Query RAG for general guidance
    3. Diagnose pattern (internal only, not shown)
    4. Respond conversationally with actionable insight
    """
    
    reader_id = state["user"]["id"]
    message = state["message"].strip()
    profile = get_reader_profile(reader_id)
    
    # Query RAG for context
    rag_context = _query_rag_for_insights(message)
    
    # Diagnose pattern (internal)
    pattern = _diagnose_pattern(profile)
    
    reading_log = profile.get("reading_log", [])
    preferences = profile.get("preferences", {})
    
    # ================================================================
    # CASE 1: No reading history
    # ================================================================
    
    if not reading_log:
        response = (
            "You don't have any reading logged yet! "
            "Let's start fresh — what kind of book are you in the mood for?"
        )
        return {
            "handler_result": {"pattern": "no_history"},
            "response": response,
            "rag_context": rag_context,
        }
    
    # ================================================================
    # CASE 2: Long gap + dormant (slump recovery)
    # ================================================================
    
    if pattern == "slump_dormant":
        response = (
            "I see you haven't read in a while. That's totally normal! "
            "Here's what helps: start with something **short and forgiving** — "
            "something you can read in bursts without losing track. "
            "What sounds good: a quick adventure, or something cozy?"
        )
        return {
            "handler_result": {"pattern": "slump_dormant"},
            "response": response,
            "rag_context": rag_context,
        }
    
    # ================================================================
    # CASE 3: Pacing mismatch pattern
    # ================================================================
    
    if pattern == "pacing_mismatch":
        dnf_reasons = [
            log.get("dnf_reason", "")
            for log in reading_log
            if log.get("outcome") == "dnf" and log.get("dnf_reason")
        ]
        dnf_count = len(dnf_reasons)
        
        response = (
            f"I see a clear pattern: you've DNF'd **{dnf_count} books** all for the same reason — **pacing**. "
            f"This isn't about losing your taste, it's about finding the right **speed**. "
            f"You need breakneck narratives, not slow burns. "
            f"Want me to find something that moves?"
        )
        return {
            "handler_result": {"pattern": "pacing_mismatch", "dnf_count": dnf_count},
            "response": response,
            "rag_context": rag_context,
        }
    
    # ================================================================
    # CASE 4: General slump (multiple DNFs)
    # ================================================================
    
    if pattern == "general_slump":
        dnf_count = sum(1 for log in reading_log if log.get("outcome") == "dnf")
        finish_count = sum(1 for log in reading_log if log.get("outcome") == "finished")
        
        response = (
            f"You've been in a bit of a slump lately — "
            f"{finish_count} finishes, {dnf_count} DNFs. "
            f"The good news: your *taste* hasn't changed. "
            f"You just need to find the right **book at the right moment**. "
            f"Tell me: what did you love most about the books you *did* finish?"
        )
        return {
            "handler_result": {"pattern": "general_slump", "dnf_count": dnf_count, "finish_count": finish_count},
            "response": response,
            "rag_context": rag_context,
        }
    
    # ================================================================
    # CASE 5: Normal pattern (mostly finishing)
    # ================================================================
    
    finish_count = sum(1 for log in reading_log if log.get("outcome") == "finished")
    avg_rating = (
        sum(log.get("rating", 0) for log in reading_log if log.get("rating"))
        / len([log for log in reading_log if log.get("rating")])
        if any(log.get("rating") for log in reading_log)
        else 0
    )
    
    response = (
        f"You've finished **{finish_count} books** with an average rating of **{avg_rating:.1f}/5**. "
        f"Your reading is steady and you know what you like. "
        f"What would you like to explore next?"
    )
    
    return {
        "handler_result": {"pattern": "normal", "finish_count": finish_count, "avg_rating": avg_rating},
        "response": response,
        "rag_context": rag_context,
    }