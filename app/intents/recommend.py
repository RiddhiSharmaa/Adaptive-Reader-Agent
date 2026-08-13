"""
Handler for get_recommendation with RAG integration.
"""

from app.reader_modeling import find_peer_match, process_feedback_signal
from app.tools import get_reader_profile, search_books, get_book_by_id
from app.rag_index import knowledge_collection
from anthropic import Anthropic
import os
import json

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def _query_rag(pacing: str, mood: str, trope: str) -> str:
    """Query RAG knowledge base."""
    query = f"{pacing} pacing {mood} mood {trope} reader preferences"
    
    try:
        results = knowledge_collection.query(
            query_texts=[query],
            n_results=3
        )
        
        if results and results.get("documents"):
            docs = results["documents"][0]
            return "\n".join(docs) if docs else ""
    except Exception as e:
        print(f"[RAG ERROR] {e}")
    
    return ""


def _extract_pacing_mood_from_text(user_message: str) -> tuple[str, str]:
    """Parse free-text into pacing + mood."""
    prompt = f"""Extract reading preferences from:
"{user_message}"

Return ONLY JSON:
{{"pacing": "<breakneck|brisk|measured|slow-burn|contemplative>", "mood": "<value>"}}

Return ONLY JSON, no other text."""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        result = json.loads(text)
        return result.get("pacing", "brisk"), result.get("mood", "engaging")
    except Exception as e:
        print(f"[SIGNAL EXTRACTION ERROR] {e}")
        return "brisk", "engaging"


def _is_valid_fiction(book: dict) -> bool:
    """
    Validate that book is actual fiction/novel, not a technical doc.
    Filters out: patents, government docs, instruction manuals, etc.
    """
    
    # Get categories and description
    categories = book.get("categories", [])
    description = book.get("description", "").lower()
    title = book.get("title", "").lower()
    
    # ❌ EXCLUDE these
    bad_keywords = [
        "patent", "gazette", "trademark", "government", "manual",
        "instruction", "technical", "reference", "dictionary",
        "encyclopedia", "directory", "index", "database",
        "companion", "guide", "introduction", "primer", "handbook",  # ⭐ ADDED
        "critical study", "essay", "analysis", "commentary"   
    ]
    
    for keyword in bad_keywords:
        if keyword in title or keyword in description:
            return False
    
    # ✅ ACCEPT these categories
    good_categories = [
        "Fiction", "Novel", "Fantasy", "Mystery", "Thriller",
        "Science Fiction", "Romance", "Historical", "Literary",
        "Adventure", "Classics"
    ]
    
    # If book has good category tags, accept it
    for cat in categories:
        if any(good in cat for good in good_categories):
            return True
    
    # If description sounds like fiction, accept it
    fiction_keywords = [
        "protagonist", "story", "novel", "narrative", "tale",
        "character", "adventure", "mystery", "discovers",
        "journey", "quest", "follows"
    ]
    
    for keyword in fiction_keywords:
        if keyword in description:
            return True
    
    # Default: reject (safer than recommending random doc)
    return False


def _build_query(pacing: str, mood: str, trope: str, context: str = "") -> str:
    """
    Build improved search query with fiction context.
    ⭐ KEY: Add "novel" or "classic" or "fiction" to query
    """
    parts = []
    
    # Add pacing + mood
    if pacing:
        parts.append(pacing)
    if mood:
        parts.append(mood)
    if trope:
        parts.append(trope)
    
    # Add context hints
    if context:
        parts.append(context)
    
    # Always add fiction/novel keyword
    parts.append("novel")
    
    return " ".join(parts) if parts else "engaging fiction novel"


def handle_get_recommendation(state: dict) -> dict:
    reader_id = state["user"]["id"]
    message = state["message"].strip()
    profile = get_reader_profile(reader_id)
    
    # ================================================================
    # CASE 1: PEER MATCH (unchanged)
    # ================================================================
    peer_book_id = find_peer_match(reader_id, similarity_threshold=0.6)
    if peer_book_id:
        book = get_book_by_id(peer_book_id)
        if book and _is_valid_fiction(book):
            rag_context = _query_rag("unknown", "unknown", "unknown")
            response = (
                f"I found a match: **{book['title']}** by {book['author']}. "
                f"Other readers like you loved it."
            )
            return {
                "handler_result": {"mode": "resolve", "book": book},
                "response": response,
                "rag_context": rag_context,
            }
    
    # ================================================================
    # CASE 2: CLARIFICATION ANSWER DETECTED (moved BEFORE stored prefs)
    # ================================================================
    
    pacing_keywords = ["fast", "quick", "breakneck", "brisk", "slow", "measured", "contemplative"]
    mood_keywords = ["gripping", "tense", "cozy", "reflective", "dark", "hopeful", "emotional", "adventure"]
    author_keywords = ["same author", "different author", "author", "different world", "standalone"]
    
    has_pacing = any(kw in message.lower() for kw in pacing_keywords)
    has_mood = any(kw in message.lower() for kw in mood_keywords)
    has_author = any(kw in message.lower() for kw in author_keywords)  # ⭐ ADD THIS
    
    context = ""
    if "classic" in message.lower():
        context = "classic"
    if "1900" in message or "19th" in message or "sci-fi" in message.lower():
        context = "sci-fi" if "sci-fi" in message.lower() else "historical"
    
    if "series" in message.lower() or "book " in message.lower():
        rag_context = _query_rag("unknown", "unknown", "")
        
        response = (
            "I hear you — series burnout is real! A few options: "
            "**Do you want to stay with the author (their standalone works), "
            "or try a completely fresh author?**"
        )
        
        state["series_context"] = True
        
        return {
            "handler_result": {"mode": "clarify", "type": "series"},
            "response": response,
            "rag_context": rag_context,
        }

    if has_pacing or has_mood or has_author:
        pacing, mood = _extract_pacing_mood_from_text(message)
        
        rag_context = _query_rag(pacing, mood, "")
        
        # ⭐ BUILD QUERY WITH CURRENT MESSAGE, NOT STORED PREFS
        query = _build_query(pacing, mood, "", context)
        candidates = search_books(query, max_results=15)
        
        valid_candidates = [b for b in candidates if _is_valid_fiction(b)]
        
        if valid_candidates:
            top = valid_candidates[0]
            response = (
                f"Perfect! For {pacing} narratives with a {mood} feel, "
                f"I recommend **{top['title']}** by {top['author']}."
            )
            
            process_feedback_signal(
                reader_id=reader_id,
                signal_name=f"prefers_{pacing}_pacing",
                confidence=0.7,
                source="clarification_answer",
                temporary=False,
            )
            
            return {
                "handler_result": {"mode": "discover", "candidates": valid_candidates},
                "response": response,
                "rag_context": rag_context,
                "profile_updates": {
                    "pending_signals": [
                        {"signal_name": f"prefers_{pacing}_pacing", "confidence": 0.7, "source": "clarification_answer"},
                    ]
                },
            }
        else:
            response = (
                f"I'm having trouble finding the perfect {pacing}, {mood} match "
                f"in the current database. Could you give me another hint?"
            )
            return {
                "handler_result": {"mode": "discover", "candidates": []},
                "response": response,
                "rag_context": rag_context,
            }
    
    # ================================================================
    # CASE 3: PROFILE HAS PREFERENCES (now AFTER clarification check)
    # ================================================================
    
    preferences = profile.get("preferences", {})
    
    if preferences and len(preferences) >= 2:
        pacing = preferences.get("pacing", {}).get("value", "brisk")
        mood = preferences.get("mood", {}).get("value", "engaging")
        trope = preferences.get("trope", {}).get("value", "")
        
        rag_context = _query_rag(pacing, mood, trope)
        
        query = _build_query(pacing, mood, trope)
        candidates = search_books(query, max_results=15)
        
        valid_candidates = [b for b in candidates if _is_valid_fiction(b)]
        
        if valid_candidates:
            top = valid_candidates[0]
            response = (
                f"Based on your history, try **{top['title']}** by {top['author']}. "
                f"It matches the {pacing} pacing and {mood} feel you've enjoyed."
            )
            return {
                "handler_result": {"mode": "discover", "candidates": valid_candidates},
                "response": response,
                "rag_context": rag_context,
            }

        
    
    # ================================================================
    # CASE 4: DEFAULT CLARIFICATION
    # ================================================================
    
    rag_context = _query_rag("unknown", "unknown", "unknown")
    
    response = (
        "I'd love to help you find something great. "
        "Could you tell me: **Are you looking for something fast-paced and gripping, "
        "or something slower and more reflective?**"
    )
    
    return {
        "handler_result": {"mode": "clarify"},
        "response": response,
        "rag_context": rag_context,
    }