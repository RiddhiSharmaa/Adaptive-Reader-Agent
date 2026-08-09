"""
LangGraph skeleton for the Adaptive Reader Agent.

Flow:
    receive_message -> classify_intent -> [unclear? -> clarify -> respond]
                                        -> planner -> route by intent
                                        -> (one of 4 handlers)
                                        -> reader_modeling_update (conditional)
                                        -> respond -> END

Handlers and reader_modeling are still stubs at this stage (see
app/intents/*.py and app/reader_modeling.py) — this file only wires the
shape of the graph so the routing logic can be tested before any handler
has real logic in it.
"""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

from app.intent_classifier import classify_intent as run_classifier


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    message: str
    user: dict                      # current persona from data/users.json
    intent: Optional[str]
    intent_confidence: Optional[float]
    needed_sources: Optional[dict]  # {"reader_model": bool, "rag": bool, "google_books": bool}
    reader_profile: Optional[dict]
    rag_context: Optional[str]
    book_data: Optional[list]
    handler_result: Optional[dict]  # raw result from whichever handler ran
    profile_updates: Optional[dict]  # what reader_modeling_update should write, if anything
    response: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_intent_node(state: AgentState) -> AgentState:
    result = run_classifier(state["message"])
    state["intent"] = result["intent"]
    state["intent_confidence"] = result["confidence"]
    return state


def planner_node(state: AgentState) -> AgentState:
    """
    Decide which of {reader_model, rag, google_books} this specific
    request needs, given its intent. Google Books is near-universal here
    since no book title/metadata is ever stored locally — only book_id is,
    so resolving any book reference means calling out to it.

    get_recommendation is the one intent where this isn't a fixed lookup —
    it's a sequential decision:

      1. Check the reader model FIRST, including peer_boost. If a peer
         reader with a similar profile has a proven positive outcome
         (finished + high rating) for a specific book_id, use that book_id
         directly. In this case Google Books is still called, but only to
         resolve that one known book_id into display metadata — no
         discovery search needed, because the reader model already
         supplied the candidate.
      2. Only if no strong peer match exists do we fall back to RAG
         (pacing/tone/trope expertise) + a Google Books discovery search
         to find new candidates from scratch.

    This is why `google_books_mode` exists below: "resolve" (cheap, one
    known book) vs "discover" (broader search) are different calls even
    though both hit the same tool.
    """
    intent = state["intent"]

    if intent == "get_recommendation":
        peer_match = _find_peer_match(state)
        if peer_match:
            state["needed_sources"] = {
                "reader_model": True,
                "rag": False,
                "google_books": True,
                "google_books_mode": "resolve",
                "peer_match_book_id": peer_match,
            }
        else:
            state["needed_sources"] = {
                "reader_model": True,
                "rag": True,
                "google_books": True,
                "google_books_mode": "discover",
            }
        return state

    decision_table = {
        "log_reading_outcome": {"reader_model": True, "rag": False, "google_books": True,
                                 "google_books_mode": "resolve"},
        "reading_insights": {"reader_model": True, "rag": True, "google_books": True,
                              "google_books_mode": "resolve"},
        "view_aggregate_reception": {"reader_model": False, "rag": False, "google_books": True,
                                      "google_books_mode": "resolve"},
    }

    state["needed_sources"] = decision_table.get(
        intent, {"reader_model": False, "rag": False, "google_books": False}
    )
    return state


def _find_peer_match(state: AgentState) -> Optional[str]:
    """
    Look across other readers' profiles for a peer with a similar
    preference profile who has a book in their reading_log with a proven
    positive outcome (finished, rating >= 4). Returns that book_id if
    found, else None.

    Delegates to reader_modeling.find_peer_match(), which does the actual
    weighted similarity comparison and proven-outcome check.
    """
    from app.reader_modeling import find_peer_match
    reader_id = state["user"]["id"]
    return find_peer_match(reader_id)


def clarify_node(state: AgentState) -> AgentState:
    """Handles 'unclear' intent — ask, don't guess."""
    state["response"] = (
        "I'm not sure exactly what you're looking for — could you tell me a "
        "bit more? For example, are you after a recommendation, logging how "
        "a book went, or asking about your own reading patterns?"
    )
    return state


# --- handler stubs — real logic lands in app/intents/*.py ---

def recommend_node(state: AgentState) -> AgentState:
    from app.intents.recommend import handle_get_recommendation
    result = handle_get_recommendation(state)
    state["handler_result"] = result["handler_result"]
    state["response"] = result["response"]
    return state


def log_outcome_node(state: AgentState) -> AgentState:
    from app.intents.log_outcome import handle_log_reading_outcome
    result = handle_log_reading_outcome(state)
    state["handler_result"] = result["handler_result"]
    state["response"] = result["response"]
    state["profile_updates"] = result.get("profile_updates")
    return state


def insights_node(state: AgentState) -> AgentState:
    from app.intents.insights import handle_reading_insights
    result = handle_reading_insights(state)
    state["handler_result"] = result["handler_result"]
    state["response"] = result["response"]
    state["profile_updates"] = result.get("profile_updates")
    return state


def author_stats_node(state: AgentState) -> AgentState:
    from app.intents.author_stats import handle_view_aggregate_reception
    result = handle_view_aggregate_reception(state)
    state["handler_result"] = result["handler_result"]
    state["response"] = result["response"]
    state["profile_updates"] = result.get("profile_updates")
    return state


def reader_modeling_update_node(state: AgentState) -> AgentState:
    """
    Shared mechanism — only writes if the handler produced profile_updates.
    reading_insights and view_aggregate_reception never produce these
    (read-only intents); get_recommendation currently doesn't either (it
    only reads reader_model + files nothing). log_reading_outcome is the
    primary source of pending_signals right now.

    Note: this only STAGES signals via process_feedback_signal — it never
    promotes anything into `preferences` directly. Promotion still requires
    the separate confidence-threshold + explicit-confirmation gate in
    reader_modeling.try_promote_signal(), which happens on a LATER turn
    once the reader has actually confirmed the signal.
    """
    updates = state.get("profile_updates")
    if not updates:
        return state

    from app.reader_modeling import process_feedback_signal
    reader_id = state["user"]["id"]

    for signal in updates.get("pending_signals", []):
        if signal is None:
            continue
        process_feedback_signal(
            reader_id=reader_id,
            signal_name=signal["signal_name"],
            confidence=signal["confidence"],
            source=signal["source"],
        )

    return state


def respond_node(state: AgentState) -> AgentState:
    if not state.get("response"):
        state["response"] = str(state.get("handler_result"))
    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_classify(state: AgentState) -> Literal["planner", "clarify"]:
    return "clarify" if state["intent"] == "unclear" else "planner"


def route_after_planner(state: AgentState) -> str:
    return state["intent"]  # matches node names below


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("planner", planner_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("get_recommendation", recommend_node)
    graph.add_node("log_reading_outcome", log_outcome_node)
    graph.add_node("reading_insights", insights_node)
    graph.add_node("view_aggregate_reception", author_stats_node)
    graph.add_node("reader_modeling_update", reader_modeling_update_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"planner": "planner", "clarify": "clarify"},
    )

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "get_recommendation": "get_recommendation",
            "log_reading_outcome": "log_reading_outcome",
            "reading_insights": "reading_insights",
            "view_aggregate_reception": "view_aggregate_reception",
        },
    )

    for handler_node in [
        "get_recommendation",
        "log_reading_outcome",
        "reading_insights",
        "view_aggregate_reception",
    ]:
        graph.add_edge(handler_node, "reader_modeling_update")

    graph.add_edge("reader_modeling_update", "respond")
    graph.add_edge("clarify", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


if __name__ == "__main__":
    # Smoke test — routing only, since handlers are still stubs.
    app_graph = build_graph()
    test_messages = [
        "What should I read next? Something breakneck-paced.",
        "I finished Winter's Ledger, loved it.",
        "How is my book doing with readers?",
        "hey",
    ]
    for msg in test_messages:
        result = app_graph.invoke({
            "message": msg,
            "user": {"user_id": "reader_01", "role": "reader"},
        })
        print(f"'{msg}' -> intent={result['intent']}, sources={result.get('needed_sources')}")
        print(f"   response: {result['response']}\n")