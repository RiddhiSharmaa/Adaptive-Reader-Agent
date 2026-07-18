"""
Planner: the core routing logic.

Given a classified intent and the current reader/author context,
decides which of the three knowledge sources to consult —
Reader Model, RAG knowledge base, Google Books API — and in what
combination. Checks Reader Model confidence scores before deciding
whether to recommend directly or ask a clarifying question first.
"""