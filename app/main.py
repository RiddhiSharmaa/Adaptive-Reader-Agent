"""
FastAPI entrypoint for the Adaptive Reader Agent.

Wires up the chat endpoint, loads the LangGraph planner graph,
and exposes the Reader/Author persona switcher to the UI.
"""

from fastapi import FastAPI

app = FastAPI(title="Adaptive Reader Agent")


@app.get("/health")
def health():
    return {"status": "ok"}