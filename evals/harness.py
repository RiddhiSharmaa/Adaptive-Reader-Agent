"""
Adaptive Reader Agent Evaluation Harness.

Run from project root:

    python evals/harness.py

Evaluates four dimensions:

1. intent_classification
2. tool_selection
3. recommendation_relevance
4. rag_faithfulness

Test case schema:

{
    "id": "...",
    "input": "...",
    "expected": "...",
    "category": "...",
    "dimension": "..."
}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. "
        f"Expected it in: {PROJECT_ROOT / '.env'}"
    )

EVALS_DIR = PROJECT_ROOT / "evals"

CASES_FILE = EVALS_DIR / "test_cases.json"
RESULTS_FILE = EVALS_DIR / "evaluation_results.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT AGENT AND RAG
# ============================================================

from app.graph import build_graph
from app import rag_index

from evals.evaluators import (
    evaluate_test_case,
    extract_expected_tools,
)


# ============================================================
# TEST CASE LOADING
# ============================================================

def load_test_cases() -> list[dict]:
    """Load and validate evaluation test cases."""

    if not CASES_FILE.exists():
        raise FileNotFoundError(
            f"Test cases file not found:\n{CASES_FILE}"
        )

    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not isinstance(cases, list):
        raise ValueError(
            "test_cases.json must contain a JSON list."
        )

    required_fields = {
        "id",
        "input",
        "expected",
        "category",
        "dimension",
    }

    valid_dimensions = {
        "intent_classification",
        "tool_selection",
        "recommendation_relevance",
        "rag_faithfulness",
    }

    seen_ids: set[str] = set()

    for index, case in enumerate(cases, start=1):

        if not isinstance(case, dict):
            raise ValueError(
                f"Test case #{index} must be a JSON object."
            )

        missing = required_fields - set(case.keys())

        if missing:
            raise ValueError(
                f"Test case #{index} is missing fields: "
                f"{sorted(missing)}"
            )

        test_id = case["id"]

        if test_id in seen_ids:
            raise ValueError(
                f"Duplicate test case id: {test_id}"
            )

        seen_ids.add(test_id)

        if case["dimension"] not in valid_dimensions:
            raise ValueError(
                f"Test case {test_id} has invalid dimension: "
                f"{case['dimension']}"
            )

    return cases


# ============================================================
# FORMAL LANGCHAIN / LANGGRAPH TOOL CALL EXTRACTION
# ============================================================

def extract_tool_calls(result: Any) -> list[dict]:
    """
    Extract formal LangChain/LangGraph tool calls.

    Direct Python application operations are captured separately
    by trace_agent_execution().
    """

    found: list[dict] = []
    visited: set[int] = set()

    def add_call(call: dict) -> None:
        name = (
            call.get("name")
            or call.get("tool")
            or call.get("tool_name")
        )

        if not name:
            return

        found.append(
            {
                "name": name,
                "args": call.get("args", {}),
            }
        )

    def inspect(value: Any) -> None:

        if value is None:
            return

        object_id = id(value)

        if object_id in visited:
            return

        visited.add(object_id)

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(value, dict):

            calls = value.get("tool_calls")

            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        add_call(call)

            for child in value.values():
                inspect(child)

            return

        # ----------------------------------------------------
        # LangChain message
        # ----------------------------------------------------

        if hasattr(value, "tool_calls"):

            try:
                calls = value.tool_calls
            except Exception:
                calls = []

            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict):
                        add_call(call)

        # ----------------------------------------------------
        # Lists / tuples
        # ----------------------------------------------------

        if isinstance(value, (list, tuple)):

            for child in value:
                inspect(child)

    inspect(result)

    return deduplicate_tool_calls(found)


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_tool_calls(
    calls: list[dict],
) -> list[dict]:
    """
    Remove duplicate calls while preserving order.

    Tool selection is evaluated at the operation level, not by
    number of times an operation was called.
    """

    unique: list[dict] = []
    seen: set[tuple] = set()

    for call in calls:

        name = call.get("name", "")
        args = call.get("args", {})

        try:
            serialized_args = json.dumps(
                args,
                sort_keys=True,
                default=str,
            )
        except Exception:
            serialized_args = str(args)

        key = (
            name,
            serialized_args,
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(call)

    return unique


# ============================================================
# DIRECT APPLICATION OPERATION TRACING
# ============================================================

"""
These are the user-facing/application-level operations that count
toward the tool-selection metric.

Internal helpers such as:

    _resolve_book
    _extract_filters
    _query_rag
    _rank_candidates
    _retrieve_context
    _find_peer_match

are NOT included.

Important:

find_peer_match() is included because it is an explicitly evaluated
operation in G1_TC_001.

process_feedback_signal() is included because it is an explicitly
evaluated operation in G1_TC_003.

The fact that find_peer_match() internally calls get_reader_profile()
does not make get_reader_profile() an independent agent decision.
However, for the current tool-selection metric we evaluate the
observable operation set, so duplicate/nested operations are simply
deduplicated.
"""

TRACEABLE_FUNCTIONS = {
    "get_reader_profile",
    "update_reader_profile",
    "log_book",
    "search_books",
    "get_book_by_id",
    "find_peer_match",
    "process_feedback_signal",
}


def trace_agent_execution(
    graph,
    initial_state: dict,
) -> tuple[Any, list[dict]]:
    """
    Execute the graph while tracing evaluated application operations.

    Only functions explicitly listed in TRACEABLE_FUNCTIONS are
    captured.

    Internal implementation helpers are intentionally ignored.
    """

    calls: list[dict] = []

    previous_trace = sys.gettrace()

    def tracer(frame, event, arg):

        if event != "call":
            return tracer

        function_name = frame.f_code.co_name

        if function_name not in TRACEABLE_FUNCTIONS:
            return tracer

        module_name = frame.f_globals.get("__name__", "")

        # Only trace application code.
        if not module_name.startswith("app."):
            return tracer

        calls.append(
            {
                "name": function_name,
                "module": module_name,
                "args": {},
            }
        )

        return tracer

    try:
        sys.settrace(tracer)

        result = graph.invoke(initial_state)

    finally:
        sys.settrace(previous_trace)

    return result, deduplicate_tool_calls(calls)


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    graph,
    message: str,
) -> dict:
    """Run one message through the Adaptive Reader graph."""

    initial_state = {
        "message": message,

        "user": {
            "id": "reader_01",
            "role": "reader",
        },

        "intent": None,
        "intent_confidence": None,

        "needed_sources": None,

        "reader_profile": None,
        "rag_context": None,
        "book_data": None,

        "handler_result": None,
        "profile_updates": None,

        "response": None,

        # Evaluation fields.
        "messages": [],
        "tool_calls": [],
    }

    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    result, traced_calls = trace_agent_execution(
        graph,
        initial_state,
    )

    if result is None:
        result = {}

    # --------------------------------------------------------
    # Extract formal LangGraph tool calls
    # --------------------------------------------------------

    formal_tool_calls = extract_tool_calls(result)

    # --------------------------------------------------------
    # Merge formal tools + application operations
    # --------------------------------------------------------

    all_tool_calls = deduplicate_tool_calls(
        formal_tool_calls + traced_calls
    )

    # --------------------------------------------------------
    # Normalize graph output
    # --------------------------------------------------------

    if isinstance(result, dict):

        return {
            "intent": result.get("intent"),

            "response": result.get(
                "response",
                "",
            ),

            "tool_calls": all_tool_calls,

            # RAG evaluation data.
            "rag_context": result.get(
                "rag_context"
            ),

            "needed_sources": result.get(
                "needed_sources"
            ),

            # Recommendation evaluation data.
            "reader_profile": result.get(
                "reader_profile"
            ),

            "book_data": result.get(
                "book_data"
            ),

            # Other agent state.
            "handler_result": result.get(
                "handler_result"
            ),

            "profile_updates": result.get(
                "profile_updates"
            ),
        }

    return {
        "intent": None,
        "response": "",
        "tool_calls": all_tool_calls,
        "rag_context": None,
        "needed_sources": None,
        "reader_profile": None,
        "book_data": None,
        "handler_result": None,
        "profile_updates": None,
    }


# ============================================================
# PRINT TOOL CALLS
# ============================================================

def print_tool_calls(
    tool_calls: list[dict],
) -> None:
    """Print actual evaluated application operations."""

    if not tool_calls:
        print("Actual tool calls: NONE")
        return

    print("Actual tool calls:")

    for call in tool_calls:

        name = call.get(
            "name",
            "unknown",
        )

        module = call.get("module")

        if module:
            print(
                f"  - {name} ({module})"
            )
        else:
            print(
                f"  - {name}"
            )


# ============================================================
# PRINT EXPECTED TOOLS
# ============================================================

def print_expected_tools(
    test_case: dict,
) -> None:
    """Print evaluator-derived expected tools."""

    if test_case["dimension"] != "tool_selection":
        return

    expected_tools = extract_expected_tools(
        test_case
    )

    if not expected_tools:
        print("Expected tools: NONE")
        return

    print("Expected tools:")

    for tool in expected_tools:
        print(f"  - {tool}")


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    graph,
    test_case: dict,
    index: int,
    total: int,
) -> dict:

    test_id = test_case["id"]
    dimension = test_case["dimension"]
    message = test_case["input"]

    print()
    print("=" * 70)
    print(f"TEST {index}/{total}")
    print()

    print(
        f"[{test_id}] {dimension}"
    )

    print(
        f"Category: {test_case['category']}"
    )

    print(
        f"Input: {message}"
    )

    # --------------------------------------------------------
    # Expected tools
    # --------------------------------------------------------

    if dimension == "tool_selection":

        print()
        print_expected_tools(test_case)

    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    try:

        actual = run_agent(
            graph,
            message,
        )

    except Exception as exc:

        print(
            f"AGENT ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            **test_case,
            "actual": {
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            "evaluation": {
                "score": None,
                "reason": (
                    "Agent execution failed before "
                    "evaluation."
                ),
            },
        }

    # --------------------------------------------------------
    # Print intent
    # --------------------------------------------------------

    if dimension == "intent_classification":

        print()

        print(
            f"Actual intent: "
            f"{actual.get('intent')}"
        )

    # --------------------------------------------------------
    # Print tools
    # --------------------------------------------------------

    if dimension == "tool_selection":

        print()

        print_tool_calls(
            actual.get(
                "tool_calls",
                [],
            )
        )

    # --------------------------------------------------------
    # Print response
    # --------------------------------------------------------

    if dimension in {
        "recommendation_relevance",
        "rag_faithfulness",
    }:

        print()
        print("Agent response:")

        print(
            actual.get(
                "response",
                "",
            )
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    try:

        evaluation = evaluate_test_case(
            test_case,
            actual,
        )

    except Exception as exc:

        print(
            f"EVALUATOR ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        evaluation = {
            "score": None,
            "reason": (
                f"Evaluator failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()

    if evaluation["score"] is None:

        print("Score: ERROR")

    else:

        print(
            f"Score: "
            f"{evaluation['score']}/5"
        )

    print(
        f"Reason: "
        f"{evaluation['reason']}"
    )

    return {
        **test_case,
        "actual": actual,
        "evaluation": evaluation,
    }


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: list[dict],
) -> None:
    """Print per-dimension and overall evaluation summary."""

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    dimensions = [
        "intent_classification",
        "tool_selection",
        "recommendation_relevance",
        "rag_faithfulness",
    ]

    for dimension in dimensions:

        cases = [
            result
            for result in results
            if result["dimension"] == dimension
        ]

        if not cases:
            continue

        valid_scores = [
            result["evaluation"]["score"]
            for result in cases
            if isinstance(
                result["evaluation"].get("score"),
                int,
            )
        ]

        print()
        print(dimension)
        print("-" * 50)

        print(
            f"Cases: {len(cases)}"
        )

        if not valid_scores:

            print("Average: N/A")
            continue

        average = (
            sum(valid_scores)
            / len(valid_scores)
        )

        fully_correct = sum(
            score == 5
            for score in valid_scores
        )

        print(
            f"Average: {average:.2f}/5"
        )

        print(
            f"Score 5: "
            f"{fully_correct}/"
            f"{len(valid_scores)}"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    all_scores = [
        result["evaluation"]["score"]
        for result in results
        if isinstance(
            result["evaluation"].get("score"),
            int,
        )
    ]

    print()

    if all_scores:

        print(
            "Overall average: "
            f"{sum(all_scores) / len(all_scores):.2f}/5"
        )

        print(
            "Overall scored cases: "
            f"{len(all_scores)}/{len(results)}"
        )

    else:

        print("Overall average: N/A")


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: list[dict],
) -> None:
    """Save detailed evaluation results as JSON."""

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()

    print(
        "ADAPTIVE READER AGENT — "
        "EVALUATION HARNESS"
    )

    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Test cases:   {CASES_FILE}"
    )

    print(
        f"Results:      {RESULTS_FILE}"
    )

    # ============================================================
    # INITIALIZE RAG INDEX
    # ============================================================

    print()
    print("Initializing RAG index...")

    rag_index.index_knowledge_base(chunk_fn=rag_index.structure_aware_chunk)

    # ============================================================
    # LOAD TEST CASES
    # ============================================================

    test_cases = load_test_cases()

    print()

    print(
        f"Loaded {len(test_cases)} test cases."
    )

    # ============================================================
    # BUILD GRAPH
    # ============================================================

    print()

    print("Building agent graph...")

    graph = build_graph()

    # ============================================================
    # RUN CASES
    # ============================================================

    results: list[dict] = []

    total = len(test_cases)

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        result = run_case(
            graph,
            test_case,
            index,
            total,
        )

        results.append(result)

    # ============================================================
    # SUMMARY
    # ============================================================

    print_summary(results)

    # ============================================================
    # SAVE
    # ============================================================

    save_results(results)

    print()
    print("Detailed results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()