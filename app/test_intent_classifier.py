"""
Standalone test for the intent classifier — run this before wiring the
classifier into LangGraph. Cheaper to catch misclassification here than
inside a graph with 4 handlers attached.

Usage:
    python -m app.test_intent_classifier
"""

from app.intent_classifier import classify_intent

TEST_CASES = [
    # --- clear cases, one per intent ---
    ("What should I read next? Something breakneck-paced.", "get_recommendation"),
    ("I just finished Winter's Ledger, I loved it, 5 stars.", "log_reading_outcome"),
    ("Why do I keep giving up on books halfway through?", "reading_insights"),
    ("How is my book doing with readers so far?", "view_aggregate_reception"),

    # --- rephrasings of the same intents, less obvious ---
    ("I'm in the mood for something dark and twisty, any ideas?", "get_recommendation"),
    ("Couldn't get through the last one, too slow to start. DNF.", "log_reading_outcome"),
    ("Have my tastes changed since last year?", "reading_insights"),
    ("What's my average rating from readers?", "view_aggregate_reception"),

    # --- messy / ambiguous cases -> should land on "unclear" ---
    ("tell me about it", "unclear"),                     # no antecedent
    ("what's the weather like today", "unclear"),          # off-topic
    ("hey", "unclear"),                                    # too vague to act on
    ("can you help me", "unclear"),                        # too vague
    ("I read a book once", "unclear"),                     # not an actionable outcome report
    ("what do you think about my book and also what should I read", "unclear"),
    # ^ two intents at once (author stats + recommendation) — should NOT
    #   confidently pick one; should surface as unclear so the handler
    #   can ask which one they mean
]


def run_tests():
    correct = 0
    for message, expected in TEST_CASES:
        result = classify_intent(message)
        got = result["intent"]
        status = "PASS" if got == expected else "FAIL"
        if got == expected:
            correct += 1
        print(f"[{status}] \"{message}\"")
        print(f"       expected={expected}  got={got}  confidence={result['confidence']}")
        print(f"       reasoning: {result['reasoning']}")
        print()

    print(f"{correct}/{len(TEST_CASES)} correct")


if __name__ == "__main__":
    run_tests()