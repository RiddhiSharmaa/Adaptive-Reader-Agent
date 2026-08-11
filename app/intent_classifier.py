"""
Intent classifier for the Adaptive Reader Agent.

Deliberately standalone right now — no LangGraph, no state graph. The
classifier is the "nervous system" of the agent: get this wrong and every
downstream handler acts on a bad signal. Test it in isolation against messy
inputs before wiring it into planner.py.

Five categories: the 4 real intents, plus "unclear" for anything that
doesn't confidently map to one. There's no separate "off_topic" bucket —
off-topic and ambiguous both get routed to a clarifying question, so
"unclear" covers both rather than forcing the model to pick between two
similar buckets.
"""

import json
import os
from anthropic import Anthropic

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
)

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

VALID_INTENTS = {
    "get_recommendation",
    "log_reading_outcome",
    "reading_insights",
    "view_aggregate_reception",
    "unclear",
}

CLASSIFIER_SYSTEM_PROMPT = """You are the intent classifier for a reading companion agent.

Classify the user's message into exactly one of these categories:

- get_recommendation: the reader EXPLICITLY asks for a book suggestion or 
  describes a SPECIFIC mood/preference they want matched to a book.
  MUST include explicit intent words like "recommend", "suggest", "what should 
  I read", "something with X", "looking for a book that", or specific 
  preference descriptors like "fast-paced", "dark", "funny", "sad", etc.
  Examples: "what should I read next", "I want something fast-paced and dark",
  "recommend me something like the last book I finished", "looking for a thriller"
  
  COUNTEREXAMPLE: "Books have been feeling boring" is NOT get_recommendation 
  (it's a vague complaint, not an explicit request) — classify as unclear.

- log_reading_outcome: the reader is reporting what happened with a book
  they read or attempted — finished it, gave up on it, rated it, or is
  giving feedback on a specific book.
  Examples: "I finished Winter's Ledger, loved it", "I couldn't get through
  that last one, too slow", "give it a 4/5"

- reading_insights: the reader is asking about their OWN reading patterns,
  history, or preferences over time — not asking for a new recommendation.
  Examples: "why do I keep DNFing fantasy books", "what genres do I finish
  the most", "have my preferences changed lately"

- view_aggregate_reception: an author is asking about how readers are
  receiving THEIR book — never an individual reader's specific feedback.
  Examples: "how is my book doing", "what's my completion rate",
  "why are people DNFing my book"

- unclear: the message doesn't confidently fit one of the above — this
  includes vague complaints ("I'm bored", "nothing sounds good"), off-topic
  messages, ambiguous requests that could mean several things, or messages 
  missing the info needed to classify (e.g. "tell me about it" with no 
  prior context). When in doubt, classify as unclear.

Respond ONLY with a JSON object, no other text:
{"intent": "<one of the five category names above>", "confidence": <float 0-1>, "reasoning": "<one sentence>"}
"""


def classify_intent(message: str) -> dict:
    """
    Classify a single user message into one of the 5 categories above.

    Returns:
        {
          "intent": str,        # one of VALID_INTENTS
          "confidence": float,  # 0-1, model's self-reported confidence
          "reasoning": str      # one sentence, useful for debugging/demo
        }

    Falls back to {"intent": "unclear", "confidence": 0.0, ...} if the
    model's response can't be parsed as valid JSON or names an intent
    outside VALID_INTENTS — fail closed into a clarifying question rather
    than letting a malformed response silently pick a random handler.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=CLASSIFIER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
        if result.get("intent") not in VALID_INTENTS:
            raise ValueError(f"Unrecognized intent: {result.get('intent')}")
        return result
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[classify_intent DEBUG] Failed to parse. Raw model output was:\n{raw_text!r}\nError: {e}")
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "reasoning": f"Failed to parse classifier response: {e}",
        }