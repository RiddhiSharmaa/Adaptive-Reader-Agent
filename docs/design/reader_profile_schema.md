# Reader Profile Schema — Design Rationale

## Why this shape

The reader profile is the single source of truth that every intent handler
reads from and (in some cases) writes to. Getting its shape right up front
matters more than any one handler, because the classifier, planner, and all
four handlers depend on it.

```json
{
  "reader_id": "reader_01",
  "preferences": {
    "pacing": { "value": "breakneck", "personal_confidence": 0.8, "peer_boost": 0.0 }
  },
  "reading_log": [
    {
      "log_id": "log_001",
      "book_id": "b001",
      "outcome": "finished",
      "rating": 5,
      "dnf_reason": null,
      "timestamp": "2026-06-10T14:00:00Z",
      "signals_extracted": ["pacing", "trope"]
    }
  ],
  "signals_pending_confirmation": [
    {
      "signal": "prefers_standalone",
      "source": "free_text_feedback",
      "confidence": 0.4,
      "temporary": true,
      "timestamp": "2026-07-20T11:00:00Z"
    }
  ]
}
```

## Design decisions

### 1. `preferences` is open-ended, not a fixed key set

Preferences are not hardcoded to `pacing` / `tone` / `trope`. Any signal key
can appear as long as it follows the shape `{ value, personal_confidence,
peer_boost }`. This matters because genre-neutral reader modeling means new
signal types (POV style, series preference, narrative structure) should be
addable without a schema migration — the agent's model of "what a preference
is" stays generic, not tied to a fixed taxonomy decided in advance.

### 2. Promotion from pending to confirmed requires both confidence AND explicit confirmation

An entry in `signals_pending_confirmation` is only promoted into `preferences`
when:
- **(a)** its confidence score crosses a threshold, AND
- **(b)** the reader has explicitly confirmed it (e.g. responded positively
  to a check-in question, or a recommendation surfaced it and got positive
  feedback).

Confidence alone does not auto-promote a signal — a single strongly-worded
comment shouldn't calcify into a permanent stated preference. Confirmation
alone does not promote it either — a reader casually agreeing to a
low-evidence guess shouldn't outweigh a lack of supporting data. Requiring
both prevents the agent from overfitting to one-off remarks (e.g. "hated the
ending" should not become "hates all dark endings forever") while also
avoiding silently locking in an unconfirmed guess.

### 3. `reading_log` entries carry `signals_extracted`

Each log entry records which preference keys it contributed to updating.
This gives the agent (and a TA, or Riddhi during a demo) a traceable path
from a specific reading outcome back to a specific belief update — "why does
the agent think I like breakneck pacing?" has a concrete answer: log_001 and
log_002.

## What this schema deliberately does not do

- It does not store genre as a matching field. Genre collapses too much
  variation to be useful as a primary preference axis (see the RAG knowledge
  base's Genre Taxonomy doc for the full reasoning).
- It does not store book metadata. Book records come live from `search_books()`
  (stubbed now, real Google Books API call in Part 3) — the reader profile
  only stores the reader's *reaction* to a book (`book_id`, `outcome`,
  `rating`), never a duplicated copy of the book's own data.