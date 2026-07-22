# Genre Taxonomy

A reference guide to genre categories used in this system, and — critically — why genre is treated as the least informative signal available to the agent rather than the primary one.

## Why Genre Is Last, Not First

Every other document in this knowledge base makes some version of the same argument: pacing, trope, and mood/tone are more predictive of reader satisfaction than genre alone, because two books sharing a genre tag can differ wildly on all three of those axes, while two books from different genres can align closely on all three. This document exists to give the agent a working genre vocabulary — genre is still a useful coarse filter, especially for live metadata lookups against Google Books and Open Library, which are organized by genre and subject tags — but it should never be the primary basis for a recommendation once pacing, trope, and mood/tone signals are available.

The practical rule: genre is useful for narrowing a large search space (e.g., when calling `search_books()` against a live API) and for a reader's own sense of identity as "a mystery reader" or "a fantasy reader," which is worth acknowledging conversationally. It is not useful as the main matching variable in the Reader Model, because it collapses too much real variation.

## Core Genre Categories

For this system, genre is tracked at a broad level, with the understanding that most real books blend more than one:

1. **Fantasy** — includes secondary-world fantasy, urban fantasy, and fairy-tale-adjacent fiction; unified by non-realistic world rules that matter to the plot.
2. **Science fiction** — includes hard SF, space opera, and near-future speculative fiction; unified by extrapolation from technology, science, or society rather than magic.
3. **Mystery / crime** — includes classic whodunits, procedurals, and noir; unified by a central puzzle or crime driving the plot.
4. **Thriller** — includes psychological thrillers, espionage, and technothrillers; unified by sustained tension and personal danger to the protagonist, often with a ticking-clock structure.
5. **Literary fiction** — character- and prose-driven work where the appeal is often more about interiority and craft than plot mechanics; can overlap heavily with any of the above.
6. **Historical fiction** — grounded in a real past era, where the historical setting itself is a major driver of the reading experience.
7. **Horror** — unified by an intent to frighten or unsettle, ranging from atmospheric dread to explicit threat.
8. **Nonfiction narrative** — includes memoir, narrative history, and long-form journalism, structured with story techniques even though the content is factual.

This list is intentionally broad rather than exhaustive — subgenres and blends (a historical mystery, a literary science fiction novel) are common and should be handled by tagging a book with more than one category rather than forcing a single label.

## Genre Blending Is the Norm, Not the Exception

Most contemporary books don't sit cleanly in one category, and the agent should expect and handle blending rather than treating it as an edge case. A locked-room mystery set on a generation ship is simultaneously mystery and science fiction; a historical novel built around a central crime is simultaneously historical fiction and mystery. When live metadata returns multiple genre or subject tags for a title, the agent should retain all of them rather than picking one to simplify — discarding secondary genre tags throws away information that can matter for a reader whose real preference is trope- or mood-driven across genre lines (see the Trope Encyclopedia's point on trope preference cutting across stated genre).

## Genre and the Live Metadata Layer

Because genre metadata comes from Google Books or Open Library rather than from a local catalog, genre tags returned by these APIs are the vocabulary used going forward, but they should not be assumed to be clean or consistent — the same book can be tagged differently across sources, and subject tags from these APIs are often more granular or more inconsistent than the eight core categories above. When the agent needs to reason about genre, it should map incoming API tags loosely onto the categories in this document rather than trying to preserve every raw subject tag as a distinct category the Reader Model has to track.

## Using Genre Conversationally

Even though genre carries little weight in the actual matching logic, it still matters in how the agent talks to a reader. A reader who identifies as "a fantasy reader" has expressed something about their reading identity worth acknowledging, even if the agent's internal reasoning is actually driven by that reader's pacing and mood preferences. Dismissing genre entirely in conversation, even while correctly deprioritizing it internally, would feel tone-deaf to a reader — the agent should use genre as a natural conversational anchor while quietly doing the real matching work on the axes covered by the other six documents in this knowledge base.