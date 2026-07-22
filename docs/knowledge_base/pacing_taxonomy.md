# Pacing Taxonomy

A reference guide to how narrative pacing works, how to recognize it in a book, and how it maps to reader preference signals.

## What Pacing Actually Means

Pacing is the felt speed at which a story moves — not the length of the book, but how much *narrative distance* is covered per unit of reading time. A 900-page book can feel fast; a 200-page book can feel glacial. Pacing is produced by a combination of scene-to-summary ratio, sentence and paragraph length, chapter length, how often the point-of-view shifts, and how much time passes between beats of action.

Pacing is not a single dial from "slow" to "fast." It has at least three independent axes that a reader can respond to differently:

- **Plot pacing** — how quickly events happen and stakes escalate.
- **Prose pacing** — how quickly the sentences themselves read, independent of plot (short punchy sentences vs. long subordinate clauses).
- **Emotional pacing** — how quickly the story lets a reader sit with a feeling before moving on, vs. rushing through emotional beats to get to the next event.

A book can be fast in one axis and slow in another. A thriller with short chapters and constant plot movement (fast plot pacing) can still have dense, ornate prose (slow prose pacing). Recognizing which axis a reader responds to is more useful than a single global label.

## The Five-Category Pacing Scale

For this system, pacing is classified into five categories, used consistently across recommendations and reader modeling:

1. **Breakneck** — Constant forward motion. Minimal exposition, frequent cliffhangers, chapters under 10 pages, action or revelation in nearly every scene. Common in thrillers and some YA.
2. **Brisk** — Steady momentum with occasional pauses for character or world development. The reader rarely feels stalled, but the book isn't relentless. Most commercial fiction sits here.
3. **Measured** — Deliberate alternation between event and reflection. Scenes are given room to breathe. Common in literary fiction and character-driven fantasy.
4. **Slow-burn** — Extended build with delayed payoff, often deliberately withholding plot movement in favor of atmosphere, relationship development, or world texture. The payoff, when it comes, is usually large.
5. **Contemplative** — Pacing is not the point. Event may be minimal or entirely absent for long stretches; the reading experience is about interiority, prose, or mood rather than forward motion.

These categories are not a quality judgment. A contemplative novel executed well is not "slower than it should be" — contemplative is the intended mode.

## Signals That Indicate Pacing From Reader Behavior

Because the agent can't read the book itself in most cases, pacing preference has to be inferred from how a reader talks about their experience. Useful signals include:

- A DNF (did-not-finish) reason citing "nothing happens" or "I got bored" → signals a preference away from Measured/Slow-burn/Contemplative, toward Brisk/Breakneck.
- A DNF reason citing "too much happening, I couldn't keep up" or "exhausting" → signals the opposite, a preference toward Measured or Slow-burn.
- Explicit praise like "couldn't put it down" or "read it in one sitting" → strong signal for Breakneck/Brisk, but should be checked against *why* — was it plot momentum, or was it emotional investment in a slower book? These aren't the same signal.
- A reader completing and rating highly a book independently tagged Slow-burn, especially after DNFing a Breakneck book, is a stronger signal than either data point alone — this is where confidence scoring in Reader Modeling should combine two consistent low-confidence signals into one higher-confidence inference.

## Pacing Interacts With Mood, Not Just Genre

Pacing preference is frequently mistaken for a genre preference, but it usually isn't one. A reader who says "I don't like fantasy" after DNFing two slow-burn epic fantasy novels may actually have a pacing preference (they want Brisk regardless of genre), not a genre aversion. This is a key reason the Reader Model tracks pacing as its own confidence-scored dimension rather than folding it into genre tags — a reader's true preference might be "Brisk pacing across any genre," which genre tags alone can't express.

Pacing preference is also not fixed. A reader who normally prefers Brisk pacing may temporarily want Contemplative books during a period of stress or grief-adjacent reading (see the Reading Slump Recovery Guide), which is why pacing signals need the temporary-vs-durable classification applied just like any other inferred preference.

## Using Pacing in Recommendations

When the Planner is weighing a recommendation, pacing should be treated as a near-hard filter once confidence is high, not just one input among many blended into a general match score. A reader with high-confidence Breakneck preference will actively resent a Measured recommendation even if it matches every other stated interest (genre, trope, tone) — pacing mismatch is one of the most common causes of a DNF, and DNFs erode trust in the agent faster than a slightly-off genre match does.

When confidence is low or mixed (e.g., a new reader, or conflicting signals), pacing should be surfaced as a clarifying question rather than guessed — "Are you looking for something fast-moving, or something you can sink into slowly?" is a legitimate Planner move when the Reader Model doesn't yet support a confident inference.