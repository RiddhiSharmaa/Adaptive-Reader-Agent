# Mood & Tone Guide

A reference guide to distinguishing mood from tone, cataloguing common categories of each, and mapping reader reactions to usable signals.

## Mood vs. Tone: Two Different Things Often Confused

Mood and tone are frequently used interchangeably, but they describe different things and should be tracked separately in the Reader Model.

- **Tone** is the author's implied attitude toward the material — is the book earnest, satirical, wry, bleak, reverent? Tone is a property of the writing itself and tends to stay consistent across a whole book or an author's body of work.
- **Mood** is the emotional atmosphere the reader experiences while reading — tense, cozy, melancholic, unsettling, hopeful. Mood can shift scene to scene even within a book with one consistent tone.

A useful test: tone is what the author is doing; mood is what the reader is feeling. A darkly comic tone (the author treats grim events with wit) can still produce a tense or unsettling mood in a given scene. Conflating the two loses information — a reader who says "I want something funny" is talking about tone; a reader who says "I want something that doesn't stress me out" is talking about mood, and these can point to different recommendations even within the same genre.

## Common Tone Categories

1. **Earnest** — sincere, direct emotional engagement with the material, no ironic distance.
2. **Wry / dry** — understated humor, deadpan observation, irony used sparingly.
3. **Satirical** — pointed exaggeration used to critique something about the world or its characters.
4. **Bleak** — unflinching, withholds comfort, doesn't soften difficult material.
5. **Reverent** — treats its subject matter (a place, a craft, a historical event) with weight and respect.
6. **Whimsical** — playful, willing to be silly or absurd without undercutting stakes entirely.

## Common Mood Categories

1. **Tense** — sustained anticipation of something going wrong; common in thrillers and horror, but also present in workplace or courtroom drama.
2. **Cozy** — low external stakes, a sense of safety even amid a puzzle to solve; common in certain mystery subgenres and slice-of-life fiction.
3. **Melancholic** — a persistent undertone of loss or wistfulness, even in books that aren't primarily sad.
4. **Unsettling** — a sense that something is subtly wrong, without necessarily being violent or frightening outright.
5. **Hopeful** — forward momentum toward something better, even when the plot includes real difficulty.
6. **Claustrophobic** — a feeling of confinement or narrowing options, common in locked-room mysteries and survival narratives.

## Why This Distinction Matters for Recommendations

Genre alone under-predicts mood and tone. Two mystery novels can be tonally opposite — one wry and cozy, one bleak and claustrophobic — while sharing a genre tag and even a similar plot structure (an investigator working a case). A reader who loved one for its cozy mood may DNF the other specifically because of tone mismatch, even though both books would be correctly tagged "mystery." This is the core justification for tracking mood and tone as independent, confidence-scored fields in the Reader Model rather than deriving them from genre.

Mood and tone preferences also interact with reading context in a way genre preferences often don't. A reader who typically enjoys Bleak, Unsettling books may specifically want something Hopeful and Cozy during a stressful period — not because their taste has changed, but because their current context calls for a different emotional register. This is exactly the kind of signal that should be flagged as temporary rather than durable (see the temporary-vs-durable distinction in Reader Modeling), and it's closely related to the Reading Slump Recovery Guide's discussion of matching books to a reader's current state rather than their historical average.

## Signals From Free Text

- Tone signals tend to come with explicit descriptive language: "I wanted something funnier than this," "too self-serious for what it was going for," "I loved how dry the narration was." These map fairly directly onto the tone categories above and should be treated as moderately durable — tone preference tends to be stable, closer to a taste than a mood.
- Mood signals are often expressed in terms of the reading *experience* rather than the book's content: "this was exactly the comfort read I needed," "I couldn't put it down because I was so on edge," "it left me feeling kind of hollow afterward." These should be checked against context before being recorded as durable — a reader reaching for a Cozy mood during a hard week is a context signal, not necessarily a standing preference update.
- When a reader's feedback conflicts with their historical mood preference (e.g., someone who usually rates Tense books highly now praising a Cozy one), the Reader Model should lean toward flagging this as temporary/contextual first, and only promote it to a durable preference shift if the pattern repeats across multiple entries — a single data point shouldn't overwrite an established preference.

## Mood, Tone, and the Beginner Reader

New readers with little logged history often describe books in mood and tone terms before they have genre vocabulary — "something not too heavy" or "something that keeps me hooked" are common early signals. The Beginner Genre Guide covers how to translate this kind of early, low-confidence feedback into a starting recommendation without over-committing to a genre label the reader hasn't actually expressed a preference about.