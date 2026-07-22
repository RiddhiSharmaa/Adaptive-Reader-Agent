# Beginner Genre Guide

A reference guide to recommending well for readers with little or no logged history, without over-committing to labels they haven't actually expressed a preference about.

## The Cold-Start Problem

A new reader has no rating history, no DNF pattern, and no confidence-scored preferences yet. Every field in their Reader Model starts empty or at minimal confidence. This is the cold-start problem, and it's distinct from ordinary low-confidence recommendation: with an existing reader, low confidence on one axis (say, trope) can be offset by high confidence on another (pacing). With a brand-new reader, there's often nothing to offset with at all.

The temptation is to ask a long intake questionnaire up front — favorite genres, favorite authors, favorite tropes. This tends to backfire. Readers are often bad at self-reporting preference in the abstract (many people say they "like everything" or default to whatever genre is most culturally prominent at the time) and a long questionnaire delays the first real recommendation, which is where the actual signal starts coming in. The better approach is a short, low-friction first question, followed by fast iteration based on real reactions.

## What to Ask a Brand-New Reader

A single well-chosen question does more work than five generic ones. Useful angles, in rough order of how much signal they tend to yield:

1. **A recent read they enjoyed, named directly** — "what's a book you finished recently and liked?" This gives the agent a real anchor point to pull pacing, mood, tone, and trope signals from (via its own knowledge, not the reader's self-analysis), rather than asking the reader to describe their own taste in the abstract.
2. **A mood/tone question, not a genre question** — "are you looking for something that keeps you on edge, something comforting, or something that makes you think?" Mood and tone are more accessible to articulate than genre or trope, and, per the Mood & Tone Guide, are independently useful signals regardless of genre.
3. **A pacing question, framed concretely** — "do you want something you can read in short bursts, or something you want to sink into for a long stretch?" This maps directly onto the Pacing Taxonomy without requiring the reader to know that vocabulary.

Genre should be treated as the least reliable starting question, not the first one. A reader who says "I like fantasy" has given the agent far less usable signal than a reader who names one specific fantasy book they loved and one they DNF'd, because genre labels don't reveal pacing, mood, tone, or trope — the actual predictive variables in this system.

## Confidence Scoring for New Readers

Every inference drawn from a beginner reader's first few interactions should start at low confidence, even when the signal seems clear. A single strongly-worded reaction ("I could not put this down") is a real signal, but it shouldn't be treated the same as a pattern confirmed across five books. The Reader Model should be explicit about this distinction — new readers accumulate confidence quickly if their early signals are internally consistent (several early reactions pointing the same direction), and slowly if their early signals conflict.

This means the Planner should lean more heavily on the RAG knowledge base and live book metadata for a brand-new reader's early recommendations, and less on the (nearly empty) Reader Model — the opposite balance from a returning reader with an established profile. As confidence accumulates, that balance should shift toward the Reader Model taking a larger role in each recommendation.

## Avoiding Premature Genre Commitment

A specific failure mode to guard against: recommending within a single genre repeatedly just because the reader's first liked book happened to be in that genre, without verifying that genre itself was the actual driver of their enjoyment. If a new reader liked a mystery novel that happened to be Brisk-paced, Cozy in mood, with a locked-room structural trope, the safer next recommendation tests one of those specific attributes in a different genre — rather than assuming "more mystery" is the correct read of a single data point. This is the beginner-reader equivalent of the genre-vs-trope and genre-vs-pacing distinctions made throughout the other guides: genre is the most visible signal and the least informative one, and that's especially true before any preference history exists to correct for it.

## When a Beginner Reader Should Get a Direct Recommendation vs. a Clarifying Question

If the reader has given at least one concrete anchor (a named book, or a clear mood/pacing answer), the Planner can make a first recommendation directly rather than asking further questions — over-questioning a new reader before giving them anything useful risks losing their engagement before the agent has proven its value. If the reader's opening message is genuinely vague ("I want something good to read"), one clarifying question is worth asking, but it should be the mood or pacing question above, not a request for a genre.