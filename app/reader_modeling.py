"""
Reader Modeling: shared mechanism (not a standalone intent).

Processes free-text reader feedback into structured signals,
classifies each signal as temporary or durable, and updates
reader_profiles.json with confidence-scored preferences.

Called by multiple intent handlers (e.g. log_outcome), not routed
to directly by the intent classifier.
"""