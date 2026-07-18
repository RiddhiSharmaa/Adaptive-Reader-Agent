"""
Abstract tool layer — the seam between intent handlers and data sources.

This is what gets swapped out in Part 3 (mock JSON -> live API)
without touching the planner or intent handlers.
"""


def get_reader_profile(reader_id: str) -> dict:
    raise NotImplementedError


def update_reader_profile(reader_id: str, updates: dict) -> dict:
    raise NotImplementedError


def log_book(reader_id: str, book_data: dict) -> dict:
    raise NotImplementedError


def search_books(query: str) -> list[dict]:
    raise NotImplementedError


def get_author_stats(author_id: str) -> dict:
    raise NotImplementedError