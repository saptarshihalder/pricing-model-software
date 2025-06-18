"""Utility functions for the pricing tools."""
import re


def canonical_key(name: str) -> str:
    """Return a lowercase alphanumeric key for a category or filename."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())
