"""
Token/character budget for explanation prompt sections.
Prevents giant prompts and keeps responses stable.
"""

from typing import List, Dict, Any


# Max chars per section (approx; tokens ~= chars/4)
SECTION_BUDGETS = {
    "prediction": 500,
    "sentiment": 1200,
    "technical": 800,
    "news": 1000,
    "social": 1000,
    "fundamental": 1200,
    "insider": 600,
    "feature_importance": 400,
    "risk": 500,
}

# Top N items to keep per section (reduces noise)
TOP_N_ITEMS = {
    "headlines": 5,
    "posts": 5,
    "articles": 5,
}


def truncate_section(text: str, max_chars: int, section_name: str = "") -> str:
    """Truncate section to max_chars. Append ellipsis if truncated."""
    if not text or len(text) <= max_chars:
        return text or ""
    return text[: max_chars - 3].rsplit("\n", 1)[0] + "\n..."


def take_top_n(items: List[Dict], key: str, n: int) -> List[Dict]:
    """Keep top N items by sort key. Use for headlines, posts, etc."""
    if not items or n <= 0:
        return []
    sorted_items = sorted(items, key=lambda x: x.get(key, 0), reverse=True)
    return sorted_items[:n]
