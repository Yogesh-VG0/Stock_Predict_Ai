"""
Fundamentals section renderer. TODO: migrate from main.build_fundamental_analysis.
"""

from ml_backend.explain.budget import truncate_section, SECTION_BUDGETS


def render_fundamentals(sentiment: dict, ticker: str) -> str:
    """Render fundamentals section with budget."""
    return truncate_section("", SECTION_BUDGETS["fundamental"], "fundamental")
