"""
Social section renderer. TODO: migrate from main.build_social_sentiment_analysis.
"""

from ml_backend.explain.budget import truncate_section, SECTION_BUDGETS


def render_social(sentiment: dict, ticker: str) -> str:
    """Render social section with budget."""
    return truncate_section("", SECTION_BUDGETS["social"], "social")
