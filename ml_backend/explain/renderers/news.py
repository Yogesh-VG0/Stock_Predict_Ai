"""
News section renderer. TODO: migrate from main.build_news_intelligence_analysis.
"""

from ml_backend.explain.budget import truncate_section, take_top_n, SECTION_BUDGETS, TOP_N_ITEMS


def render_news(sentiment: dict, ticker: str) -> str:
    """Render news section with budget."""
    # Placeholder; actual logic migrates from main.build_news_intelligence_analysis
    return truncate_section("", SECTION_BUDGETS["news"], "news")
