"""
Technicals section renderer. TODO: migrate from main.build_advanced_technical_analysis.
"""

from ml_backend.explain.budget import truncate_section, SECTION_BUDGETS


def render_technicals(technicals: dict, ticker: str) -> str:
    """Render technicals section with budget."""
    return truncate_section("", SECTION_BUDGETS["technical"], "technical")
