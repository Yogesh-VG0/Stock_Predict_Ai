"""
AI explanation module: prompt building and token/length budgeting.
"""

from ml_backend.explain.budget import truncate_section, take_top_n, SECTION_BUDGETS, TOP_N_ITEMS

__all__ = ["truncate_section", "take_top_n", "SECTION_BUDGETS", "TOP_N_ITEMS"]
