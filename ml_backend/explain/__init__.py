"""
AI explanation module: prompt building, section renderers, token/length budgeting.
TODO: Migrate build_comprehensive_explanation_prompt and helpers from api/main.py.
"""

from ml_backend.explain.budget import truncate_section, take_top_n, SECTION_BUDGETS, TOP_N_ITEMS

__all__ = ["truncate_section", "take_top_n", "SECTION_BUDGETS", "TOP_N_ITEMS"]
