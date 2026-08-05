# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Localized wording for chart recommendations produced without an LLM.

The heuristic path runs precisely when no model is reachable, so its ``reason``
can be neither generated nor translated at call time — it has to ship as static
per-language templates. Unknown language codes fall back to English.
"""

from typing import Any, Dict, Optional, Sequence

FALLBACK_LANGUAGE = "en"

# chart_type as produced by VisualizationTool -> message key
_CHART_KEYS: Dict[str, str] = {
    "Line Chart": "line",
    "Bar Chart": "bar",
    "Pie Chart": "pie",
    "Scatter Plot": "scatter",
}

# language code -> message key -> template. ``{x}`` is the dimension column,
# ``{y}`` the comma-joined metric columns.
_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "line": "Line chart over '{x}' shows the trend of {y} over time.",
        "bar": "Bar chart compares {y} across the categories in '{x}'.",
        "pie": "Pie chart shows the share of {y} across the categories in '{x}'.",
        "scatter": "Scatter plot reveals the relationship between {y} and '{x}'.",
        "unknown": "Unable to determine an ideal visualization for the provided data.",
        "empty": "Dataset does not contain any records.",
    },
    "zh": {
        "line": "折线图按 '{x}' 展示 {y} 随时间的变化趋势。",
        "bar": "柱状图按 '{x}' 的各个类别对比 {y}。",
        "pie": "饼图展示 {y} 在 '{x}' 各类别中的占比。",
        "scatter": "散点图展示 {y} 与 '{x}' 之间的关系。",
        "unknown": "无法根据所提供的数据判断合适的图表类型。",
        "empty": "数据集中没有任何记录。",
    },
}


def normalize_language(code: Optional[str]) -> str:
    """Reduce a language tag to a key present in the message table.

    Regional tags collapse onto their base language (``zh-CN``/``zh_CN`` → ``zh``);
    an exact match wins first, so adding a regional entry to the table is enough
    to give it its own wording. Anything unmapped falls back to English.
    """
    if not code:
        return FALLBACK_LANGUAGE
    normalized = str(code).strip().lower().replace("_", "-")
    if normalized in _MESSAGES:
        return normalized
    base = normalized.split("-", 1)[0]
    return base if base in _MESSAGES else FALLBACK_LANGUAGE


def _render(key: str, language: Optional[str], **params: str) -> str:
    table = _MESSAGES[normalize_language(language)]
    template = table.get(key) or _MESSAGES[FALLBACK_LANGUAGE].get(key, "")
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return template


def reason_for_chart(
    chart_type: str,
    language: Optional[str] = None,
    x_col: Any = "",
    y_cols: Sequence[Any] = (),
) -> str:
    """Explain a heuristic chart pick in the caller's language.

    Falls back to the ``unknown`` wording for chart types we have no template
    for, and whenever an axis is absent — a sentence naming an empty column
    reads worse than admitting the data was inconclusive. Labels are stringified
    rather than tested for truthiness, so a DataFrame column literally named
    ``0`` is a label, not a missing axis.
    """
    key = _CHART_KEYS.get(chart_type)
    x_label = "" if x_col is None else str(x_col)
    y_labels = [str(col) for col in y_cols if col is not None]
    if not key or not x_label or not y_labels:
        return _render("unknown", language)
    return _render(key, language, x=x_label, y=", ".join(y_labels))


def empty_dataset_reason(language: Optional[str] = None) -> str:
    """Wording for a request whose dataset carries no rows/columns."""
    return _render("empty", language)
