# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import get_prompt_manager
from datus.schemas.visualization import VisualizationInput, VisualizationOutput, VisualizationWithContextOutput
from datus.tools.base import BaseTool
from datus.tools.llms_tools.visualization_messages import empty_dataset_reason, reason_for_chart
from datus.utils.language_utils import resolve_language_name
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class VisualizationTool(BaseTool):
    """Tool that recommends an appropriate visualization for the provided dataset."""

    tool_name = "visualization_tool"
    tool_description = "Recommend a chart configuration (chart type, x axis, y axes) for a dataset."

    PROMPT_TEMPLATE = "visualization_system"
    CONTEXT_PROMPT_TEMPLATE = "visualization_with_context"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        model: Optional[LLMBaseModel] = None,
        prompt_version: Optional[str] = None,
        preview_rows: int = 5,
        max_preview_char: int = 1500,
        max_y_cols: int = 3,
        max_pie_categories: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_config = agent_config
        self.prompt_version = prompt_version
        self.preview_rows = preview_rows
        self.max_preview_char = max_preview_char
        self.max_y_cols = max_y_cols
        self.max_pie_categories = max_pie_categories

        # Lazy model resolution — ``/model`` can swap the active target at
        # runtime, and ``LLMBaseModel.create_model`` caches the resulting
        # instance, so reading :attr:`model` per-call is both correct and
        # cheap. An explicit ``model=`` kwarg still wins (used by tests and
        # callers that pre-bind a custom client).
        self._injected_model = model
        self.preview_rows = preview_rows
        self.max_preview_char = max_preview_char

    @property
    def model(self) -> Optional[LLMBaseModel]:
        if self._injected_model is not None:
            return self._injected_model
        if self.agent_config is None:
            return None
        try:
            return LLMBaseModel.create_model(agent_config=self.agent_config)
        except Exception as exc:
            logger.debug(f"Lazy visualization model resolution failed: {exc}")
            return None

    def execute(
        self,
        input_data: VisualizationInput,
        language: Optional[str] = None,
        total_rows: Optional[int] = None,
    ) -> VisualizationOutput:
        """Generate visualization recommendation using LLM if available, otherwise heuristics.

        ``language`` pins the human-readable output (``reason``) to a language
        code; it falls back to ``agent_config.language`` when unset.
        ``total_rows`` states the full result size when ``input_data`` is a sample.
        """
        if not isinstance(input_data, VisualizationInput):
            raise TypeError("VisualizationTool expects VisualizationInput as input data.")

        dataframe = self._convert_to_dataframe(input_data.data)
        if dataframe is None:
            error_msg = "VisualizationInput data must be a pandas.DataFrame, list, or pyarrow.Table."
            logger.error(error_msg)
            return self._error_output(error_msg, "Unknown")

        if dataframe.empty or dataframe.shape[1] == 0:
            error_msg = "Provided dataset is empty or has no columns."
            logger.error(error_msg)
            return self._error_output(
                error_msg, "Unknown", reason=empty_dataset_reason(self._effective_language(language))
            )

        result = None
        if self.model:
            try:
                result = self._llm_based_recommendation(dataframe, language=language, total_rows=total_rows)
            except Exception as exc:
                logger.warning(f"LLM visualization recommendation failed, falling back to heuristics: {exc}")

        if result is None:
            result = self._rule_based_recommendation(dataframe, language=language)

        return result

    def execute_with_context(
        self,
        input_data: VisualizationInput,
        sql: Optional[str] = None,
        user_question: Optional[str] = None,
        language: Optional[str] = None,
        total_rows: Optional[int] = None,
    ) -> VisualizationWithContextOutput:
        """Generate visualization with data context (showing, period, filters, insight).

        Uses a single merged LLM call. Falls back to rule-based heuristics
        (without context metadata) if the LLM call fails.

        ``language`` pins the human-readable output (``reason``, ``filters``,
        ``insight``) to a language code; it falls back to
        ``agent_config.language`` when unset. ``total_rows`` states the full
        result size when ``input_data`` is a sample.
        """
        if not isinstance(input_data, VisualizationInput):
            raise TypeError("VisualizationTool expects VisualizationInput as input data.")

        dataframe = self._convert_to_dataframe(input_data.data)
        if dataframe is None:
            error_msg = "VisualizationInput data must be a pandas.DataFrame, list, or pyarrow.Table."
            logger.error(error_msg)
            return VisualizationWithContextOutput(
                success=False,
                error=error_msg,
                chart_type="Unknown",
                x_col="",
                y_cols=[],
                reason=error_msg,
            )

        if dataframe.empty or dataframe.shape[1] == 0:
            error_msg = "Provided dataset is empty or has no columns."
            logger.error(error_msg)
            return VisualizationWithContextOutput(
                success=False,
                error=error_msg,
                chart_type="Unknown",
                x_col="",
                y_cols=[],
                reason=empty_dataset_reason(self._effective_language(language)),
            )

        # Try context-aware LLM call
        if self.model:
            try:
                result = self._llm_with_context(dataframe, sql, user_question, language=language, total_rows=total_rows)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning(f"Context-aware LLM call failed, falling back to heuristics: {exc}")

        # Fall back directly to rule-based heuristics (no second LLM call)
        base = self._rule_based_recommendation(dataframe, language=language)
        return VisualizationWithContextOutput(
            success=base.success,
            error=base.error,
            chart_type=base.chart_type,
            x_col=base.x_col,
            y_cols=base.y_cols,
            reason=base.reason,
        )

    # ------------------------------------------------------------------ #
    # Context-aware LLM recommendation
    # ------------------------------------------------------------------ #
    def _llm_with_context(
        self,
        df: pd.DataFrame,
        sql: Optional[str],
        user_question: Optional[str],
        language: Optional[str] = None,
        total_rows: Optional[int] = None,
    ) -> Optional[VisualizationWithContextOutput]:
        """Single LLM call returning chart config + context metadata."""
        prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
            self.CONTEXT_PROMPT_TEMPLATE,
            version=self.prompt_version,
            columns_info=self._format_columns_info(df),
            data_preview=self._format_data_preview(df),
            sampling_note=self._sampling_note(df, total_rows),
            sql=sql or "",
            user_question=user_question or "",
            language_directive=self._language_directive(language),
        )

        response = self.model.generate_with_json_output(prompt)
        if not isinstance(response, dict):
            logger.warning("Context-aware LLM response is not a dict, ignoring it")
            return None

        # ── Parse chart config ────────────────────────────────────
        chart_type = self._normalize_chart_type(response.get("chart_type", ""))
        x_col = response.get("x_col") or ""
        y_cols = response.get("y_cols") or []
        if isinstance(y_cols, str):
            y_cols = [y_cols]

        x_col = x_col if x_col in df.columns else self._select_dimension(df, exclude=set(y_cols))
        y_cols = self._sanitize_y_cols(df, y_cols, exclude={x_col})

        reason = (response.get("reason") or "").strip()
        if not reason:
            reason = self._default_reason(chart_type, x_col, y_cols, language=language)

        # ── Parse context metadata ────────────────────────────────
        period = response.get("period")
        if not isinstance(period, str):
            period = None

        filters = response.get("filters")
        if not isinstance(filters, list):
            filters = []
        filters = [f for f in filters if isinstance(f, str)]

        insight = response.get("insight")
        if not isinstance(insight, str):
            insight = None

        return VisualizationWithContextOutput(
            success=True,
            error=None,
            chart_type=chart_type,
            x_col=x_col or "",
            y_cols=y_cols,
            reason=reason,
            period=period,
            filters=filters,
            insight=insight,
        )

    # ------------------------------------------------------------------ #
    # Response language
    # ------------------------------------------------------------------ #
    def _effective_language(self, language: Optional[str]) -> str:
        """Resolve the pinned language code, or ``""`` when none is configured."""
        # A blank explicit override is "unset", not "no language" — otherwise a
        # caller sending an empty field would suppress the configured default.
        explicit = str(language).strip() if language else ""
        fallback = getattr(self.agent_config, "language", None)
        return explicit or (str(fallback).strip() if fallback else "")

    def _language_directive(self, language: Optional[str]) -> str:
        """Render the ``response_language`` section, or ``""`` when unpinned.

        This tool is a standalone LLM call with no system prompt, so the
        directive every agentic node gets injected has to be restated in the
        prompt itself — otherwise an English template is answered in English
        regardless of the language the caller asked for.
        """
        code = self._effective_language(language)
        if not code:
            return ""
        language_name = resolve_language_name(code)
        try:
            section = get_prompt_manager(agent_config=self.agent_config).render_template(
                "response_language",
                version=None,
                language_code=code,
                language_name=language_name,
            )
        except Exception as exc:
            # A render failure must not silently drop a pinned language.
            logger.warning(f"Failed to render response_language template: {exc}")
            return f"# Response Language\n- Use: {language_name} ({code})"
        return section.strip() or f"# Response Language\n- Use: {language_name} ({code})"

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #
    def _convert_to_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        """Normalize supported data formats into a pandas DataFrame."""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pa.Table):
            return data.to_pandas()
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            try:
                return pd.DataFrame(data)
            except Exception as exc:
                logger.error(f"Failed to convert list data to DataFrame: {exc}")
                return None
        logger.error(f"Unsupported data type for visualization: {type(data)}")
        return None

    # ------------------------------------------------------------------ #
    # LLM recommendation
    # ------------------------------------------------------------------ #
    def _llm_based_recommendation(
        self, df: pd.DataFrame, language: Optional[str] = None, total_rows: Optional[int] = None
    ) -> Optional[VisualizationOutput]:
        """Use LLM to recommend visualization configuration."""
        prompt = get_prompt_manager(agent_config=self.agent_config).render_template(
            self.PROMPT_TEMPLATE,
            version=self.prompt_version,
            columns_info=self._format_columns_info(df),
            data_preview=self._format_data_preview(df),
            sampling_note=self._sampling_note(df, total_rows),
            language_directive=self._language_directive(language),
        )

        response = self.model.generate_with_json_output(prompt)
        if not isinstance(response, dict):
            logger.warning("LLM response for visualization is not a dict, ignoring it")
            return None

        chart_type = self._normalize_chart_type(response.get("chart_type", ""))
        x_col = response.get("x_col") or ""
        y_cols = response.get("y_cols") or []
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        reason = (response.get("reason") or "").strip()

        x_col = x_col if x_col in df.columns else self._select_dimension(df, exclude=set(y_cols))
        y_cols = self._sanitize_y_cols(df, y_cols, exclude={x_col})

        if not reason:
            reason = self._default_reason(chart_type, x_col, y_cols, language=language)

        return VisualizationOutput(
            success=True,
            error=None,
            chart_type=chart_type,
            x_col=x_col or "",
            y_cols=y_cols,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Heuristic recommendation fallback
    # ------------------------------------------------------------------ #
    def _rule_based_recommendation(self, df: pd.DataFrame, language: Optional[str] = None) -> VisualizationOutput:
        """Recommend visualization using lightweight heuristics.

        ``reason`` comes from a static translation table rather than an f-string:
        this path is reached exactly when no LLM is available, so the wording
        cannot be generated — but the caller still asked for a language.
        """
        numeric_cols = [col for col in df.columns if self._is_numeric(df[col])]
        datetime_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
        categorical_cols = [col for col in df.columns if self._is_categorical(df[col])]

        chart_type = "Unknown"
        x_col = ""
        y_cols: List[str] = []

        pie_candidate = (
            len(categorical_cols) == 1
            and len(numeric_cols) == 1
            and df[categorical_cols[0]].nunique(dropna=True) <= self.max_pie_categories
        )

        if datetime_cols and numeric_cols:
            x_col = datetime_cols[0]
            y_cols = self._select_numeric_metrics(numeric_cols, exclude={x_col})
            chart_type = "Line Chart"
        elif pie_candidate:
            x_col = categorical_cols[0]
            y_cols = [numeric_cols[0]]
            chart_type = "Pie Chart"
        elif categorical_cols and numeric_cols:
            x_col = categorical_cols[0]
            y_cols = self._select_numeric_metrics(numeric_cols, exclude=set())
            chart_type = "Bar Chart"
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_cols = [numeric_cols[1]]
            chart_type = "Scatter Plot"

        return VisualizationOutput(
            success=True,
            error=None,
            chart_type=chart_type,
            x_col=x_col,
            y_cols=y_cols,
            reason=self._default_reason(chart_type, x_col, y_cols, language=language),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _format_columns_info(self, df: pd.DataFrame) -> str:
        info_parts = []
        for column, dtype in df.dtypes.items():
            unique_count = df[column].nunique(dropna=True)
            info_parts.append(f"{column} ({dtype}, unique={unique_count})")
        return ", ".join(info_parts)

    def _sampling_note(self, df: pd.DataFrame, total_rows: Optional[int]) -> str:
        """Tell the model the rows it got are a sample, when the caller says so.

        Both the preview and the per-column ``unique=`` counts are computed over
        the rows we were handed, so a model told nothing would size the result
        set by the sample and read its extremes as global ones.
        """
        rows = len(df)
        if not total_rows or total_rows <= rows:
            return ""

        return (
            f"The dataset below is the first {rows} rows of a {total_rows}-row result set. "
            "The preview and the column cardinalities describe that sample only — state the "
            "result size as the full total, and do not present sample extremes as global ones."
        )

    def _format_data_preview(self, df: pd.DataFrame) -> str:
        preview_df = df.head(self.preview_rows).replace({np.nan: None})
        serializable_records = []
        for row in preview_df.to_dict(orient="records"):
            serializable_records.append({key: self._serialize_value(value) for key, value in row.items()})

        preview_str = json.dumps(serializable_records, ensure_ascii=False)
        if len(preview_str) > self.max_preview_char:
            preview_str = preview_str[: self.max_preview_char] + "... (truncated)"
        return preview_str

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, (set, list, tuple)):
            return list(value)
        return value

    def _normalize_chart_type(self, chart_type: str) -> str:
        if not chart_type:
            return "Unknown"

        normalized = chart_type.strip().lower()
        mapping = {
            "bar": "Bar Chart",
            "bar chart": "Bar Chart",
            "line": "Line Chart",
            "line chart": "Line Chart",
            "scatter": "Scatter Plot",
            "scatter plot": "Scatter Plot",
            "pie": "Pie Chart",
            "pie chart": "Pie Chart",
            "unknown": "Unknown",
        }
        return mapping.get(normalized, "Unknown")

    def _default_reason(
        self, chart_type: str, x_col: str, y_cols: Sequence[str], language: Optional[str] = None
    ) -> str:
        """Localized wording for a chart pick the LLM didn't explain itself."""
        return reason_for_chart(
            chart_type,
            language=self._effective_language(language),
            x_col=x_col,
            y_cols=list(y_cols),
        )

    def _select_dimension(self, df: pd.DataFrame, exclude: Optional[set[str]] = None) -> str:
        exclude = exclude or set()
        for column in df.columns:
            if column in exclude:
                continue
            series = df[column]
            if is_datetime64_any_dtype(series) or self._is_categorical(series):
                return column
        for column in df.columns:
            if column not in exclude:
                return column
        return ""

    def _sanitize_y_cols(
        self, df: pd.DataFrame, candidate_cols: Sequence[str], exclude: Optional[set[str]] = None
    ) -> List[str]:
        exclude = exclude or set()
        sanitized = [col for col in candidate_cols if col in df.columns and col not in exclude]
        if not sanitized:
            sanitized = self._select_numeric_metrics(
                [col for col in df.columns if self._is_numeric(df[col])],
                exclude=exclude,
            )
        return sanitized

    def _select_numeric_metrics(self, numeric_cols: Sequence[str], exclude: Optional[set[str]] = None) -> List[str]:
        exclude = exclude or set()
        metrics: List[str] = []
        for col in numeric_cols:
            if col in exclude:
                continue
            metrics.append(col)
            if len(metrics) >= self.max_y_cols:
                break
        return metrics

    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        return is_numeric_dtype(series)

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return is_object_dtype(series) or is_categorical_dtype(series) or is_bool_dtype(series)

    def _error_output(self, error: str, chart_type: str, reason: str = "") -> VisualizationOutput:
        return VisualizationOutput(
            success=False,
            error=error,
            chart_type=chart_type,
            x_col="",
            y_cols=[],
            reason=reason or error,
        )
