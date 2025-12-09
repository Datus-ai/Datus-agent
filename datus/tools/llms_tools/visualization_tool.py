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
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.visualization import VisualizationInput, VisualizationOutput
from datus.tools.base import BaseTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class VisualizationTool(BaseTool):
    """Tool that recommends an appropriate visualization for the provided dataset."""

    tool_name = "visualization_tool"
    tool_description = "Recommend a chart configuration (chart type, x axis, y axes) for a dataset."

    PROMPT_TEMPLATE = "visualization_system"

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        model: Optional[LLMBaseModel] = None,
        prompt_version: str = "1.0",
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

        if model is not None:
            self.model = model
        elif agent_config is not None:
            try:
                self.model = LLMBaseModel.create_model(agent_config=agent_config)
            except Exception as exc:
                logger.warning(f"Failed to initialize visualization model, using heuristics only: {exc}")
                self.model = None
        else:
            self.model = None
        self.preview_rows = preview_rows
        self.max_preview_char = max_preview_char

    def execute(self, input_data: VisualizationInput) -> VisualizationOutput:
        """Generate visualization recommendation using LLM if available, otherwise heuristics."""
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
            return self._error_output(error_msg, "Unknown", reason="Dataset does not contain any records")

        result = None
        if self.model:
            try:
                result = self._llm_based_recommendation(dataframe)
            except Exception as exc:
                logger.warning(f"LLM visualization recommendation failed, falling back to heuristics: {exc}")

        if result is None:
            result = self._rule_based_recommendation(dataframe)

        return result

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
    def _llm_based_recommendation(self, df: pd.DataFrame) -> Optional[VisualizationOutput]:
        """Use LLM to recommend visualization configuration."""
        prompt = prompt_manager.render_template(
            self.PROMPT_TEMPLATE,
            version=self.prompt_version,
            columns_info=self._format_columns_info(df),
            data_preview=self._format_data_preview(df),
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
            reason = self._default_reason(chart_type, x_col, y_cols)

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
    def _rule_based_recommendation(self, df: pd.DataFrame) -> VisualizationOutput:
        """Recommend visualization using lightweight heuristics."""
        numeric_cols = [col for col in df.columns if self._is_numeric(df[col])]
        datetime_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
        categorical_cols = [col for col in df.columns if self._is_categorical(df[col])]

        chart_type = "Unknown"
        x_col = ""
        y_cols: List[str] = []
        reason = "Unable to determine an ideal visualization for the provided data."

        pie_candidate = (
            len(categorical_cols) == 1
            and len(numeric_cols) == 1
            and df[categorical_cols[0]].nunique(dropna=True) <= self.max_pie_categories
        )

        if datetime_cols and numeric_cols:
            x_col = datetime_cols[0]
            y_cols = self._select_numeric_metrics(numeric_cols, exclude={x_col})
            chart_type = "Line Chart"
            reason = (
                f"Detected datetime column '{x_col}' with {len(y_cols)} numeric metric(s), "
                "suggesting a line chart to highlight trends."
            )
        elif pie_candidate:
            x_col = categorical_cols[0]
            y_cols = [numeric_cols[0]]
            chart_type = "Pie Chart"
            reason = (
                f"Categorical column '{x_col}' has only {df[x_col].nunique(dropna=True)} distinct values "
                f"with numeric metric '{y_cols[0]}', making it suitable for a pie chart."
            )
        elif categorical_cols and numeric_cols:
            x_col = categorical_cols[0]
            y_cols = self._select_numeric_metrics(numeric_cols, exclude=set())
            chart_type = "Bar Chart"
            reason = (
                f"Categorical column '{x_col}' paired with numeric metrics {y_cols} "
                "is best represented using a bar chart to compare categories."
            )
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_cols = [numeric_cols[1]]
            chart_type = "Scatter Plot"
            reason = (
                f"Multiple numeric columns detected ({numeric_cols[:2]}), "
                "indicating a scatter plot is suitable for correlation analysis."
            )

        return VisualizationOutput(
            success=True,
            error=None,
            chart_type=chart_type,
            x_col=x_col,
            y_cols=y_cols,
            reason=reason,
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

    def _format_data_preview(self, df: pd.DataFrame) -> str:
        preview_df = df.head(self.preview_rows)
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

    def _default_reason(self, chart_type: str, x_col: str, y_cols: Sequence[str]) -> str:
        if chart_type == "Line Chart":
            return f"Line chart illustrates how {', '.join(y_cols) or 'metrics'} evolve over {x_col or 'time'}."
        if chart_type == "Bar Chart":
            return f"Bar chart compares numeric metrics {y_cols or ['values']} across categories in {x_col}."
        if chart_type == "Scatter Plot" and y_cols:
            return f"Scatter plot reveals the relationship between numeric fields {y_cols[0]} and {x_col}."
        if chart_type == "Pie Chart" and y_cols:
            return f"Pie chart shows the share of {y_cols[0]} across categories in {x_col}."
        return "Unable to determine the best visualization due to insufficient information."

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
