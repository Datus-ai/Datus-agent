"""Service for data visualization chart recommendations with caching."""

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional

import pandas as pd

from datus.api.models.visualization_models import CsvData
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.prompts.prompt_manager import prompt_manager
from datus.schemas.visualization import VisualizationInput, VisualizationOutput
from datus.tools.llms_tools.visualization_tool import VisualizationTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Map VisualizationTool output chart_type → short frontend key
_CHART_TYPE_MAP = {
    "Bar Chart": "Bar",
    "Line Chart": "Line",
    "Pie Chart": "Pie",
    "Scatter Plot": "Scatter",
    "Unknown": "Unknown",
}

_MAX_CACHE_SIZE = 1000

# Prompt template for combined chart + context analysis
_CONTEXT_PROMPT_TEMPLATE = "visualization_with_context"


class DataVisualizationService:
    """Wraps VisualizationTool with result caching and DataFrame conversion."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self._tool: Optional[VisualizationTool] = None
        self._model: Optional[LLMBaseModel] = None
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Tool & model (lazy, cached)
    # ------------------------------------------------------------------

    def _get_tool(self) -> VisualizationTool:
        """Return (and cache) a VisualizationTool backed by the project's LLM."""
        if self._tool is not None:
            return self._tool

        model = self._get_model()
        self._tool = VisualizationTool(agent_config=self._agent_config, model=model)
        return self._tool

    def _get_model(self) -> Optional[LLMBaseModel]:
        """Return (and cache) the LLM model instance."""
        if self._model is not None:
            return self._model

        try:
            self._model = LLMBaseModel.create_model(agent_config=self._agent_config)
        except Exception as exc:
            logger.warning(f"Unable to initialize visualization model, using heuristics: {exc}")
        return self._model

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        csv_data: CsvData,
        chart_type: Optional[str],
        sql: Optional[str],
        user_question: Optional[str],
    ) -> str:
        """Compute a stable hash for the request payload."""
        payload = json.dumps(
            {
                "columns": csv_data.columns,
                "data": csv_data.data,
                "chart_type": chart_type,
                "sql": sql,
                "user_question": user_question,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_set(self, key: str, value: Dict[str, Any]) -> None:
        """Insert into cache, evicting the least-recently-used entry if over capacity."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > _MAX_CACHE_SIZE:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        csv_data: CsvData,
        chart_type: Optional[str] = None,
        sql: Optional[str] = None,
        user_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a chart recommendation dict, using cache when available."""
        key = self._cache_key(csv_data, chart_type, sql, user_question)

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        result = self._generate_uncached(csv_data, chart_type, sql, user_question)
        self._cache_set(key, result)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_uncached(
        self,
        csv_data: CsvData,
        chart_type: Optional[str],
        sql: Optional[str],
        user_question: Optional[str],
    ) -> Dict[str, Any]:
        # ── Build DataFrame ───────────────────────────────────────
        try:
            df = pd.DataFrame(csv_data.data, columns=csv_data.columns)
        except Exception as exc:
            logger.error(f"Failed to parse csv_data into DataFrame: {exc}")
            return {
                "success": False,
                "errorCode": "INVALID_DATA",
                "errorMessage": "Failed to parse the provided CSV data.",
            }

        if df.empty or df.shape[1] == 0:
            return {
                "success": False,
                "errorCode": "EMPTY_DATA",
                "errorMessage": "Provided dataset is empty or has no columns.",
            }

        # ── Column metadata ───────────────────────────────────────
        from pandas.api.types import is_numeric_dtype

        all_columns = df.columns.tolist()
        numeric_columns = [col for col in all_columns if is_numeric_dtype(df[col])]

        # ── Choose strategy: context-aware LLM vs basic tool ─────
        has_context = bool(sql or user_question)

        if has_context and self._get_model() is not None:
            chart_result = self._llm_with_context(df, sql, user_question)
        else:
            chart_result = self._basic_tool_recommendation(df)

        if chart_result is None:
            return {
                "success": False,
                "errorCode": "VISUALIZATION_FAILED",
                "errorMessage": "Visualization analysis failed.",
            }

        if not chart_result["success"]:
            return chart_result

        # ── Apply chart_type override ─────────────────────────────
        if chart_type is not None:
            chart_result["chart_type"] = chart_type

        # ── Build chart payload ────────────────────────────────────
        mapped_type = chart_result["chart_type"]

        chart: Dict[str, Any] = {
            "chart_type": mapped_type,
            "columns": all_columns,
            "numeric_columns": numeric_columns,
        }

        if mapped_type == "Unknown":
            chart["reason"] = chart_result.get("reason", "")
        else:
            chart["x_col"] = chart_result.get("x_col", "")
            chart["y_cols"] = chart_result.get("y_cols", [])
            chart["reason"] = chart_result.get("reason", "")

        # ── Build data_insight payload (only when context provided) ─
        data_insight: Optional[Dict[str, Any]] = None
        if has_context:
            data_insight = {
                "showing": chart_result.get("showing"),
                "period": chart_result.get("period"),
                "filters": chart_result.get("filters"),
                "insight": chart_result.get("insight"),
            }

        return {
            "success": True,
            "data": {
                "chart": chart,
                "data_insight": data_insight,
            },
        }

    # ------------------------------------------------------------------
    # Strategy: basic tool (no context)
    # ------------------------------------------------------------------

    def _basic_tool_recommendation(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run VisualizationTool (LLM chart-only or heuristic fallback)."""
        tool = self._get_tool()
        try:
            viz_input = VisualizationInput(data=df)
            result: VisualizationOutput = tool.execute(viz_input)
        except Exception as exc:
            logger.error(f"Visualization tool execution failed: {exc}")
            return {
                "success": False,
                "errorCode": "VISUALIZATION_FAILED",
                "errorMessage": "Visualization analysis failed.",
            }

        if not result.success:
            return {
                "success": False,
                "errorCode": "VISUALIZATION_FAILED",
                "errorMessage": result.error or "Visualization analysis failed.",
            }

        mapped_type = _CHART_TYPE_MAP.get(result.chart_type, "Unknown")
        return {
            "success": True,
            "chart_type": mapped_type,
            "x_col": result.x_col,
            "y_cols": result.y_cols,
            "reason": result.reason,
        }

    # ------------------------------------------------------------------
    # Strategy: context-aware LLM (merged call)
    # ------------------------------------------------------------------

    def _llm_with_context(
        self,
        df: pd.DataFrame,
        sql: Optional[str],
        user_question: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call that returns chart config + metadata + insight."""
        tool = self._get_tool()
        columns_info = tool._format_columns_info(df)
        data_preview = tool._format_data_preview(df)

        prompt = prompt_manager.render_template(
            _CONTEXT_PROMPT_TEMPLATE,
            columns_info=columns_info,
            data_preview=data_preview,
            sql=sql or "",
            user_question=user_question or "",
        )

        model = self._get_model()
        try:
            response = model.generate_with_json_output(prompt)
        except Exception as exc:
            logger.error(f"Context-aware visualization LLM call failed: {exc}")
            # Fall back to basic tool
            return self._basic_tool_recommendation(df)

        if not isinstance(response, dict):
            logger.warning("Context-aware LLM response is not a dict, falling back to basic tool")
            return self._basic_tool_recommendation(df)

        # ── Parse chart config ────────────────────────────────────
        chart_type = tool._normalize_chart_type(response.get("chart_type", ""))
        mapped_type = _CHART_TYPE_MAP.get(chart_type, "Unknown")

        x_col = response.get("x_col") or ""
        y_cols = response.get("y_cols") or []
        if isinstance(y_cols, str):
            y_cols = [y_cols]

        # Sanitize against actual columns
        x_col = x_col if x_col in df.columns else tool._select_dimension(df, exclude=set(y_cols))
        y_cols = tool._sanitize_y_cols(df, y_cols, exclude={x_col})

        reason = (response.get("reason") or "").strip()
        if not reason:
            reason = tool._default_reason(chart_type, x_col, y_cols)

        # ── Parse context metadata ────────────────────────────────
        showing = response.get("showing")
        if isinstance(showing, dict):
            showing = {
                "metrics": showing.get("metrics") or [],
                "dimensions": showing.get("dimensions") or [],
            }
        else:
            showing = None

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

        return {
            "success": True,
            "chart_type": mapped_type,
            "x_col": x_col,
            "y_cols": y_cols,
            "reason": reason,
            "showing": showing,
            "period": period,
            "filters": filters,
            "insight": insight,
        }
