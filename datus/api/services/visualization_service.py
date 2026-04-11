"""Service for data visualization chart recommendations with caching."""

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, Optional

import pandas as pd

from datus.api.models.visualization_models import CsvData
from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.visualization import VisualizationInput
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


class DataVisualizationService:
    """Wraps VisualizationTool with result caching and DataFrame conversion."""

    def __init__(self, agent_config: AgentConfig):
        self._agent_config = agent_config
        self._tool: Optional[VisualizationTool] = None
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Tool (lazy, cached)
    # ------------------------------------------------------------------

    def _get_tool(self) -> VisualizationTool:
        """Return (and cache) a VisualizationTool backed by the project's LLM."""
        if self._tool is not None:
            return self._tool

        model = None
        try:
            model = LLMBaseModel.create_model(agent_config=self._agent_config)
        except Exception as exc:
            logger.warning(f"Unable to initialize visualization model, using heuristics: {exc}")

        self._tool = VisualizationTool(agent_config=self._agent_config, model=model)
        return self._tool

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

        # ── Execute tool ──────────────────────────────────────────
        tool = self._get_tool()
        has_context = bool(sql or user_question)

        try:
            viz_input = VisualizationInput(data=df)
            if has_context:
                result = tool.execute_with_context(viz_input, sql=sql, user_question=user_question)
            else:
                result = tool.execute(viz_input)
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

        # ── Apply chart_type override ─────────────────────────────
        mapped_type = _CHART_TYPE_MAP.get(result.chart_type, "Unknown")
        if chart_type is not None:
            mapped_type = chart_type

        # ── Build chart payload ───────────────────────────────────
        chart: Dict[str, Any] = {
            "chart_type": mapped_type,
            "columns": all_columns,
            "numeric_columns": numeric_columns,
        }

        if mapped_type == "Unknown":
            chart["reason"] = result.reason
        else:
            chart["x_col"] = result.x_col
            chart["y_cols"] = result.y_cols
            chart["reason"] = result.reason

        # ── Build data_insight payload ────────────────────────────
        data_insight: Optional[Dict[str, Any]] = None
        if has_context:
            data_insight = {
                "period": getattr(result, "period", None),
                "filters": getattr(result, "filters", None),
                "insight": getattr(result, "insight", None),
            }

        return {
            "success": True,
            "data": {
                "chart": chart,
                "data_insight": data_insight,
            },
        }
