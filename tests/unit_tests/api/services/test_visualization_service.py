# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/services/visualization_service.py — CI level, zero external deps."""

from unittest.mock import Mock, patch

import pytest

from datus.api.models.visualization_models import CsvData
from datus.api.services.visualization_service import DataVisualizationService

_LLM_PATH = "datus.api.services.visualization_service.LLMBaseModel"
_VIZ_TOOL_PATH = "datus.api.services.visualization_service.VisualizationTool"
_TOOL_PROMPT_PATH = "datus.tools.llms_tools.visualization_tool.get_prompt_manager"


@pytest.fixture
def mock_agent_config():
    return Mock()


@pytest.fixture
def csv_data():
    return CsvData(
        columns=["date", "sales", "profit"],
        data=[
            {"date": "2024-01-01", "sales": 100, "profit": 20},
            {"date": "2024-01-02", "sales": 150, "profit": 35},
        ],
    )


def _mock_tool_result(success=True, chart_type="Line Chart", x_col="date", y_cols=None, reason="ok", error=None):
    result = Mock()
    result.success = success
    result.chart_type = chart_type
    result.x_col = x_col
    result.y_cols = ["sales"] if y_cols is None else y_cols
    result.reason = reason
    result.error = error
    return result


# ═══════════════════════════════════════════════════════════════════
# 1. Tool initialization
# ═══════════════════════════════════════════════════════════════════


class TestToolInit:
    def test_creates_tool_with_model(self, mock_agent_config):
        with patch(_LLM_PATH) as mock_llm:
            mock_llm.create_model.return_value = Mock()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            tool = svc._get_tool()
        assert tool is not None
        assert tool.model is not None

    def test_caches_tool_instance(self, mock_agent_config):
        with patch(_LLM_PATH) as mock_llm:
            mock_llm.create_model.return_value = Mock()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            tool1 = svc._get_tool()
            tool2 = svc._get_tool()
        assert tool1 is tool2

    def test_falls_back_when_model_fails(self, mock_agent_config):
        with patch(_LLM_PATH) as mock_llm:
            mock_llm.create_model.side_effect = Exception("no key")
            svc = DataVisualizationService(agent_config=mock_agent_config)
            tool = svc._get_tool()
        assert tool.model is None


# ═══════════════════════════════════════════════════════════════════
# 2. generate() without context — basic tool path
# ═══════════════════════════════════════════════════════════════════


class TestGenerateBasic:
    def test_returns_line_chart(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)

        assert result["success"] is True
        chart = result["data"]["chart"]
        assert chart["chart_type"] == "Line"
        assert chart["x_col"] == "date"
        assert chart["columns"] == ["date", "sales", "profit"]
        assert chart["numeric_columns"] == ["sales", "profit"]
        # No data_insight when sql not provided
        assert result["data"]["data_insight"] is None

    def test_returns_unknown_without_axes(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result(
                chart_type="Unknown", x_col="", y_cols=[], reason="Cannot determine"
            )
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)

        chart = result["data"]["chart"]
        assert chart["chart_type"] == "Unknown"
        assert chart["reason"] == "Cannot determine"
        assert "x_col" not in chart

    def test_caller_overrides_chart_type(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data, chart_type="Bar")

        assert result["data"]["chart"]["chart_type"] == "Bar"


# ═══════════════════════════════════════════════════════════════════
# 3. generate() with context — merged LLM call path
# ═══════════════════════════════════════════════════════════════════


class TestGenerateWithContext:
    def _mock_llm_response(self):
        return {
            "chart_type": "Bar Chart",
            "x_col": "date",
            "y_cols": ["sales", "profit"],
            "reason": "Categorical comparison",
            "period": "2024-01-01 ~ 2024-01-02",
            "filters": ["BP购买"],
            "insight": "Sales increased by 50% from day 1 to day 2.",
        }

    def _setup_context_svc(self, mock_agent_config, llm_response):
        """Create a service with real VisualizationTool helpers but mocked LLM."""
        from datus.tools.llms_tools.visualization_tool import VisualizationTool

        mock_model = Mock()
        mock_model.generate_with_json_output.return_value = llm_response

        with patch(_LLM_PATH) as mock_llm:
            mock_llm.create_model.return_value = mock_model
            svc = DataVisualizationService(agent_config=mock_agent_config)
            svc._tool = VisualizationTool(model=mock_model)
        return svc, mock_model

    def test_returns_chart_with_context_metadata(self, mock_agent_config, csv_data):
        svc, _ = self._setup_context_svc(mock_agent_config, self._mock_llm_response())
        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, sql="SELECT date, sales FROM t")

        assert result["success"] is True
        chart = result["data"]["chart"]
        assert chart["chart_type"] == "Bar"
        assert chart["x_col"] == "date"
        di = result["data"]["data_insight"]
        assert di["period"] == "2024-01-01 ~ 2024-01-02"
        assert di["filters"] == ["BP购买"]
        assert di["insight"] == "Sales increased by 50% from day 1 to day 2."

    def test_with_user_question_only(self, mock_agent_config, csv_data):
        svc, _ = self._setup_context_svc(mock_agent_config, self._mock_llm_response())
        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, user_question="Show me sales trends")

        assert result["success"] is True
        di = result["data"]["data_insight"]
        assert di["insight"] is not None

    def test_chart_type_override_with_context(self, mock_agent_config, csv_data):
        svc, _ = self._setup_context_svc(mock_agent_config, self._mock_llm_response())
        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, sql="SELECT ...", chart_type="Line")

        assert result["data"]["chart"]["chart_type"] == "Line"

    def test_falls_back_to_heuristics_on_llm_exception(self, mock_agent_config, csv_data):
        """LLM failure should fall back directly to heuristics, no second LLM call."""
        svc, mock_model = self._setup_context_svc(mock_agent_config, self._mock_llm_response())
        mock_model.generate_with_json_output.side_effect = Exception("LLM down")

        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, sql="SELECT ...")

        assert result["success"] is True
        # Should use heuristic result, not LLM
        chart = result["data"]["chart"]
        assert chart["chart_type"] in ("Bar", "Line", "Pie", "Scatter", "Unknown")
        # No second LLM call — only the one that failed
        assert mock_model.generate_with_json_output.call_count == 1

    def test_falls_back_to_heuristics_on_invalid_response(self, mock_agent_config, csv_data):
        svc, mock_model = self._setup_context_svc(mock_agent_config, self._mock_llm_response())
        mock_model.generate_with_json_output.return_value = "not a dict"

        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, sql="SELECT ...")

        assert result["success"] is True
        assert mock_model.generate_with_json_output.call_count == 1

    def test_falls_back_to_basic_when_no_model(self, mock_agent_config, csv_data):
        """When LLM model cannot be created, should use basic tool even with sql."""
        with patch(_LLM_PATH) as mock_llm, patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_llm.create_model.side_effect = Exception("no key")
            mock_cls.return_value.execute_with_context.return_value = _mock_tool_result()

            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data, sql="SELECT ...")

        assert result["success"] is True

    def test_sanitizes_invalid_metadata_types(self, mock_agent_config, csv_data):
        """LLM returning wrong types for metadata should be handled gracefully."""
        response = {
            "chart_type": "Bar Chart",
            "x_col": "date",
            "y_cols": ["sales"],
            "reason": "ok",
            "period": 12345,
            "filters": "not a list",
            "insight": ["not", "a", "string"],
        }
        svc, _ = self._setup_context_svc(mock_agent_config, response)
        with patch(_TOOL_PROMPT_PATH):
            result = svc.generate(csv_data, sql="SELECT ...")

        di = result["data"]["data_insight"]
        assert di["period"] is None
        assert di["filters"] == []
        assert di["insight"] is None


# ═══════════════════════════════════════════════════════════════════
# 4. generate() — errors
# ═══════════════════════════════════════════════════════════════════


class TestGenerateErrors:
    def test_empty_data(self, mock_agent_config):
        csv_data = CsvData(columns=[], data=[])
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH):
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)
        assert result["success"] is False
        assert result["errorCode"] == "EMPTY_DATA"

    def test_tool_exception(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.side_effect = Exception("boom")
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)
        assert result["success"] is False
        assert result["errorCode"] == "VISUALIZATION_FAILED"
        assert result["errorMessage"] == "Visualization analysis failed."

    def test_tool_returns_failure(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result(success=False, error="LLM unavailable")
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)
        assert result["success"] is False
        assert result["errorCode"] == "VISUALIZATION_FAILED"


# ═══════════════════════════════════════════════════════════════════
# 5. Caching
# ═══════════════════════════════════════════════════════════════════


class TestCaching:
    def test_same_input_returns_cached_result(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result1 = svc.generate(csv_data)
            result2 = svc.generate(csv_data)

        assert result1 is result2
        mock_cls.return_value.execute.assert_called_once()

    def test_different_sql_not_cached(self, mock_agent_config, csv_data):
        """Same csv_data but different sql should be separate cache entries."""
        with patch(_LLM_PATH) as mock_llm, patch(_VIZ_TOOL_PATH) as mock_cls, patch(_TOOL_PROMPT_PATH):
            mock_model = Mock()
            mock_llm.create_model.return_value = mock_model
            mock_cls.return_value.execute_with_context.return_value = _mock_tool_result()

            svc = DataVisualizationService(agent_config=mock_agent_config)
            svc.generate(csv_data, sql="SELECT a")
            svc.generate(csv_data, sql="SELECT b")

        assert mock_cls.return_value.execute_with_context.call_count == 2

    def test_evicts_lru_when_over_capacity(self, mock_agent_config):
        import datus.api.services.visualization_service as viz_mod

        original = viz_mod._MAX_CACHE_SIZE
        viz_mod._MAX_CACHE_SIZE = 2
        try:
            with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
                mock_cls.return_value.execute.return_value = _mock_tool_result()
                svc = DataVisualizationService(agent_config=mock_agent_config)

                data_a = CsvData(columns=["a", "v"], data=[{"a": 1, "v": 2}])
                data_b = CsvData(columns=["b", "v"], data=[{"b": 1, "v": 2}])
                data_c = CsvData(columns=["c", "v"], data=[{"c": 1, "v": 2}])

                svc.generate(data_a)
                svc.generate(data_b)
                assert len(svc._cache) == 2

                svc.generate(data_a)
                assert mock_cls.return_value.execute.call_count == 2

                svc.generate(data_c)
                assert len(svc._cache) == 2

                svc.generate(data_a)
                assert mock_cls.return_value.execute.call_count == 3

                svc.generate(data_b)
                assert mock_cls.return_value.execute.call_count == 4
        finally:
            viz_mod._MAX_CACHE_SIZE = original
