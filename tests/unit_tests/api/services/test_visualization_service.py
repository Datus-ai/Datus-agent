# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/api/services/visualization_service.py — CI level, zero external deps."""

from unittest.mock import Mock, patch

import pytest

from datus.api.models.visualization_models import CsvData
from datus.api.services.visualization_service import DataVisualizationService

_LLM_PATH = "datus.api.services.visualization_service.LLMBaseModel"
_VIZ_TOOL_PATH = "datus.api.services.visualization_service.VisualizationTool"


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
    result.y_cols = y_cols or ["sales"]
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
# 2. generate() — success
# ═══════════════════════════════════════════════════════════════════


class TestGenerateSuccess:
    def test_returns_line_chart(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)

        assert result["success"] is True
        chart = result["data"]["data"]
        assert chart["chart_type"] == "Line"
        assert chart["x_col"] == "date"
        assert chart["columns"] == ["date", "sales", "profit"]
        assert chart["numeric_columns"] == ["sales", "profit"]

    def test_returns_unknown_without_axes(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result(
                chart_type="Unknown", x_col="", y_cols=[], reason="Cannot determine"
            )
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)

        chart = result["data"]["data"]
        assert chart["chart_type"] == "Unknown"
        assert chart["reason"] == "Cannot determine"
        assert "x_col" not in chart
        # columns metadata still present for Unknown
        assert chart["columns"] == ["date", "sales", "profit"]
        assert chart["numeric_columns"] == ["sales", "profit"]

    def test_caller_overrides_chart_type(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data, chart_type="Bar")

        assert result["data"]["data"]["chart_type"] == "Bar"


# ═══════════════════════════════════════════════════════════════════
# 3. generate() — errors
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
            mock_cls.return_value.execute.return_value = _mock_tool_result(
                success=False, error="LLM unavailable"
            )
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result = svc.generate(csv_data)
        assert result["success"] is False
        assert result["errorCode"] == "VISUALIZATION_FAILED"


# ═══════════════════════════════════════════════════════════════════
# 4. Caching
# ═══════════════════════════════════════════════════════════════════


class TestCaching:
    def test_same_input_returns_cached_result(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result1 = svc.generate(csv_data)
            result2 = svc.generate(csv_data)

        assert result1 is result2
        # Tool should only be called once
        mock_cls.return_value.execute.assert_called_once()

    def test_different_chart_type_not_cached(self, mock_agent_config, csv_data):
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            svc.generate(csv_data, chart_type=None)
            svc.generate(csv_data, chart_type="Bar")

        assert mock_cls.return_value.execute.call_count == 2

    def test_different_data_not_cached(self, mock_agent_config, csv_data):
        csv_data2 = CsvData(
            columns=["x", "y"],
            data=[{"x": 1, "y": 2}],
        )
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH) as mock_cls:
            mock_cls.return_value.execute.return_value = _mock_tool_result()
            svc = DataVisualizationService(agent_config=mock_agent_config)
            svc.generate(csv_data)
            svc.generate(csv_data2)

        assert mock_cls.return_value.execute.call_count == 2

    def test_error_result_is_also_cached(self, mock_agent_config):
        csv_data = CsvData(columns=[], data=[])
        with patch(_LLM_PATH), patch(_VIZ_TOOL_PATH):
            svc = DataVisualizationService(agent_config=mock_agent_config)
            result1 = svc.generate(csv_data)
            result2 = svc.generate(csv_data)
        assert result1 is result2
        assert result1["success"] is False

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

                svc.generate(data_a)  # cache: [a]
                svc.generate(data_b)  # cache: [a, b]
                assert len(svc._cache) == 2

                # Access data_a again to promote it (LRU: b is now oldest)
                svc.generate(data_a)  # cache: [b, a] — cache hit, no tool call
                assert mock_cls.return_value.execute.call_count == 2

                # Insert data_c: should evict data_b (LRU), not data_a
                svc.generate(data_c)  # cache: [a, c]
                assert len(svc._cache) == 2

                # data_a should still be cached (not evicted)
                svc.generate(data_a)  # cache hit
                assert mock_cls.return_value.execute.call_count == 3

                # data_b was evicted, re-generating it calls tool again
                svc.generate(data_b)  # cache miss
                assert mock_cls.return_value.execute.call_count == 4
        finally:
            viz_mod._MAX_CACHE_SIZE = original
