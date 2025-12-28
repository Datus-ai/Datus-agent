# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for DimensionAttributionUtil
"""

from unittest.mock import Mock

import pytest

from datus.tools.func_tool.attribution_utils import AttributionAnalysisResult, DimensionAttributionUtil
from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.tools.semantic_tools.models import QueryResult, TimeRange


class MockAdapter(BaseSemanticAdapter):
    """Mock semantic adapter for testing."""

    def __init__(self):
        config = Mock()
        config.namespace = "test"
        super().__init__(config, service_type="mock")
        self.query_results = []
        self.call_count = 0

    async def query_metrics(self, metrics, dimensions=None, **kwargs):
        """Mock query_metrics implementation."""
        self.call_count += 1
        if self.query_results:
            return self.query_results.pop(0)
        # Default response
        return QueryResult(
            columns=metrics + (dimensions or []),
            data=[],
            metadata={},
        )

    async def get_dimensions(self, metric_name, path=None):
        """Mock get_dimensions implementation."""
        return ["region", "country", "product_category"]

    async def list_metrics(self, path=None, limit=100, offset=0):
        """Mock list_metrics implementation."""
        return []

    async def validate_semantic(self):
        """Mock validate_semantic implementation."""
        from datus.tools.semantic_tools.models import ValidationResult

        return ValidationResult(valid=True, issues=[])


@pytest.fixture
def mock_adapter():
    """Create mock adapter."""
    return MockAdapter()


@pytest.fixture
def attribution_tool(mock_adapter):
    """Create attribution tool with mock adapter."""
    return DimensionAttributionUtil(mock_adapter)


class TestDimensionAttributionUtil:
    """Tests for DimensionAttributionUtil."""

    @pytest.mark.asyncio
    async def test_extract_metric_value_handles_invalid_data(self, attribution_tool):
        """Test that metric value extraction handles invalid data gracefully."""
        assert attribution_tool._extract_metric_value({"revenue": "invalid"}, "revenue") == 0.0
        assert attribution_tool._extract_metric_value({"revenue": None}, "revenue") == 0.0
        assert attribution_tool._extract_metric_value({}, "revenue") == 0.0

    @pytest.mark.asyncio
    async def test_extract_dimension_value_handles_missing(self, attribution_tool):
        """Test that dimension value extraction handles missing values."""
        assert attribution_tool._extract_dimension_value({}, "region") == "Unknown"
        assert attribution_tool._extract_dimension_value({"region": None}, "region") == "None"

    @pytest.mark.asyncio
    async def test_attribution_analyze_basic(self, attribution_tool, mock_adapter):
        """Test unified attribution_analyze method."""
        # Setup mock data for ranking dimensions
        mock_adapter.query_results = [
            # Baseline - project_title
            QueryResult(
                columns=["payment_amount", "project_title"],
                data=[
                    {"payment_amount": 10000.0, "project_title": "Project A"},
                    {"payment_amount": 8000.0, "project_title": "Project B"},
                ],
                metadata={},
            ),
            # Current - project_title
            QueryResult(
                columns=["payment_amount", "project_title"],
                data=[
                    {"payment_amount": 27000.0, "project_title": "Project A"},
                    {"payment_amount": 8500.0, "project_title": "Project B"},
                ],
                metadata={},
            ),
            # Baseline - project_type
            QueryResult(
                columns=["payment_amount", "project_type"],
                data=[
                    {"payment_amount": 12000.0, "project_type": "Regular"},
                    {"payment_amount": 6000.0, "project_type": "Premium"},
                ],
                metadata={},
            ),
            # Current - project_type
            QueryResult(
                columns=["payment_amount", "project_type"],
                data=[
                    {"payment_amount": 20000.0, "project_type": "Regular"},
                    {"payment_amount": 15500.0, "project_type": "Premium"},
                ],
                metadata={},
            ),
            # Baseline - selected dimension (project_title)
            QueryResult(
                columns=["payment_amount", "project_title"],
                data=[
                    {"payment_amount": 10000.0, "project_title": "Project A"},
                    {"payment_amount": 8000.0, "project_title": "Project B"},
                ],
                metadata={},
            ),
            # Current - selected dimension (project_title)
            QueryResult(
                columns=["payment_amount", "project_title"],
                data=[
                    {"payment_amount": 27000.0, "project_title": "Project A"},
                    {"payment_amount": 8500.0, "project_title": "Project B"},
                ],
                metadata={},
            ),
        ]

        result = await attribution_tool.attribution_analyze(
            metric_name="payment_amount",
            candidate_dimensions=["project_title", "project_type"],
            baseline_period=TimeRange(start="2026-01-01", end="2026-01-01"),
            current_period=TimeRange(start="2026-01-08", end="2026-01-08"),
            max_selected_dimensions=1,
            top_n_values=10,
        )

        assert isinstance(result, AttributionAnalysisResult)
        assert result.metric_name == "payment_amount"
        assert result.candidate_dimensions == ["project_title", "project_type"]
        assert len(result.dimension_ranking) == 2
        assert len(result.selected_dimensions) <= 1
        assert len(result.top_dimension_values) > 0

    @pytest.mark.asyncio
    async def test_attribution_analyze_with_anomaly_context(self, attribution_tool, mock_adapter):
        """Test attribution_analyze with anomaly context."""
        # Setup mock data
        mock_adapter.query_results = [
            # Baseline - region
            QueryResult(
                columns=["revenue", "region"],
                data=[
                    {"revenue": 5000.0, "region": "North America"},
                    {"revenue": 3000.0, "region": "Europe"},
                ],
                metadata={},
            ),
            # Current - region
            QueryResult(
                columns=["revenue", "region"],
                data=[
                    {"revenue": 9000.0, "region": "North America"},
                    {"revenue": 3100.0, "region": "Europe"},
                ],
                metadata={},
            ),
            # Baseline - selected dimension (region)
            QueryResult(
                columns=["revenue", "region"],
                data=[
                    {"revenue": 5000.0, "region": "North America"},
                    {"revenue": 3000.0, "region": "Europe"},
                ],
                metadata={},
            ),
            # Current - selected dimension (region)
            QueryResult(
                columns=["revenue", "region"],
                data=[
                    {"revenue": 9000.0, "region": "North America"},
                    {"revenue": 3100.0, "region": "Europe"},
                ],
                metadata={},
            ),
        ]

        anomaly_context = {
            "rule": "wow_growth_gt_20pct",
            "observed_change_pct": 0.5,
        }

        result = await attribution_tool.attribution_analyze(
            metric_name="revenue",
            candidate_dimensions=["region"],
            baseline_period=TimeRange(start="2026-01-01", end="2026-01-01"),
            current_period=TimeRange(start="2026-01-08", end="2026-01-08"),
            anomaly_context=anomaly_context,
            max_selected_dimensions=1,
        )

        assert result.anomaly_context == anomaly_context
        assert result.anomaly_context["rule"] == "wow_growth_gt_20pct"

    @pytest.mark.asyncio
    async def test_attribution_analyze_multiple_dimensions(self, attribution_tool, mock_adapter):
        """Test attribution_analyze with multiple dimensions selected."""
        # Setup mock data for 3 dimensions
        mock_adapter.query_results = [
            # Baseline - dim1
            QueryResult(
                columns=["metric", "dim1"],
                data=[{"metric": 100.0, "dim1": "A"}, {"metric": 50.0, "dim1": "B"}],
                metadata={},
            ),
            # Current - dim1
            QueryResult(
                columns=["metric", "dim1"],
                data=[{"metric": 300.0, "dim1": "A"}, {"metric": 60.0, "dim1": "B"}],
                metadata={},
            ),
            # Baseline - dim2
            QueryResult(
                columns=["metric", "dim2"],
                data=[{"metric": 80.0, "dim2": "X"}, {"metric": 70.0, "dim2": "Y"}],
                metadata={},
            ),
            # Current - dim2
            QueryResult(
                columns=["metric", "dim2"],
                data=[{"metric": 200.0, "dim2": "X"}, {"metric": 160.0, "dim2": "Y"}],
                metadata={},
            ),
            # Baseline - dim3
            QueryResult(
                columns=["metric", "dim3"],
                data=[{"metric": 75.0, "dim3": "P"}, {"metric": 75.0, "dim3": "Q"}],
                metadata={},
            ),
            # Current - dim3
            QueryResult(
                columns=["metric", "dim3"],
                data=[{"metric": 180.0, "dim3": "P"}, {"metric": 180.0, "dim3": "Q"}],
                metadata={},
            ),
            # For top 2 dimensions (dim1, dim2) - baseline
            QueryResult(
                columns=["metric", "dim1", "dim2"],
                data=[
                    {"metric": 50.0, "dim1": "A", "dim2": "X"},
                    {"metric": 40.0, "dim1": "A", "dim2": "Y"},
                    {"metric": 30.0, "dim1": "B", "dim2": "X"},
                    {"metric": 30.0, "dim1": "B", "dim2": "Y"},
                ],
                metadata={},
            ),
            # For top 2 dimensions (dim1, dim2) - current
            QueryResult(
                columns=["metric", "dim1", "dim2"],
                data=[
                    {"metric": 180.0, "dim1": "A", "dim2": "X"},
                    {"metric": 120.0, "dim1": "A", "dim2": "Y"},
                    {"metric": 35.0, "dim1": "B", "dim2": "X"},
                    {"metric": 25.0, "dim1": "B", "dim2": "Y"},
                ],
                metadata={},
            ),
        ]

        result = await attribution_tool.attribution_analyze(
            metric_name="metric",
            candidate_dimensions=["dim1", "dim2", "dim3"],
            baseline_period=TimeRange(start="2026-01-01", end="2026-01-01"),
            current_period=TimeRange(start="2026-01-08", end="2026-01-08"),
            max_selected_dimensions=2,
        )

        assert len(result.dimension_ranking) == 3
        assert len(result.selected_dimensions) <= 2
