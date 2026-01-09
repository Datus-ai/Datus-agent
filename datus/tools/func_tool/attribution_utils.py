# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Dimension Attribution Analysis Tool

Provides adapter-agnostic dimension attribution analysis capabilities.
Only depends on BaseSemanticAdapter abstract interface.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from datus.tools.semantic_tools.base import BaseSemanticAdapter
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# ==================== Data Models ====================


class DimensionRanking(BaseModel):
    """Ranking score for a dimension's explanatory power."""

    dimension: str = Field(..., description="Dimension name")
    score: float = Field(..., description="Importance score (0-1)")


class DimensionValueContribution(BaseModel):
    """Delta contribution of a dimension value."""

    dimension_values: Dict[str, str] = Field(..., description="Dimension value(s)")
    baseline: float = Field(..., description="Baseline period metric value")
    current: float = Field(..., description="Current period metric value")
    delta: float = Field(..., description="Absolute change (current - baseline)")
    contribution_pct_of_total_delta: float = Field(..., description="Percentage contribution to total delta")


class AttributionAnalysisResult(BaseModel):
    """Result of unified attribution analysis."""

    metric_name: str = Field(..., description="Metric being analyzed")
    candidate_dimensions: List[str] = Field(..., description="Input candidate dimensions")
    dimension_ranking: List[DimensionRanking] = Field(..., description="Dimensions ranked by importance")
    selected_dimensions: List[str] = Field(..., description="Dimensions selected for analysis")
    top_dimension_values: List[DimensionValueContribution] = Field(
        ..., description="Top contributors by dimension values"
    )
    anomaly_context: Optional[Dict] = Field(None, description="Anomaly detection context")
    comparison_metadata: Dict = Field(..., description="Comparison period metadata")


# ==================== Attribution Util ====================


class DimensionAttributionUtil:
    """
    Adapter-agnostic dimension attribution analysis utility.

    Only depends on BaseSemanticAdapter abstract interface:
    - get_dimensions()
    - query_metrics()

    Works with any semantic layer backend (MetricFlow, dbt, Cube, etc.)
    """

    def __init__(self, adapter: BaseSemanticAdapter):
        self.adapter = adapter

    async def attribution_analyze(
        self,
        metric_name: str,
        candidate_dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        path: Optional[List[str]] = None,
        anomaly_context: Optional[Dict] = None,
        max_selected_dimensions: int = 3,
        top_n_values: int = 10,
    ) -> AttributionAnalysisResult:
        """
        Unified attribution analysis: ranks dimensions and calculates delta contributions.

        This method:
        1. Evaluates all candidate dimensions for explanatory power
        2. Selects top dimensions (up to max_selected_dimensions)
        3. Calculates delta attribution for selected dimension values

        Args:
            metric_name: Metric to analyze
            candidate_dimensions: List of dimensions to evaluate (1-N dimensions)
            baseline_start: Baseline period start date (e.g., "2026-01-01")
            baseline_end: Baseline period end date (e.g., "2026-01-07")
            current_start: Current period start date (e.g., "2026-01-08")
            current_end: Current period end date (e.g., "2026-01-14")
            path: Metric path for scoping
            anomaly_context: Optional anomaly detection context (rule, observed_change, etc.)
            max_selected_dimensions: Maximum number of dimensions to select (default: 3)
            top_n_values: Number of top dimension values to return

        Returns:
            AttributionAnalysisResult with:
            - dimension_ranking: All dimensions ranked by importance score
            - selected_dimensions: Top dimensions selected for analysis
            - top_dimension_values: Delta contributions of top dimension values

        Example:
            result = await tool.attribution_analyze(
                metric_name="payment_amount",
                candidate_dimensions=["project_title", "project_type", "region"],
                baseline_start="2026-01-01",
                baseline_end="2026-01-01",
                current_start="2026-01-08",
                current_end="2026-01-08",
                anomaly_context={
                    "rule": "wow_growth_gt_20pct",
                    "observed_change_pct": 0.8017
                }
            )
        """
        # Step 1: Rank dimensions by importance
        dimension_rankings = await self._rank_dimensions(
            metric_name=metric_name,
            candidate_dimensions=candidate_dimensions,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
            path=path,
        )

        # Step 2: Select top dimensions
        selected_dimensions = [ranking.dimension for ranking in dimension_rankings[:max_selected_dimensions]]

        # Step 3: Calculate delta contributions for selected dimensions
        top_dimension_values = await self._calculate_delta_contributions(
            metric_name=metric_name,
            dimensions=selected_dimensions,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            current_start=current_start,
            current_end=current_end,
            path=path,
            top_n=top_n_values,
        )

        return AttributionAnalysisResult(
            metric_name=metric_name,
            candidate_dimensions=candidate_dimensions,
            dimension_ranking=dimension_rankings,
            selected_dimensions=selected_dimensions,
            top_dimension_values=top_dimension_values,
            anomaly_context=anomaly_context,
            comparison_metadata={
                "baseline": {"start": baseline_start, "end": baseline_end},
                "current": {"start": current_start, "end": current_end},
            },
        )

    async def _rank_dimensions(
        self,
        metric_name: str,
        candidate_dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        path: Optional[List[str]] = None,
    ) -> List[DimensionRanking]:
        """
        Rank dimensions by their explanatory power for the metric change.

        Uses variance-based importance: dimensions with higher variance in delta
        contributions have more explanatory power.
        """
        import math

        rankings = []

        for dimension in candidate_dimensions:
            # Query both periods
            baseline_result = await self.adapter.query_metrics(
                metrics=[metric_name],
                dimensions=[dimension],
                path=path,
                time_start=baseline_start,
                time_end=baseline_end,
            )

            current_result = await self.adapter.query_metrics(
                metrics=[metric_name],
                dimensions=[dimension],
                path=path,
                time_start=current_start,
                time_end=current_end,
            )

            logger.debug(
                f"Ranking dimension '{dimension}': baseline={len(baseline_result.data)} rows, "
                f"current={len(current_result.data)} rows"
            )

            # Build lookup for baseline values
            baseline_lookup = {}
            for row in baseline_result.data:
                dim_val = self._extract_dimension_value(row, dimension)
                metric_val = self._extract_metric_value(row, metric_name)
                baseline_lookup[dim_val] = metric_val

            # Calculate deltas and variance
            deltas = []
            for row in current_result.data:
                dim_val = self._extract_dimension_value(row, dimension)
                current_val = self._extract_metric_value(row, metric_name)
                baseline_val = baseline_lookup.get(dim_val, 0.0)
                delta = current_val - baseline_val
                deltas.append(delta)

            # Precompute current dimension values for O(1) lookup
            current_dim_vals = {self._extract_dimension_value(row, dimension) for row in current_result.data}

            # Also check values that disappeared (exist in baseline but not in current)
            for dim_val, baseline_val in baseline_lookup.items():
                if dim_val not in current_dim_vals:
                    delta = 0.0 - baseline_val
                    deltas.append(delta)

            # Calculate variance as importance score
            if len(deltas) > 1:
                mean_delta = sum(deltas) / len(deltas)
                variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
                std_dev = math.sqrt(variance)

                # Normalize score using coefficient of variation
                # Higher variance relative to mean = more explanatory power
                if abs(mean_delta) > 0:
                    score = min(1.0, std_dev / abs(mean_delta))
                else:
                    score = 0.5 if std_dev > 0 else 0.0
            else:
                score = 0.0

            rankings.append(DimensionRanking(dimension=dimension, score=score))

        # Sort by score descending
        rankings.sort(key=lambda r: r.score, reverse=True)

        return rankings

    async def _calculate_delta_contributions(
        self,
        metric_name: str,
        dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        path: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> List[DimensionValueContribution]:
        """
        Calculate delta contributions for dimension values.

        For single dimension: returns top N values by absolute delta
        For multiple dimensions: returns top N dimension combinations
        """
        if not dimensions:
            return []

        # Query both periods with dimension breakdown
        baseline_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=dimensions,
            path=path,
            time_start=baseline_start,
            time_end=baseline_end,
        )

        current_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=dimensions,
            path=path,
            time_start=current_start,
            time_end=current_end,
        )

        # Build lookup for baseline values
        baseline_lookup = {}
        for row in baseline_result.data:
            dim_key = tuple(self._extract_dimension_value(row, dim) for dim in dimensions)
            metric_val = self._extract_metric_value(row, metric_name)
            baseline_lookup[dim_key] = metric_val

        # Calculate total delta
        baseline_total = sum(baseline_lookup.values())
        current_total = sum(self._extract_metric_value(row, metric_name) for row in current_result.data)
        total_delta = current_total - baseline_total

        # Calculate contributions
        contributions = []

        for row in current_result.data:
            dim_key = tuple(self._extract_dimension_value(row, dim) for dim in dimensions)
            current_val = self._extract_metric_value(row, metric_name)
            baseline_val = baseline_lookup.get(dim_key, 0.0)
            delta = current_val - baseline_val

            # Build dimension_values dict
            dimension_values = {dim: self._extract_dimension_value(row, dim) for dim in dimensions}

            # Calculate contribution percentage
            contribution_pct = (delta / total_delta * 100) if total_delta != 0 else 0.0

            contributions.append(
                DimensionValueContribution(
                    dimension_values=dimension_values,
                    baseline=baseline_val,
                    current=current_val,
                    delta=delta,
                    contribution_pct_of_total_delta=contribution_pct,
                )
            )

        # Precompute current dimension keys for O(1) lookup
        current_dim_keys = {
            tuple(self._extract_dimension_value(row, dim) for dim in dimensions) for row in current_result.data
        }

        # Also include values that disappeared (exist in baseline but not in current)
        for dim_key, baseline_val in baseline_lookup.items():
            if dim_key not in current_dim_keys:
                delta = 0.0 - baseline_val
                dimension_values = {dim: dim_key[i] for i, dim in enumerate(dimensions)}
                contribution_pct = (delta / total_delta * 100) if total_delta != 0 else 0.0

                contributions.append(
                    DimensionValueContribution(
                        dimension_values=dimension_values,
                        baseline=baseline_val,
                        current=0.0,
                        delta=delta,
                        contribution_pct_of_total_delta=contribution_pct,
                    )
                )

        # Sort by absolute delta descending
        contributions.sort(key=lambda c: abs(c.delta), reverse=True)

        return contributions[:top_n]

    def _extract_metric_value(self, row: Dict, metric_name: str) -> float:
        """Extract metric value from query result row."""
        value = row.get(metric_name, 0)
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _extract_dimension_value(self, row: Dict, dimension: str) -> str:
        """Extract dimension value from query result row."""
        value = row.get(dimension, "Unknown")
        return str(value)
