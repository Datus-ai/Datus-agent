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
    score: float = Field(
        ..., description="Max contribution ratio (max_abs_delta / total_delta), can exceed 1 when deltas offset"
    )


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
        1. Queries total metric values for baseline and current periods
        2. Evaluates all candidate dimensions (single query per dimension)
        3. Calculates both dimension scores and value contributions in one pass
        4. Selects top dimensions and returns top contributing values

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
        """
        # Step 1: Query total metric values (no dimension breakdown)
        baseline_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=path,
            time_start=baseline_start,
            time_end=baseline_end,
        )
        current_total_result = await self.adapter.query_metrics(
            metrics=[metric_name],
            dimensions=[],
            path=path,
            time_start=current_start,
            time_end=current_end,
        )

        baseline_total = (
            self._extract_metric_value(baseline_total_result.data[0], metric_name)
            if baseline_total_result.data
            else 0.0
        )
        current_total = (
            self._extract_metric_value(current_total_result.data[0], metric_name) if current_total_result.data else 0.0
        )
        total_delta = current_total - baseline_total

        # Step 2: Analyze each dimension (one query per dimension, calculate both score and contributions)
        dimension_rankings: List[DimensionRanking] = []
        all_contributions: Dict[str, List[DimensionValueContribution]] = {}

        for dimension in candidate_dimensions:
            # Query both periods for this dimension
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
                f"Analyzing dimension '{dimension}': baseline={len(baseline_result.data)} rows, "
                f"current={len(current_result.data)} rows"
            )

            # Build lookup for baseline values
            baseline_lookup = {}
            for row in baseline_result.data:
                dim_val = self._extract_dimension_value(row, dimension)
                metric_val = self._extract_metric_value(row, metric_name)
                baseline_lookup[dim_val] = metric_val

            # Calculate contributions for current values
            contributions = []
            deltas = []

            for row in current_result.data:
                dim_val = self._extract_dimension_value(row, dimension)
                current_val = self._extract_metric_value(row, metric_name)
                baseline_val = baseline_lookup.get(dim_val, 0.0)
                delta = current_val - baseline_val
                deltas.append(delta)

                contribution_pct = (delta / total_delta * 100) if total_delta != 0 else 0.0
                contributions.append(
                    DimensionValueContribution(
                        dimension_values={dimension: dim_val},
                        baseline=baseline_val,
                        current=current_val,
                        delta=delta,
                        contribution_pct_of_total_delta=contribution_pct,
                    )
                )

            # Also include values that disappeared (exist in baseline but not in current)
            current_dim_vals = {self._extract_dimension_value(row, dimension) for row in current_result.data}
            for dim_val, baseline_val in baseline_lookup.items():
                if dim_val not in current_dim_vals:
                    delta = 0.0 - baseline_val
                    deltas.append(delta)

                    contribution_pct = (delta / total_delta * 100) if total_delta != 0 else 0.0
                    contributions.append(
                        DimensionValueContribution(
                            dimension_values={dimension: dim_val},
                            baseline=baseline_val,
                            current=0.0,
                            delta=delta,
                            contribution_pct_of_total_delta=contribution_pct,
                        )
                    )

            # Calculate dimension score (max contribution ratio)
            if len(deltas) > 0 and abs(total_delta) > 0:
                max_abs_delta = max(abs(d) for d in deltas)
                score = max_abs_delta / abs(total_delta)
            else:
                score = 0.0

            dimension_rankings.append(DimensionRanking(dimension=dimension, score=score))
            all_contributions[dimension] = contributions

        # Step 3: Sort dimensions by score and select top N
        dimension_rankings.sort(key=lambda r: r.score, reverse=True)
        selected_dimensions = [ranking.dimension for ranking in dimension_rankings[:max_selected_dimensions]]

        # Step 4: Collect contributions from selected dimensions and sort by contribution percentage
        selected_contributions = []
        for dim in selected_dimensions:
            selected_contributions.extend(all_contributions[dim])

        selected_contributions.sort(key=lambda c: c.contribution_pct_of_total_delta, reverse=True)
        top_dimension_values = selected_contributions[:top_n_values]

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
