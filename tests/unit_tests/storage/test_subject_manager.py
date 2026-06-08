# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.subject_manager."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.subject_manager import SubjectUpdater


def _build_updater() -> SubjectUpdater:
    """Create a SubjectUpdater with all storage dependencies mocked."""
    mock_config = MagicMock()
    mock_config.current_datasource = "test_ds"
    mock_config.project_name = "test_project"
    mock_storage = MagicMock()
    with patch("datus.storage.subject_manager.get_storage", return_value=mock_storage):
        updater = SubjectUpdater(mock_config)
    return updater


# ---------------------------------------------------------------------------
# update_metrics_detail
# ---------------------------------------------------------------------------


class TestUpdateMetricsDetail:
    """Tests for SubjectUpdater.update_metrics_detail."""

    @pytest.mark.ci
    def test_empty_update_values_returns_early(self):
        """Empty update_values should return without calling storage."""
        updater = _build_updater()
        updater.update_metrics_detail(["Finance"], "revenue", {})
        updater.metrics_storage.update_entry.assert_not_called()

    @pytest.mark.ci
    def test_non_empty_update_values_calls_storage(self):
        """Non-empty update_values should delegate to metrics_storage.update_entry."""
        updater = _build_updater()
        updater.update_metrics_detail(["Finance", "Revenue"], "dau", {"description": "daily active users"})
        updater.metrics_storage.update_entry.assert_called_once_with(
            ["Finance", "Revenue"],
            "dau",
            {"description": "daily active users"},
        )


# ---------------------------------------------------------------------------
# update_historical_sql
# ---------------------------------------------------------------------------


class TestUpdateHistoricalSql:
    """Tests for SubjectUpdater.update_historical_sql."""

    @pytest.mark.ci
    def test_empty_update_values_returns_early(self):
        """Empty update_values should return without calling storage."""
        updater = _build_updater()
        updater.update_historical_sql(["Analytics"], "query_1", {})
        updater.reference_sql_storage.update_entry.assert_not_called()

    @pytest.mark.ci
    def test_non_empty_update_values_calls_storage(self):
        """Non-empty update_values should delegate to reference_sql_storage.update_entry."""
        updater = _build_updater()
        updater.update_historical_sql(["Analytics"], "query_1", {"sql": "SELECT 1"})
        updater.reference_sql_storage.update_entry.assert_called_once_with(
            ["Analytics"],
            "query_1",
            {"sql": "SELECT 1"},
        )


# ---------------------------------------------------------------------------
# delete_metric
# ---------------------------------------------------------------------------


class TestDeleteMetric:
    """Tests for SubjectUpdater.delete_metric."""

    @pytest.mark.ci
    def test_delete_metric_delegates_to_storage(self):
        """Should delegate to metrics_storage.delete_metric and return its result."""
        updater = _build_updater()
        updater.metrics_storage.delete_metric.return_value = {"success": True, "message": "deleted"}
        result = updater.delete_metric(["Finance"], "old_metric")
        updater.metrics_storage.delete_metric.assert_called_once_with(
            ["Finance"],
            "old_metric",
        )
        assert result == {"success": True, "message": "deleted"}


# ---------------------------------------------------------------------------
# delete_reference_sql
# ---------------------------------------------------------------------------


class TestDeleteReferenceSql:
    """Tests for SubjectUpdater.delete_reference_sql."""

    @pytest.mark.ci
    def test_delete_reference_sql_delegates_to_storage(self):
        """Should delegate to reference_sql_storage.delete_reference_sql."""
        updater = _build_updater()
        updater.reference_sql_storage.delete_reference_sql.return_value = True
        result = updater.delete_reference_sql(["Analytics", "Reports"], "old_query")
        updater.reference_sql_storage.delete_reference_sql.assert_called_once_with(
            ["Analytics", "Reports"],
            "old_query",
        )
        assert result is True

    @pytest.mark.ci
    def test_delete_reference_sql_not_found(self):
        """Should return False when the entry is not found."""
        updater = _build_updater()
        updater.reference_sql_storage.delete_reference_sql.return_value = False
        result = updater.delete_reference_sql(["Analytics"], "nonexistent")
        assert result is False
