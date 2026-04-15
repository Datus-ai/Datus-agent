# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for GenDashboardAgenticNode schema models.
"""

import pytest
from pydantic import ValidationError

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.gen_dashboard_agentic_node_models import GenDashboardNodeInput, GenDashboardNodeResult


class TestGenDashboardNodeInput:
    """Tests for GenDashboardNodeInput model."""

    def test_inherits_from_base_input(self):
        """GenDashboardNodeInput should inherit from BaseInput."""
        assert issubclass(GenDashboardNodeInput, BaseInput)

    def test_required_user_message(self):
        """user_message is required."""
        with pytest.raises(ValidationError):
            GenDashboardNodeInput()

    def test_minimal_creation(self):
        """Create with only required field."""
        inp = GenDashboardNodeInput(user_message="Create a sales dashboard")
        assert inp.user_message == "Create a sales dashboard"
        assert inp.database is None
        assert inp.prompt_version is None

    def test_full_creation(self):
        """Create with all fields."""
        inp = GenDashboardNodeInput(
            user_message="List dashboards",
            database="superset_db",
            prompt_version="1.0",
        )
        assert inp.user_message == "List dashboards"
        assert inp.database == "superset_db"
        assert inp.prompt_version == "1.0"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        inp = GenDashboardNodeInput(user_message="test", database="mydb", prompt_version="1.0")
        d = inp.model_dump()
        assert d["user_message"] == "test"
        assert d["database"] == "mydb"
        assert d["prompt_version"] == "1.0"

        restored = GenDashboardNodeInput(**d)
        assert restored.user_message == "test"
        assert restored.database == "mydb"

    def test_extra_fields_forbidden(self):
        """Extra fields should be rejected (BaseInput uses extra='forbid')."""
        with pytest.raises(ValidationError):
            GenDashboardNodeInput(user_message="test", unknown_field="value")


class TestGenDashboardNodeResult:
    """Tests for GenDashboardNodeResult model."""

    def test_inherits_from_base_result(self):
        """GenDashboardNodeResult should inherit from BaseResult."""
        assert issubclass(GenDashboardNodeResult, BaseResult)

    def test_success_result(self):
        """Create a successful result."""
        result = GenDashboardNodeResult(
            success=True,
            response="Dashboard created successfully",
            tokens_used=150,
        )
        assert result.success is True
        assert result.response == "Dashboard created successfully"
        assert result.tokens_used == 150
        assert result.error is None
        assert result.dashboard_result is None

    def test_default_values(self):
        """Default values should be applied."""
        result = GenDashboardNodeResult(success=True)
        assert result.response == ""
        assert result.tokens_used == 0
        assert result.dashboard_result is None

    def test_with_dashboard_result(self):
        """Create result with structured dashboard data."""
        dashboard_data = {
            "dashboard_id": "42",
            "dashboard_url": "http://superset:8088/superset/dashboard/42/",
            "charts_created": 3,
        }
        result = GenDashboardNodeResult(
            success=True,
            response="Created dashboard with 3 charts",
            dashboard_result=dashboard_data,
            tokens_used=500,
        )
        assert result.dashboard_result["dashboard_id"] == "42"
        assert result.dashboard_result["charts_created"] == 3

    def test_error_result(self):
        """Create a failed result."""
        result = GenDashboardNodeResult(
            success=False,
            error="BI platform connection failed",
            response="Sorry, I encountered an error.",
            tokens_used=0,
        )
        assert result.success is False
        assert result.error == "BI platform connection failed"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        result = GenDashboardNodeResult(
            success=True,
            response="Dashboard listed",
            dashboard_result={"count": 5},
            tokens_used=200,
        )
        d = result.model_dump()
        assert d["success"] is True
        assert d["response"] == "Dashboard listed"
        assert d["dashboard_result"] == {"count": 5}
        assert d["tokens_used"] == 200

        restored = GenDashboardNodeResult(**d)
        assert restored.response == "Dashboard listed"
        assert restored.dashboard_result == {"count": 5}
