# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SchedulerAgenticNode schema models.
"""

import pytest
from pydantic import ValidationError

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.scheduler_agentic_node_models import SchedulerNodeInput, SchedulerNodeResult


class TestSchedulerNodeInput:
    """Tests for SchedulerNodeInput model."""

    def test_inherits_from_base_input(self):
        """SchedulerNodeInput should inherit from BaseInput."""
        assert issubclass(SchedulerNodeInput, BaseInput)

    def test_required_user_message(self):
        """user_message is required."""
        with pytest.raises(ValidationError):
            SchedulerNodeInput()

    def test_minimal_creation(self):
        """Create with only required field."""
        inp = SchedulerNodeInput(user_message="Submit a daily SQL job")
        assert inp.user_message == "Submit a daily SQL job"
        assert inp.database is None
        assert inp.prompt_version is None

    def test_full_creation(self):
        """Create with all fields."""
        inp = SchedulerNodeInput(
            user_message="List all jobs",
            database="airflow_db",
            prompt_version="1.0",
        )
        assert inp.user_message == "List all jobs"
        assert inp.database == "airflow_db"
        assert inp.prompt_version == "1.0"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        inp = SchedulerNodeInput(user_message="test", database="mydb", prompt_version="1.0")
        d = inp.model_dump()
        assert d["user_message"] == "test"
        assert d["database"] == "mydb"
        assert d["prompt_version"] == "1.0"

        restored = SchedulerNodeInput(**d)
        assert restored.user_message == "test"
        assert restored.database == "mydb"

    def test_extra_fields_forbidden(self):
        """Extra fields should be rejected (BaseInput uses extra='forbid')."""
        with pytest.raises(ValidationError):
            SchedulerNodeInput(user_message="test", unknown_field="value")


class TestSchedulerNodeResult:
    """Tests for SchedulerNodeResult model."""

    def test_inherits_from_base_result(self):
        """SchedulerNodeResult should inherit from BaseResult."""
        assert issubclass(SchedulerNodeResult, BaseResult)

    def test_success_result(self):
        """Create a successful result."""
        result = SchedulerNodeResult(
            success=True,
            response="Job submitted successfully",
            tokens_used=150,
        )
        assert result.success is True
        assert result.response == "Job submitted successfully"
        assert result.tokens_used == 150
        assert result.error is None
        assert result.scheduler_result is None

    def test_default_values(self):
        """Default values should be applied."""
        result = SchedulerNodeResult(success=True)
        assert result.response == ""
        assert result.tokens_used == 0
        assert result.scheduler_result is None

    def test_with_scheduler_result(self):
        """Create result with structured scheduler data."""
        scheduler_data = {
            "job_id": "daily_sales_summary",
            "status": "active",
            "schedule": "0 8 * * *",
        }
        result = SchedulerNodeResult(
            success=True,
            response="Created job with daily schedule",
            scheduler_result=scheduler_data,
            tokens_used=500,
        )
        assert result.scheduler_result["job_id"] == "daily_sales_summary"
        assert result.scheduler_result["status"] == "active"

    def test_error_result(self):
        """Create a failed result."""
        result = SchedulerNodeResult(
            success=False,
            error="Scheduler connection failed",
            response="Sorry, I encountered an error.",
            tokens_used=0,
        )
        assert result.success is False
        assert result.error == "Scheduler connection failed"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        result = SchedulerNodeResult(
            success=True,
            response="Jobs listed",
            scheduler_result={"count": 5},
            tokens_used=200,
        )
        d = result.model_dump()
        assert d["success"] is True
        assert d["response"] == "Jobs listed"
        assert d["scheduler_result"] == {"count": 5}
        assert d["tokens_used"] == 200

        restored = SchedulerNodeResult(**d)
        assert restored.response == "Jobs listed"
        assert restored.scheduler_result == {"count": 5}
