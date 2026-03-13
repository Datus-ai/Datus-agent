# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for ExploreAgenticNode schema models.
"""

import pytest
from pydantic import ValidationError

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.explore_agentic_node_models import ExploreNodeInput, ExploreNodeResult


class TestExploreNodeInput:
    """Tests for ExploreNodeInput model."""

    def test_inherits_from_base_input(self):
        """ExploreNodeInput should inherit from BaseInput."""
        assert issubclass(ExploreNodeInput, BaseInput)

    def test_required_user_message(self):
        """user_message is required."""
        with pytest.raises(ValidationError):
            ExploreNodeInput()

    def test_minimal_creation(self):
        """Create with only required field."""
        inp = ExploreNodeInput(user_message="Find sales tables")
        assert inp.user_message == "Find sales tables"
        assert inp.database is None

    def test_full_creation(self):
        """Create with all fields."""
        inp = ExploreNodeInput(
            user_message="Explore order tables",
            database="test_db",
        )
        assert inp.user_message == "Explore order tables"
        assert inp.database == "test_db"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        inp = ExploreNodeInput(user_message="test", database="mydb")
        d = inp.model_dump()
        assert d["user_message"] == "test"
        assert d["database"] == "mydb"

        restored = ExploreNodeInput(**d)
        assert restored.user_message == "test"
        assert restored.database == "mydb"


class TestExploreNodeResult:
    """Tests for ExploreNodeResult model."""

    def test_inherits_from_base_result(self):
        """ExploreNodeResult should inherit from BaseResult."""
        assert issubclass(ExploreNodeResult, BaseResult)

    def test_success_result(self):
        """Create a successful result."""
        result = ExploreNodeResult(
            success=True,
            response="Found 3 tables related to orders",
            tokens_used=150,
        )
        assert result.success is True
        assert result.response == "Found 3 tables related to orders"
        assert result.tokens_used == 150
        assert result.error is None

    def test_default_values(self):
        """Default values should be applied."""
        result = ExploreNodeResult(success=True)
        assert result.response == ""
        assert result.tokens_used == 0

    def test_error_result(self):
        """Create a failed result."""
        result = ExploreNodeResult(
            success=False,
            error="Database connection failed",
            response="Sorry, I encountered an error.",
            tokens_used=0,
        )
        assert result.success is False
        assert result.error == "Database connection failed"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        result = ExploreNodeResult(
            success=True,
            response="Schema findings here",
            tokens_used=200,
        )
        d = result.model_dump()
        assert d["success"] is True
        assert d["response"] == "Schema findings here"
        assert d["tokens_used"] == 200

        restored = ExploreNodeResult(**d)
        assert restored.response == "Schema findings here"
