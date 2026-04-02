# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ExtKnowledge agentic node schema models."""

import pytest
from pydantic import ValidationError

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.ext_knowledge_agentic_node_models import ExtKnowledgeNodeInput, ExtKnowledgeNodeResult


class TestExtKnowledgeNodeInput:
    """Tests for ExtKnowledgeNodeInput model."""

    def test_inherits_from_base_input(self):
        """ExtKnowledgeNodeInput should inherit from BaseInput."""
        assert issubclass(ExtKnowledgeNodeInput, BaseInput)

    def test_required_user_message(self):
        """user_message is required."""
        with pytest.raises(ValidationError):
            ExtKnowledgeNodeInput()

    def test_minimal_creation(self):
        """Create with only required field."""
        inp = ExtKnowledgeNodeInput(user_message="Define revenue")
        assert inp.user_message == "Define revenue"
        assert inp.catalog is None
        assert inp.database is None
        assert inp.db_schema is None
        assert inp.question is None

    def test_full_creation_with_db_context(self):
        """Create with all database context fields."""
        inp = ExtKnowledgeNodeInput(
            user_message="Define revenue",
            catalog="my_catalog",
            database="my_db",
            db_schema="my_schema",
        )
        assert inp.catalog == "my_catalog"
        assert inp.database == "my_db"
        assert inp.db_schema == "my_schema"

    def test_catalog_field_optional(self):
        """catalog defaults to None."""
        inp = ExtKnowledgeNodeInput(user_message="test")
        assert inp.catalog is None

    def test_database_field_optional(self):
        """database defaults to None."""
        inp = ExtKnowledgeNodeInput(user_message="test")
        assert inp.database is None

    def test_db_schema_field_optional(self):
        """db_schema defaults to None."""
        inp = ExtKnowledgeNodeInput(user_message="test")
        assert inp.db_schema is None

    def test_serialization_roundtrip(self):
        """Model should serialize and deserialize correctly with db context fields."""
        inp = ExtKnowledgeNodeInput(
            user_message="Define metric",
            catalog="cat",
            database="db",
            db_schema="sch",
            question="What is the metric?",
            search_text="revenue",
        )
        d = inp.model_dump()
        assert d["catalog"] == "cat"
        assert d["database"] == "db"
        assert d["db_schema"] == "sch"

        restored = ExtKnowledgeNodeInput(**d)
        assert restored.catalog == "cat"
        assert restored.database == "db"
        assert restored.db_schema == "sch"
        assert restored.question == "What is the metric?"

    def test_extra_fields_forbidden(self):
        """Extra fields should be rejected (BaseInput has extra='forbid')."""
        with pytest.raises(ValidationError):
            ExtKnowledgeNodeInput(user_message="test", unknown_field="value")


class TestExtKnowledgeNodeResult:
    """Tests for ExtKnowledgeNodeResult model."""

    def test_inherits_from_base_result(self):
        """ExtKnowledgeNodeResult should inherit from BaseResult."""
        assert issubclass(ExtKnowledgeNodeResult, BaseResult)

    def test_success_result(self):
        """Create a successful result."""
        result = ExtKnowledgeNodeResult(
            success=True,
            response="Generated knowledge for revenue calculation",
            tokens_used=200,
        )
        assert result.success is True
        assert result.response == "Generated knowledge for revenue calculation"
        assert result.tokens_used == 200
        assert result.error is None

    def test_default_values(self):
        """Default values should be applied."""
        result = ExtKnowledgeNodeResult(success=True, response="ok")
        assert result.tokens_used == 0
        assert result.ext_knowledge_file is None
        assert result.error is None

    def test_error_result(self):
        """Create a failed result."""
        result = ExtKnowledgeNodeResult(
            success=False,
            response="Failed",
            error="LLM timeout",
        )
        assert result.success is False
        assert result.error == "LLM timeout"

    def test_serialization(self):
        """Model should serialize and deserialize correctly."""
        result = ExtKnowledgeNodeResult(
            success=True,
            response="Knowledge generated",
            ext_knowledge_file="/tmp/knowledge.yaml",
            tokens_used=300,
        )
        d = result.model_dump()
        assert d["ext_knowledge_file"] == "/tmp/knowledge.yaml"

        restored = ExtKnowledgeNodeResult(**d)
        assert restored.ext_knowledge_file == "/tmp/knowledge.yaml"
