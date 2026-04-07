# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillCreatorAgenticNode input/result models.
"""

import pytest
from pydantic import ValidationError

from datus.schemas.gen_skill_agentic_node_models import SkillCreatorNodeInput, SkillCreatorNodeResult


class TestSkillCreatorNodeInput:
    """Tests for SkillCreatorNodeInput model."""

    def test_required_user_message(self):
        """user_message is required."""
        input_data = SkillCreatorNodeInput(user_message="Create a SQL skill")
        assert input_data.user_message == "Create a SQL skill"

    def test_missing_user_message_raises(self):
        """Missing user_message should raise ValidationError."""
        with pytest.raises(ValidationError):
            SkillCreatorNodeInput()

    def test_optional_storage_location(self):
        """storage_location defaults to None."""
        input_data = SkillCreatorNodeInput(user_message="test")
        assert input_data.storage_location is None

    def test_storage_location_project(self):
        """storage_location can be set to 'project'."""
        input_data = SkillCreatorNodeInput(user_message="test", storage_location="project")
        assert input_data.storage_location == "project"

    def test_storage_location_user(self):
        """storage_location can be set to 'user'."""
        input_data = SkillCreatorNodeInput(user_message="test", storage_location="user")
        assert input_data.storage_location == "user"

    def test_optional_target_skill(self):
        """target_skill defaults to None."""
        input_data = SkillCreatorNodeInput(user_message="test")
        assert input_data.target_skill is None

    def test_target_skill_set(self):
        """target_skill can be set for edit mode."""
        input_data = SkillCreatorNodeInput(user_message="edit sql-optimizer", target_skill="sql-optimizer")
        assert input_data.target_skill == "sql-optimizer"

    def test_to_dict(self):
        """model_dump should produce expected dict."""
        input_data = SkillCreatorNodeInput(
            user_message="Create a skill",
            storage_location="project",
            target_skill="my-skill",
        )
        d = input_data.model_dump()
        assert d["user_message"] == "Create a skill"
        assert d["storage_location"] == "project"
        assert d["target_skill"] == "my-skill"

    def test_extra_fields_forbidden(self):
        """Extra fields should raise ValidationError (extra='forbid')."""
        with pytest.raises(ValidationError):
            SkillCreatorNodeInput(user_message="test", unknown_field="value")


class TestSkillCreatorNodeResult:
    """Tests for SkillCreatorNodeResult model."""

    def test_defaults(self):
        """Default values should be set correctly."""
        result = SkillCreatorNodeResult(success=True)
        assert result.response == ""
        assert result.skill_name is None
        assert result.skill_path is None
        assert result.tokens_used == 0
        assert result.error is None

    def test_full_result(self):
        """All fields can be set."""
        result = SkillCreatorNodeResult(
            success=True,
            response="Created skill successfully",
            skill_name="sql-analyzer",
            skill_path="/home/user/.datus/skills/sql-analyzer",
            tokens_used=1500,
        )
        assert result.success is True
        assert result.skill_name == "sql-analyzer"
        assert result.skill_path == "/home/user/.datus/skills/sql-analyzer"
        assert result.tokens_used == 1500

    def test_error_result(self):
        """Error result can be created."""
        result = SkillCreatorNodeResult(
            success=False,
            error="Template loading failed",
            response="Sorry, I encountered an error.",
        )
        assert result.success is False
        assert result.error == "Template loading failed"

    def test_serialization_roundtrip(self):
        """Result should survive JSON serialization roundtrip."""
        result = SkillCreatorNodeResult(
            success=True,
            response="Created skill",
            skill_name="my-skill",
            skill_path="./skills/my-skill",
            tokens_used=500,
        )
        json_str = result.model_dump_json()
        restored = SkillCreatorNodeResult.model_validate_json(json_str)
        assert restored.skill_name == "my-skill"
        assert restored.tokens_used == 500
