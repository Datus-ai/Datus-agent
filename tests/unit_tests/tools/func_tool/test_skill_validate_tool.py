# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SkillValidateTool.
"""

import pytest

from datus.tools.func_tool.skill_validate_tool import SkillValidateTool


@pytest.fixture
def tool():
    return SkillValidateTool()


def test_validation_uses_the_nodes_filesystem_scope(tmp_path):
    """A declared reader must not bypass the node's hidden/external path policy."""
    from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

    root = tmp_path / "project"
    root.mkdir()
    content = (
        "---\nname: scoped-skill\ndescription: A sufficiently descriptive skill for scope verification\n---\n"
        "Read the scoped project data and return the requested result.\n"
    )
    (root / "SKILL.md").write_text(content)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(content)
    hidden = root / ".datus" / "private"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(content)
    validator = SkillValidateTool(filesystem_tool=FilesystemFuncTool(root_path=str(root), strict=True))
    assert validator.validate_skill("SKILL.md").success == 1
    assert validator.validate_skill(str(outside / "SKILL.md")).success == 0
    assert validator.validate_skill(str(hidden / "SKILL.md")).success == 0


class TestSkillValidateTool:
    """Tests for validate_skill function."""

    def test_valid_skill(self, tool, tmp_path):
        """Valid SKILL.md should pass."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: my-skill\ndescription: A test skill for validating things properly\n"
            "tags: [test]\nversion: '1.0.0'\n---\n\n# My Skill\n\nInstructions here.\n"
        )
        result = tool.validate_skill(str(skill_md))
        assert result.success == 1
        assert result.result["status"] == "PASS"
        assert result.result["skill_name"] == "my-skill"

    def test_missing_file(self, tool):
        """Non-existent file should fail."""
        result = tool.validate_skill("/nonexistent/SKILL.md")
        assert result.success == 0
        assert "not found" in result.error

    def test_no_frontmatter(self, tool, tmp_path):
        """File without frontmatter should fail."""
        f = tmp_path / "SKILL.md"
        f.write_text("Just some markdown without frontmatter.")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "frontmatter" in result.error.lower()

    def test_invalid_yaml(self, tool, tmp_path):
        """Invalid YAML in frontmatter should fail."""
        f = tmp_path / "SKILL.md"
        f.write_text("---\nname: [invalid\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "YAML" in result.error

    def test_missing_required_name(self, tool, tmp_path):
        """Missing name should produce error."""
        f = tmp_path / "SKILL.md"
        f.write_text("---\ndescription: Some description that is long enough\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "name" in result.error

    def test_missing_required_description(self, tool, tmp_path):
        """Missing description should produce error."""
        f = tmp_path / "SKILL.md"
        f.write_text("---\nname: my-skill\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "description" in result.error

    def test_empty_body_warning(self, tool, tmp_path):
        """Empty body should produce error."""
        f = tmp_path / "SKILL.md"
        f.write_text("---\nname: my-skill\ndescription: A valid description for testing\n---\n")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "empty" in result.error.lower()

    def test_bad_name_format_warning(self, tool, tmp_path):
        """Name with uppercase should produce warning."""
        skill_dir = tmp_path / "MySkill"
        skill_dir.mkdir()
        f = skill_dir / "SKILL.md"
        f.write_text("---\nname: MySkill\ndescription: A valid test skill description\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 1
        assert result.result["status"] == "PASS with warnings"
        assert any("lowercase" in w for w in result.result["details"])

    def test_agent_without_context_warns(self, tool, tmp_path):
        """agent field without context: fork should warn."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        f = skill_dir / "SKILL.md"
        f.write_text(
            "---\nname: my-skill\ndescription: A valid test skill description here\nagent: Explore\n---\n\nbody\n"
        )
        result = tool.validate_skill(str(f))
        assert result.success == 1
        assert any("context: fork" in w for w in result.result["details"])

    def test_available_tools_returns_function_tool(self, tool):
        """available_tools should return a list with validate_skill tool."""
        tools = tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "validate_skill"

    def test_frontmatter_not_dict(self, tool, tmp_path):
        """Frontmatter that is not a dict should fail."""
        f = tmp_path / "SKILL.md"
        f.write_text("---\n- just a list\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 0
        assert "mapping" in result.error.lower()

    def test_wrong_filename_warning(self, tool, tmp_path):
        """Non-SKILL.md filename should produce warning."""
        f = tmp_path / "README.md"
        f.write_text("---\nname: my-skill\ndescription: A valid test skill description here\n---\n\nbody\n")
        result = tool.validate_skill(str(f))
        assert result.success == 1
        assert any("SKILL.md" in w for w in result.result["details"])
