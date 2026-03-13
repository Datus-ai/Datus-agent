# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CI-level tests for the gen-metadata skill.

Tests SKILL.md parsing, script existence, skill discovery, and CSV parsing logic.
"""

import csv
import io
from pathlib import Path

import pytest
import yaml

from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.tools.skill_tools.skill_registry import SkillRegistry

# ── Fixtures ──


@pytest.fixture
def skill_dir():
    """Return the path to the gen-metadata skill directory."""
    d = Path(__file__).resolve().parents[4] / "skills" / "gen-metadata"
    assert d.exists(), f"skills/gen-metadata directory not found at {d}"
    return d


@pytest.fixture
def skill_metadata(skill_dir):
    """Parse and return SkillMetadata from SKILL.md."""
    content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    parts = content.split("---", 2)
    assert len(parts) >= 3, "SKILL.md must have YAML frontmatter"
    frontmatter = yaml.safe_load(parts[1])
    return SkillMetadata.from_frontmatter(frontmatter, skill_dir)


# ── SKILL.md Parsing Tests ──


@pytest.mark.ci
class TestMetadataGenSkillMd:
    """Test SKILL.md frontmatter parsing and content structure."""

    def test_skill_md_exists(self, skill_dir):
        assert (skill_dir / "SKILL.md").exists()

    def test_frontmatter_name(self, skill_metadata):
        assert skill_metadata.name == "gen-metadata"

    def test_frontmatter_description(self, skill_metadata):
        assert skill_metadata.description
        assert "metadata" in skill_metadata.description.lower()

    def test_frontmatter_allowed_commands(self, skill_metadata):
        assert skill_metadata.allowed_commands
        assert "python:scripts/*.py" in skill_metadata.allowed_commands

    def test_frontmatter_tags(self, skill_metadata):
        assert skill_metadata.tags
        assert "metadata" in skill_metadata.tags

    def test_has_scripts(self, skill_metadata):
        assert skill_metadata.has_scripts()

    def test_is_model_invocable(self, skill_metadata):
        assert skill_metadata.is_model_invocable()

    def test_scripts_exist(self, skill_dir):
        scripts_dir = skill_dir / "scripts"
        assert scripts_dir.exists()
        assert (scripts_dir / "list_tables.py").exists()
        assert (scripts_dir / "get_table_ddl.py").exists()
        assert (scripts_dir / "get_sample_rows.py").exists()
        assert (scripts_dir / "write_metadata.py").exists()


# ── Skill Discovery Tests ──


@pytest.mark.ci
class TestMetadataGenSkillDiscovery:
    """Test that gen-metadata skill is discoverable by SkillRegistry."""

    def test_registry_discovers_skill(self, skill_dir):
        registry = SkillRegistry(directories=[str(skill_dir.parent)])
        skill_names = [s.name for s in registry.list_skills()]
        assert "gen-metadata" in skill_names

    def test_registry_get_skill(self, skill_dir):
        registry = SkillRegistry(directories=[str(skill_dir.parent)])
        skill = registry.get_skill("gen-metadata")
        assert skill is not None
        assert skill.name == "gen-metadata"


# ── CSV Parsing Tests ──


@pytest.mark.ci
class TestMetadataGenCsvParsing:
    """Test CSV parsing logic used in get_sample_rows.py."""

    def _parse_csv(self, csv_data):
        """Replicate the CSV parsing logic from get_sample_rows.py."""
        reader = csv.reader(io.StringIO(csv_data.strip()))
        parsed = list(reader)
        if parsed:
            columns = parsed[0]
            rows = parsed[1:]
            return {"columns": columns, "rows": rows, "count": len(rows)}
        return {"columns": [], "rows": [], "count": 0}

    def test_simple_csv(self):
        csv_data = "id,name,age\n1,Alice,30\n2,Bob,25"
        result = self._parse_csv(csv_data)
        assert result["columns"] == ["id", "name", "age"]
        assert result["rows"] == [["1", "Alice", "30"], ["2", "Bob", "25"]]
        assert result["count"] == 2

    def test_csv_with_quoted_fields(self):
        """Fields containing commas should be preserved when quoted."""
        csv_data = 'id,name,address\n1,"Smith, John","123 Main St, Apt 4"\n2,Jane,456 Oak Ave'
        result = self._parse_csv(csv_data)
        assert result["columns"] == ["id", "name", "address"]
        assert result["rows"][0] == ["1", "Smith, John", "123 Main St, Apt 4"]
        assert result["rows"][1] == ["2", "Jane", "456 Oak Ave"]

    def test_csv_with_empty_fields(self):
        csv_data = "id,name,value\n1,,100\n2,test,"
        result = self._parse_csv(csv_data)
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"][0] == ["1", "", "100"]
        assert result["rows"][1] == ["2", "test", ""]

    def test_csv_header_only(self):
        csv_data = "id,name,value"
        result = self._parse_csv(csv_data)
        assert result["columns"] == ["id", "name", "value"]
        assert result["rows"] == []
        assert result["count"] == 0

    def test_empty_csv(self):
        csv_data = ""
        result = self._parse_csv(csv_data)
        assert result["columns"] == []
        assert result["rows"] == []
        assert result["count"] == 0

    def test_csv_with_quoted_newlines(self):
        """Fields containing newlines should be preserved when quoted."""
        csv_data = 'id,description\n1,"line1\nline2"\n2,simple'
        result = self._parse_csv(csv_data)
        assert result["columns"] == ["id", "description"]
        assert result["rows"][0] == ["1", "line1\nline2"]
        assert result["rows"][1] == ["2", "simple"]


# ── SKILL.md Content Tests ──


@pytest.mark.ci
class TestMetadataGenSkillContent:
    """Test that SKILL.md content includes expected workflow sections."""

    def test_has_workflow_steps(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "Step 1" in content
        assert "Step 2" in content
        assert "list_tables" in content
        assert "get_table_ddl" in content
        assert "get_sample_rows" in content
        assert "write_metadata" in content

    def test_has_environment_variables(self, skill_dir):
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert "DATUS_CONFIG_PATH" in content
        assert "DATUS_NAMESPACE" in content
        assert "DATUS_HOME" in content
