# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Integration of validation layer with SkillMetadata / SkillRegistry / SkillManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from datus.tools.skill_tools.skill_config import SkillConfig, SkillMetadata
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.tools.skill_tools.skill_registry import SkillRegistry


class TestSkillMetadataValidatorFields:
    def test_plain_skill_defaults(self):
        m = SkillMetadata.from_frontmatter({"name": "x", "description": "y"}, Path("/tmp/x"))
        assert not m.is_validator()
        assert m.kind == "skill"
        assert m.trigger == []
        assert m.severity == "advisory"
        assert m.mode == "llm"
        assert m.targets == []

    def test_validator_fields_parsed(self):
        m = SkillMetadata.from_frontmatter(
            {
                "name": "x",
                "description": "y",
                "kind": "validator",
                "trigger": ["on_tool_end"],
                "severity": "blocking",
                "mode": "llm",
                "targets": [
                    {"type": "table"},
                    {"type": "table", "schema": "public", "table_pattern": "rev_*"},
                ],
                "allowed_agents": ["gen_table"],
            },
            Path("/tmp/x"),
        )
        assert m.is_validator()
        assert m.trigger == ["on_tool_end"]
        assert m.severity == "blocking"
        assert len(m.targets) == 2
        # Second filter: schema alias → db_schema
        assert m.targets[1].db_schema == "public"
        assert m.targets[1].table_pattern == "rev_*"

    def test_yaml_off_severity_coerced(self):
        """YAML parses bare ``off`` as False — from_frontmatter coerces back."""
        m = SkillMetadata.from_frontmatter(
            {"name": "x", "description": "y", "kind": "validator", "severity": False},
            Path("/tmp/x"),
        )
        assert m.severity == "off"

    def test_invalid_target_entry_raises(self):
        """Invalid ``targets`` entries surface as ``DatusException`` with the
        dedicated ``SKILL_FRONTMATTER_INVALID`` error code — the repo's
        guardrail forbids bare ``ValueError`` for expected config failures."""
        from datus.utils.exceptions import DatusException, ErrorCode

        with pytest.raises(DatusException) as exc:
            SkillMetadata.from_frontmatter(
                {"name": "x", "description": "y", "targets": [["not-a-dict"]]},
                Path("/tmp/x"),
            )
        assert exc.value.code == ErrorCode.SKILL_FRONTMATTER_INVALID


class TestSkillRegistryGetValidators:
    @pytest.fixture
    def registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "v1").mkdir()
            (base / "v1" / "SKILL.md").write_text(
                """---
name: v1
description: test
kind: validator
trigger: [on_tool_end]
severity: blocking
allowed_agents: [gen_table, gen_job]
---
body
"""
            )
            (base / "v2").mkdir()
            (base / "v2" / "SKILL.md").write_text(
                """---
name: v2
description: test
kind: validator
trigger: [on_end]
severity: blocking
allowed_agents: [gen_job]
---
body
"""
            )
            (base / "v-off").mkdir()
            (base / "v-off" / "SKILL.md").write_text(
                """---
name: v-off
description: test
kind: validator
trigger: [on_tool_end]
severity: off
allowed_agents: [gen_table]
---
body
"""
            )
            (base / "normal").mkdir()
            (base / "normal" / "SKILL.md").write_text(
                """---
name: normal
description: test
---
body
"""
            )
            yield SkillRegistry(config=SkillConfig(directories=[str(base)]))

    def test_on_tool_end_filter(self, registry):
        names = [s.name for s in registry.get_validators("gen_table", "on_tool_end")]
        assert names == ["v1"]  # v-off excluded by severity; v2 triggers on_end only

    def test_on_end_filter(self, registry):
        names = [s.name for s in registry.get_validators("gen_job", "on_end")]
        assert names == ["v2"]

    def test_allowed_agents_respected(self, registry):
        names = [s.name for s in registry.get_validators("gen_dashboard", "on_tool_end")]
        assert names == []

    def test_off_severity_excluded(self, registry):
        names = [s.name for s in registry.get_validators("gen_table", "on_tool_end")]
        assert "v-off" not in names


class TestSkillManagerValidatorExclusion:
    def test_validator_skills_not_in_available(self):
        """``kind=validator`` skills are hook-driven and must NOT appear in the main agent's list."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "v1").mkdir()
            (base / "v1" / "SKILL.md").write_text(
                """---
name: v1
description: test
kind: validator
trigger: [on_tool_end]
allowed_agents: [gen_table]
---
body
"""
            )
            (base / "n1").mkdir()
            (base / "n1" / "SKILL.md").write_text(
                """---
name: n1
description: normal skill
allowed_agents: [gen_table]
---
body
"""
            )
            reg = SkillRegistry(config=SkillConfig(directories=[str(base)]))
            mgr = SkillManager(registry=reg)
            names = [s.name for s in mgr.get_available_skills("gen_table")]
            assert "v1" not in names, "validator skill leaked into main agent's tool list"
            assert "n1" in names
