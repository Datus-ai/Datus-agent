from __future__ import annotations

from pathlib import Path

import yaml

from datus.tools.skill_tools.skill_config import SkillMetadata
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.tools.skill_tools.skill_registry import SkillRegistry

ROOT = Path(__file__).parents[4]
SKILL = ROOT / "datus/resources/skills/create-subagent/SKILL.md"


def _parts() -> tuple[dict, str]:
    text = SKILL.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body


def test_create_subagent_skill_requires_mutable_configuration():
    metadata, body = _parts()

    assert metadata == {
        "name": "create-subagent",
        "description": metadata["description"],
        "requires_mutable_config": True,
    }
    assert "agent.yml" in metadata["description"]
    assert "only when the runtime marks the loaded configuration as mutable" in body
    assert "do not attempt the same write by another path" in body


def test_create_subagent_skill_is_hidden_and_refused_when_config_is_immutable():
    metadata, _ = _parts()
    parsed = SkillMetadata.from_frontmatter(metadata, SKILL.parent)
    assert parsed.requires_mutable_config is True

    registry = SkillRegistry(directories=[str(SKILL.parent)])
    mutable = SkillManager(registry=registry, config_mutable=True)
    assert [skill.name for skill in mutable.get_available_skills("chat")] == ["create-subagent"]

    immutable = SkillManager(registry=registry, config_mutable=False)
    assert immutable.get_available_skills("chat") == []
    ok, message, content = immutable.load_skill("create-subagent", "chat")
    assert ok is False
    assert "read-only" in message
    assert content is None


def test_create_subagent_skill_edits_agentic_nodes_safely_and_idempotently():
    _, body = _parts()

    assert "agent.agentic_nodes.<name>" in body
    assert "Require the file to exist and be writable" in body
    assert "Never replace the whole map" in body
    assert "operation is idempotent" in body
    assert "Re-read the file, parse it as YAML" in body
    assert "reserved for builtin system agents" in body
    assert "gen_sql_summary" in body
    assert "created" in body and "updated" in body and "unchanged" in body


def test_create_subagent_skill_supports_dashboard_agent_contract():
    _, body = _parts()

    assert "node_class: gen_sql" in body
    assert "node_class: gen_report" in body
    assert "context_search_tools,db_tools.search_table,db_tools.describe_table,db_tools.execute_sql" in body
    assert "semantic_tools,context_search_tools.list_subject_tree" in body
    assert "scoped_context.datasource" in body
    assert "scoped_context.tables" in body
    assert "scoped_context.metrics" in body
    assert "scoped_context.sqls" in body
    assert "fall back to their builtin templates" in body


def test_create_subagent_skill_uses_subject_references_for_metric_and_sql_scope():
    _, body = _parts()

    assert "canonical dotted subject references" in body
    assert "stored `subject_path` and item `name`" in body
    assert "bare subject path intentionally selects its whole subtree" in body
    assert "metric storage ID" in body
    assert "SQL summary ID" in body
    assert "Resolve every metric and reference-SQL entry" in body
    assert "Refuse unresolved or ambiguous entries" in body
    assert "datasource-only visibility" in body
    assert "restore the complete pre-edit file" in body
