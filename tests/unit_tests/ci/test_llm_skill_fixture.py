"""Nightly skill prerequisites come from versioned, isolated test data."""

import sqlite3
from pathlib import Path

import pytest

from datus.tools.skill_tools import SkillManager
from tests.integration import conftest


def required_config(factory):
    try:
        return conftest.llm_agent_config.__wrapped__(factory)
    except pytest.skip.Exception as exc:
        pytest.fail(f"Required nightly Skill fixture unexpectedly skipped: {exc}")


def test_skill_config_uses_isolated_versioned_database_and_skill(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("DATUS_STRICT_NIGHTLY_REQUIREMENTS", "1")
    config = required_config(tmp_path_factory)
    ds = config.services.datasources["california_schools"]
    path = Path(ds.uri.removeprefix("sqlite:///"))
    assert path.is_relative_to(config.home)
    with sqlite3.connect(conftest.SKILL_SQLITE_DB) as source, sqlite3.connect(path) as copied:
        original_count = source.execute("SELECT COUNT(*) FROM schools").fetchone()[0]
        assert copied.execute("SELECT COUNT(*) FROM schools").fetchone()[0] == original_count
        assert (
            copied.execute("SELECT COUNT(*) FROM frpm").fetchone()
            == source.execute("SELECT COUNT(*) FROM frpm").fetchone()
        )
        copied.execute("DELETE FROM schools")
        copied.commit()
        assert copied.execute("SELECT COUNT(*) FROM schools").fetchone() == (0,)
        assert source.execute("SELECT COUNT(*) FROM schools").fetchone() == (original_count,)
    assert config.path_manager.project_root.is_relative_to(config.home)
    skill_dir = Path(config.skills_config.directories[0])
    assert skill_dir.is_relative_to(config.home)
    assert (skill_dir / "report-generator" / "SKILL.md").is_file()
    assert (skill_dir / "report-generator" / "scripts" / "generate_report.py").is_file()
    manager = SkillManager(config=config.skills_config)
    report_skill = manager.get_skill("report-generator")
    assert report_skill.name == "report-generator"
    assert report_skill.location == skill_dir / "report-generator"


def test_skill_fixture_exposes_the_tool_requested_by_its_prompt(tmp_path_factory, monkeypatch):
    from datus.agent.node.chat_agentic_node import ChatAgenticNode
    from tests.helpers.product_acceptance import isolated_connectors, local_telemetry

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    config = required_config(tmp_path_factory)
    with isolated_connectors(), local_telemetry():
        node = ChatAgenticNode(
            node_id="skill-contract",
            description="Skill fixture",
            node_type="chat",
            agent_config=config,
            execution_mode="workflow",
        )
        assert "load_skill" in {tool.name for tool in node.tools}
        prompt = node._get_system_prompt()
        assert "report-generator" in prompt


def test_required_skill_fixture_fails_for_missing_model_key(tmp_path_factory, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DATUS_STRICT_NIGHTLY_REQUIREMENTS", "1")
    with pytest.raises(pytest.fail.Exception, match="DEEPSEEK_API_KEY"):
        conftest.llm_agent_config.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("missing", ["database", "skill"])
def test_required_versioned_skill_assets_cannot_skip(tmp_path, tmp_path_factory, monkeypatch, missing):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("DATUS_STRICT_NIGHTLY_REQUIREMENTS", "1")
    if missing == "database":
        monkeypatch.setattr(conftest, "SKILL_SQLITE_DB", tmp_path / "missing.sqlite")
    else:
        monkeypatch.setattr(conftest, "SKILLS_DIR", tmp_path / "missing-skills")
    with pytest.raises(pytest.fail.Exception, match="Required versioned Skill fixture is missing"):
        conftest.llm_agent_config.__wrapped__(tmp_path_factory)
