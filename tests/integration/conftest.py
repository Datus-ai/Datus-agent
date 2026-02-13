# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared fixtures for integration tests.

Provides reusable AgentConfig, SkillManager, and PermissionManager fixtures
that load from tests/conf/agent.yml — mirroring the real agent startup flow.
"""

import os
import shutil
from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.configuration.agent_config_loader import load_agent_config
from datus.tools.permission.permission_config import (
    PermissionConfig,
    PermissionLevel,
    PermissionRule,
)
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillConfig, SkillFuncTool, SkillManager

TESTS_ROOT = Path(__file__).parent.parent  # tests/
CONF_DIR = TESTS_ROOT / "conf"
SKILLS_DIR = TESTS_ROOT / "data" / "skills"

# Real LLM integration test paths
REAL_SKILLS_DIR = Path.home() / ".datus" / "skills"
REAL_SQLITE_DB = Path.home() / ".datus" / "benchmark" / "california_schools" / "california_schools.sqlite"


# ── AgentConfig fixtures ──


@pytest.fixture(scope="module")
def agent_config(tmp_path_factory) -> AgentConfig:
    """Load AgentConfig from a temp copy of tests/conf/agent.yml.

    Uses tmp copy so tests never mutate the source config.
    The config includes skills, permissions, and agentic_nodes sections.
    """
    src = CONF_DIR / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("skill_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    shutil.copy2(src, tmp_cfg)
    config = load_agent_config(
        config=str(tmp_cfg),
        namespace="bird_school",
        reload=True,
        force=True,
        yes=True,
    )
    return config


# ── SkillConfig fixtures ──


@pytest.fixture
def skill_config() -> SkillConfig:
    """SkillConfig pointing to tests/data/skills."""
    return SkillConfig(directories=[str(SKILLS_DIR)])


@pytest.fixture
def skill_config_with_extra(tmp_path) -> tuple[SkillConfig, Path]:
    """SkillConfig with two directories: real skills + a tmp dir for dynamic tests."""
    extra_dir = tmp_path / "extra_skills"
    extra_dir.mkdir()
    return SkillConfig(directories=[str(SKILLS_DIR), str(extra_dir)]), extra_dir


# ── PermissionManager fixtures ──


@pytest.fixture
def perm_deny_admin() -> PermissionConfig:
    """Permission config that denies admin-* skills."""
    return PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
        ],
    )


@pytest.fixture
def perm_ask_sql() -> PermissionConfig:
    """Permission config that requires ASK for sql-* skills."""
    return PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="sql-*", permission=PermissionLevel.ASK),
        ],
    )


@pytest.fixture
def perm_deny_admin_with_node_override() -> tuple:
    """Global DENY admin + node override that ALLOWs admin for school_all."""
    global_config = PermissionConfig(
        default_permission=PermissionLevel.ALLOW,
        rules=[
            PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.DENY),
        ],
    )
    node_overrides = {
        "school_all": PermissionConfig(
            rules=[
                PermissionRule(tool="skills", pattern="admin-*", permission=PermissionLevel.ALLOW),
            ],
        ),
    }
    return global_config, node_overrides


# ── SkillManager fixtures ──


@pytest.fixture
def skill_manager(skill_config) -> SkillManager:
    """SkillManager without permissions (discovers all skills)."""
    return SkillManager(config=skill_config)


def pytest_collection_modifyitems(items):
    """Automatically mark all tests under integration/ with the 'integration' marker."""
    for item in items:
        if "integration" in str(item.fspath) and "unit_tests" not in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def skill_manager_with_perms(skill_config, perm_deny_admin) -> SkillManager:
    """SkillManager with permission enforcement (admin-* denied)."""
    perm_manager = PermissionManager(global_config=perm_deny_admin)
    return SkillManager(config=skill_config, permission_manager=perm_manager)


# ── SkillFuncTool fixtures ──


@pytest.fixture
def skill_func_tool(skill_manager) -> SkillFuncTool:
    """SkillFuncTool for the chatbot node (no permissions)."""
    return SkillFuncTool(manager=skill_manager, node_name="chatbot")


# ── Real LLM integration test fixtures ──


@pytest.fixture(scope="module")
def llm_agent_config(tmp_path_factory) -> AgentConfig:
    """Load AgentConfig for real LLM integration tests.

    Uses tests/conf/agent_llm_skill.yml with california_schools database
    and real skills from ~/.datus/skills/.

    Skips if prerequisites missing:
    - DEEPSEEK_API_KEY not set
    - california_schools.sqlite not found
    - ~/.datus/skills/report-generator/ not found
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    if not REAL_SQLITE_DB.exists():
        pytest.skip(f"SQLite database not found: {REAL_SQLITE_DB}")
    if not (REAL_SKILLS_DIR / "report-generator" / "SKILL.md").exists():
        pytest.skip(f"report-generator skill not found: {REAL_SKILLS_DIR}")

    src = CONF_DIR / "agent_llm_skill.yml"
    tmp_dir = tmp_path_factory.mktemp("llm_skill_conf")
    tmp_cfg = tmp_dir / "agent_llm_skill.yml"
    shutil.copy2(src, tmp_cfg)

    config = load_agent_config(
        config=str(tmp_cfg),
        namespace="california_schools",
        reload=True,
        force=True,
        yes=True,
    )
    return config
