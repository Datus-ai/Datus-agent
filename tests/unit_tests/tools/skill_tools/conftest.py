# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared fixtures for skill unit tests (no external deps required)."""

from pathlib import Path

import pytest

from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillConfig, SkillFuncTool, SkillManager

TESTS_ROOT = Path(__file__).resolve().parents[3]  # tests/
SKILLS_DIR = TESTS_ROOT / "data" / "skills"


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


@pytest.fixture
def skill_manager(skill_config) -> SkillManager:
    """SkillManager without permissions (discovers all skills)."""
    return SkillManager(config=skill_config)


@pytest.fixture
def skill_manager_with_perms(skill_config, perm_deny_admin) -> SkillManager:
    """SkillManager with permission enforcement (admin-* denied)."""
    perm_manager = PermissionManager(global_config=perm_deny_admin)
    return SkillManager(config=skill_config, permission_manager=perm_manager)


@pytest.fixture
def skill_func_tool(skill_manager) -> SkillFuncTool:
    """SkillFuncTool for the chatbot node (no permissions)."""
    return SkillFuncTool(manager=skill_manager, node_name="chatbot")
