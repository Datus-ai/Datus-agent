# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Nightly coverage for the extension flow ``extension.skills`` — the locally
testable part: skill discovery + permission filtering + real-LLM execution of
an installed *local* skill through an agentic node.

Marketplace install / update / remove / publish require the external Town
Backend and are reported scoped-out (no network in this suite).

This file complements the existing tests/integration/tools/test_skill.py (which
it does not modify):
- ``TestLocalSkillDiscoveryAndExecutionGate`` is deterministic: it builds a
  ChatAgenticNode wired with a SkillFuncTool and verifies the node actually
  exposes the local skill-execution tools (``load_skill`` /
  ``skill_execute_command``) and that permission filtering narrows the visible
  skill set by pattern.
- ``TestRealLLMSkillExecution`` is a real-LLM run: the agent is told to load a
  local skill and execute one of its commands, and we assert the run reaches a
  terminal status with the skill actually invoked.
"""

import os

import pytest

from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillFuncTool, SkillManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ============================================================================
# Deterministic: local skill discovery + permission filtering + tool exposure
# ============================================================================


@pytest.mark.nightly
class TestLocalSkillDiscoveryAndExecutionGate:
    """Local skill discovery, permission filtering, and node tool exposure.

    Uses tests/data/skills (the same fixture set the existing skill suite uses)
    via the shared ``agent_config`` fixture — no marketplace, no network.
    """

    def test_skill_func_tool_exposes_execution_tools(self, agent_config):
        """A node wired with SkillFuncTool exposes the local execution tools."""
        perm_manager = PermissionManager(global_config=agent_config.permissions_config)
        manager = SkillManager(config=agent_config.skills_config, permission_manager=perm_manager)
        func_tool = SkillFuncTool(manager=manager, node_name="school_all")

        tool_names = {t.name for t in func_tool.available_tools()}
        assert "load_skill" in tool_names, f"SkillFuncTool must expose load_skill, got: {sorted(tool_names)}"
        assert "skill_execute_command" in tool_names, (
            f"SkillFuncTool must expose skill_execute_command, got: {sorted(tool_names)}"
        )

    def test_permission_filtering_narrows_visible_skills(self, agent_config):
        """The sql-* pattern only surfaces sql-* skills, never report/admin ones."""
        perm_manager = PermissionManager(global_config=agent_config.permissions_config)
        manager = SkillManager(config=agent_config.skills_config, permission_manager=perm_manager)

        patterns = manager.parse_skill_patterns("sql-*")
        available = manager.get_available_skills("school_sql", patterns=patterns)
        names = {s.name for s in available}

        assert "sql-analysis" in names, f"sql-* should include sql-analysis, got: {sorted(names)}"
        assert "report-generator" not in names, f"sql-* must exclude report-generator, got: {sorted(names)}"
        # admin-* is DENY in the test permissions config and must never surface.
        assert "admin-tools" not in names, f"admin-tools must be denied, got: {sorted(names)}"


# ============================================================================
# Real-LLM: agent loads and executes an installed local skill
# ============================================================================


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestRealLLMSkillExecution:
    """Real-LLM execution of an installed local skill via ChatAgenticNode.

    Reuses the ``llm_agent_config`` fixture (skips automatically when the
    DEEPSEEK key, california_schools DB, or the local report-generator skill is
    missing). No marketplace network is touched.
    """

    QUESTION = (
        "Use load_skill to load the 'report-generator' skill, then use "
        "skill_execute_command to run one of its commands to produce a short "
        "report containing the text 'Alameda'. Keep it simple."
    )

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_agent_invokes_local_skill(self, llm_agent_config):
        """N924-SKILL: the agent loads a local skill and reaches a terminal status."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistoryManager
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        node = ChatAgenticNode(
            node_id="nightly_skill_exec",
            description="local skill execution",
            node_type="chat",
            agent_config=llm_agent_config,
        )
        node.input = ChatNodeInput(
            user_message=self.QUESTION,
            database="california_schools",
            max_turns=15,
        )

        assert isinstance(node.permission_manager, PermissionManager), (
            f"Node must own a PermissionManager for skill gating, got {type(node.permission_manager).__name__}"
        )
        node.permission_manager.approve_for_session("skills", "*")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        all_actions = action_manager.get_actions()
        assert len(all_actions) >= 2, f"Expected at least 2 actions, got {len(all_actions)}"

        action_types = [a.action_type for a in all_actions]
        action_messages = " ".join(a.messages for a in all_actions)
        has_load_skill = "load_skill" in action_types or "load_skill" in action_messages
        assert has_load_skill, f"Agent should invoke load_skill for a local skill. Action types: {action_types}"

        final_action = all_actions[-1]
        assert final_action.status in ("success", "failed"), (
            f"Run must reach a terminal status, got {final_action.status}"
        )
