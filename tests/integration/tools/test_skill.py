# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for AgentSkills that require real agent config or LLM APIs.

Tests that only need local test data (tests/data/skills/) are in
tests/unit_tests/tools/skill_tools/test_skill.py.
"""

import json

import pytest

from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillManager
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# ============================================================================
# Agentic Node Skill Filtering (agent.yml → node config)
# ============================================================================


@pytest.mark.nightly
class TestAgenticNodeSkillFiltering:
    """Test skill filtering based on agentic_nodes configuration in agent.yml.

    Validates that different nodes see different skills based on their
    `skills:` pattern and `permissions:` override in agent.yml.
    """

    def test_agent_config_loads_skills_config(self, agent_config):
        """AgentConfig correctly parses the skills section."""
        assert agent_config.skills_config is not None
        dirs = agent_config.skills_config.directories
        assert any("tests/data/skills" in d for d in dirs)

    def test_agent_config_loads_permissions_config(self, agent_config):
        """AgentConfig correctly parses the permissions section."""
        assert agent_config.permissions_config is not None
        rules = agent_config.permissions_config.rules
        assert len(rules) >= 2

        admin_rule = [r for r in rules if r.pattern == "admin-*"]
        assert len(admin_rule) == 1
        assert admin_rule[0].permission == PermissionLevel.DENY

    def test_agent_config_has_skill_nodes(self, agent_config):
        """AgentConfig has the test agentic nodes with skills config."""
        assert "school_sql" in agent_config.agentic_nodes
        assert "school_report" in agent_config.agentic_nodes
        assert "school_all" in agent_config.agentic_nodes

        assert agent_config.agentic_nodes["school_sql"].get("skills") == "sql-*"
        assert agent_config.agentic_nodes["school_report"].get("skills") == "report-*, data-*"
        assert agent_config.agentic_nodes["school_all"].get("skills") == "*"

    def test_school_sql_node_sees_only_sql_skills(self, agent_config):
        """school_sql node with skills: "sql-*" only sees sql-analysis and sql-optimization."""
        perm_manager = PermissionManager(global_config=agent_config.permissions_config)
        manager = SkillManager(
            config=agent_config.skills_config,
            permission_manager=perm_manager,
        )

        node_config = agent_config.agentic_nodes["school_sql"]
        patterns = manager.parse_skill_patterns(node_config["skills"])
        available = manager.get_available_skills("school_sql", patterns=patterns)
        names = {s.name for s in available}

        assert "sql-analysis" in names
        assert "sql-optimization" in names
        assert "report-generator" not in names
        assert "admin-tools" not in names
        assert "data-profiler" not in names

    def test_school_report_node_sees_report_and_data_skills(self, agent_config):
        """school_report node with skills: "report-*, data-*" sees matching skills."""
        perm_manager = PermissionManager(global_config=agent_config.permissions_config)
        manager = SkillManager(
            config=agent_config.skills_config,
            permission_manager=perm_manager,
        )

        node_config = agent_config.agentic_nodes["school_report"]
        patterns = manager.parse_skill_patterns(node_config["skills"])
        available = manager.get_available_skills("school_report", patterns=patterns)
        names = {s.name for s in available}

        assert "report-generator" in names
        assert "data-profiler" in names
        assert "sql-analysis" not in names
        assert "sql-optimization" not in names
        assert "admin-tools" not in names

    def test_school_all_node_sees_all_including_admin(self, agent_config):
        """school_all node with skills: "*" and admin override sees everything."""
        node_config = agent_config.agentic_nodes["school_all"]
        node_permissions = node_config.get("permissions", {})

        perm_manager = PermissionManager(
            global_config=agent_config.permissions_config,
            node_overrides={"school_all": node_permissions},
        )
        manager = SkillManager(
            config=agent_config.skills_config,
            permission_manager=perm_manager,
        )

        patterns = manager.parse_skill_patterns(node_config["skills"])
        available = manager.get_available_skills("school_all", patterns=patterns)
        names = {s.name for s in available}

        assert "sql-analysis" in names
        assert "sql-optimization" in names
        assert "report-generator" in names
        assert "data-profiler" in names
        assert "admin-tools" in names

    def test_xml_generation_respects_node_patterns(self, agent_config):
        """XML generated for different nodes contains different skills."""
        perm_manager = PermissionManager(global_config=agent_config.permissions_config)
        manager = SkillManager(
            config=agent_config.skills_config,
            permission_manager=perm_manager,
        )

        sql_patterns = manager.parse_skill_patterns("sql-*")
        sql_xml = manager.generate_available_skills_xml("school_sql", patterns=sql_patterns)
        assert "sql-analysis" in sql_xml
        assert "sql-optimization" in sql_xml
        assert "report-generator" not in sql_xml

        report_patterns = manager.parse_skill_patterns("report-*, data-*")
        report_xml = manager.generate_available_skills_xml("school_report", patterns=report_patterns)
        assert "report-generator" in report_xml
        assert "data-profiler" in report_xml
        assert "sql-analysis" not in report_xml


# ============================================================================
# Real LLM Skill Integration (Acceptance)
# ============================================================================


@pytest.mark.acceptance
@pytest.mark.nightly
class TestRealLLMSkillIntegration:
    """Real LLM integration test: ChatAgenticNode + Skills + california_schools.

    Sends a real question to DeepSeek, expects the LLM to:
    1. Query the california_schools database
    2. Load the report-generator skill via load_skill()
    3. Execute a report script via skill_execute_command()

    Marked as 'acceptance' - requires API key and real database.
    """

    QUESTION = (
        "What is the highest eligible free rate for K-12 students "
        "in the schools in Alameda County? "
        "After getting the result, use load_skill to load the 'report-generator' skill, "
        "then use skill_execute_command to generate a report with the final result."
    )

    @staticmethod
    def _print_report(
        model_name: str,
        duration: float,
        all_actions: list,
    ) -> None:
        """Print a structured test summary report."""
        separator = "\u2500" * 80

        tool_actions = [a for a in all_actions if a.role == "tool"]
        success_tools = [a for a in tool_actions if a.status == "success"]
        failed_tools = [a for a in tool_actions if a.status == "failed"]

        action_types = [a.action_type for a in all_actions]
        action_messages = " ".join(a.messages for a in all_actions)
        has_load_skill = "load_skill" in action_types or "load_skill" in action_messages
        has_skill_exec = "skill_execute_command" in action_types or "skill_execute_command" in action_messages

        logger.info(f"\n{separator}")
        logger.info("")
        logger.info("  SKILL INTEGRATION TEST SUMMARY")
        logger.info("")
        logger.info(separator)
        logger.info("")
        logger.info(f"  Model:            {model_name}")
        logger.info(f"  Total Duration:   {duration:.1f}s")
        logger.info(f"  Total Actions:    {len(all_actions)}")
        logger.info(
            f"  Tool Calls:       {len(tool_actions)}  "
            f"(pass {len(success_tools)} success, fail {len(failed_tools)} failed)"
        )

        if success_tools:
            logger.info("")
            logger.info("  SUCCESSFUL TOOLS:")
            for i, a in enumerate(success_tools, 1):
                name = a.action_type or a.function_name()
                logger.info(f"    {i}. {name:<30} -> success")

        if failed_tools:
            logger.info("")
            logger.info("  FAILED TOOLS:")
            for i, a in enumerate(failed_tools, 1):
                name = a.action_type or a.function_name()
                logger.info(f"    {i}. {name}")
                if a.input:
                    args_str = json.dumps(a.input, default=str, ensure_ascii=False)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    logger.info(f"       Args: {args_str}")
                error_msg = ""
                if a.output and isinstance(a.output, dict):
                    error_msg = a.output.get("error", "") or a.output.get("message", "")
                if not error_msg and a.messages:
                    error_msg = a.messages
                if error_msg:
                    if len(str(error_msg)) > 200:
                        error_msg = str(error_msg)[:200] + "..."
                    logger.info(f"       Error: {error_msg}")

        logger.info("")
        logger.info("  SKILL INVOCATION:")
        logger.info(f"    load_skill:             {'Found' if has_load_skill else 'Not found'}")
        logger.info(f"    skill_execute_command:   {'Found' if has_skill_exec else 'Not found'}")

        logger.info("")
        logger.info(separator)

    @pytest.mark.asyncio
    async def test_skill_invocation_in_chat(self, llm_agent_config):
        """Verify load_skill and skill_execute_command appear in action history."""
        import time

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistoryManager
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        model_key = llm_agent_config.target
        model_config = llm_agent_config.models.get(model_key, {})
        model_name = model_config.get("model", model_key) if isinstance(model_config, dict) else model_key

        node = ChatAgenticNode(
            node_id="test_skill_llm",
            description="LLM skill integration test",
            node_type="chat",
            agent_config=llm_agent_config,
        )

        node.input = ChatNodeInput(
            user_message=self.QUESTION,
            database="california_schools",
            max_turns=15,
        )

        assert node.permission_manager is not None, "PermissionManager not initialized"
        node.permission_manager.approve_for_session("skills", "*")

        ahm = ActionHistoryManager()
        start_time = time.time()
        async for _ in node.execute_stream(ahm):
            pass
        duration = time.time() - start_time

        all_actions = ahm.get_actions()
        self._print_report(model_name, duration, all_actions)

        action_types = [a.action_type for a in all_actions]
        action_messages = " ".join(a.messages for a in all_actions)

        has_load_skill = "load_skill" in action_types or "load_skill" in action_messages
        has_skill_exec = "skill_execute_command" in action_types or "skill_execute_command" in action_messages

        assert has_load_skill, f"Expected load_skill in action history. Action types found: {action_types}"
        if not has_skill_exec:
            import warnings

            warnings.warn(
                "skill_execute_command not found in action history. "
                "The LLM loaded the skill but did not execute a command.",
                stacklevel=2,
            )

        final_action = all_actions[-1]
        assert final_action.status in ("success", "failed"), f"Unexpected final status: {final_action.status}"
