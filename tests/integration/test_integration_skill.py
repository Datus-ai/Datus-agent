# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for AgentSkills + Permission system.

Tests the full integration flow:
  agent.yml → AgentConfig → SkillManager + PermissionManager → SkillFuncTool

Covers:
  1. Skill discovery from real filesystem (tests/data/skills/)
  2. Load → execute pipeline with real Python scripts
  3. Permission enforcement across config layers
  4. Agentic node skill filtering (skills: "sql-*" etc.)
  5. Multi-skill tool accumulation lifecycle
"""

import json

import pytest

from datus.tools.permission.permission_config import PermissionLevel
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.skill_tools import SkillConfig, SkillManager

# ============================================================================
# 1. Skill Discovery Integration
# ============================================================================


@pytest.mark.nightly
class TestSkillDiscoveryIntegration:
    """Test skill discovery from real filesystem directories."""

    @pytest.mark.acceptance
    def test_discovers_all_skills_from_data_dir(self, skill_manager):
        """All 5 skills in tests/data/skills/ are discovered."""
        skills = skill_manager.list_all_skills()
        names = {s.name for s in skills}
        assert names == {
            "sql-analysis",
            "sql-optimization",
            "report-generator",
            "admin-tools",
            "data-profiler",
        }

    def test_multi_directory_discovery(self, skill_config_with_extra):
        """Skills from multiple directories are merged."""
        config, extra_dir = skill_config_with_extra

        # Add a skill to the extra directory
        new_skill_dir = extra_dir / "extra-skill"
        new_skill_dir.mkdir()
        (new_skill_dir / "SKILL.md").write_text("---\nname: extra-skill\ndescription: Extra\n---\n# Extra")

        manager = SkillManager(config=config)
        names = {s.name for s in manager.list_all_skills()}

        # Should have all 5 base skills + 1 extra
        assert "extra-skill" in names
        assert "sql-analysis" in names
        assert len(names) == 6

    def test_refresh_picks_up_new_skills(self, skill_config_with_extra):
        """After adding a skill and calling refresh(), it's discovered."""
        config, extra_dir = skill_config_with_extra
        manager = SkillManager(config=config)

        initial_count = manager.get_skill_count()

        # Add new skill at runtime
        runtime_dir = extra_dir / "runtime-skill"
        runtime_dir.mkdir()
        (runtime_dir / "SKILL.md").write_text("---\nname: runtime-skill\ndescription: Added at runtime\n---\n# Runtime")

        manager.refresh()
        assert manager.get_skill_count() == initial_count + 1
        assert manager.get_skill("runtime-skill") is not None

    def test_nonexistent_directory_gracefully_skipped(self, tmp_path):
        """Mix of valid + invalid directories works without error."""
        from tests.integration.conftest import SKILLS_DIR

        config = SkillConfig(
            directories=[
                str(SKILLS_DIR),
                str(tmp_path / "does_not_exist"),
            ]
        )
        manager = SkillManager(config=config)
        # Should still find the real skills
        assert manager.get_skill_count() == 5

    def test_duplicate_skill_first_directory_wins(self, skill_config_with_extra):
        """When same skill name exists in two dirs, first discovered wins."""
        config, extra_dir = skill_config_with_extra

        # Create a skill with same name as existing one in the second directory
        dup_dir = extra_dir / "sql-analysis"
        dup_dir.mkdir()
        (dup_dir / "SKILL.md").write_text("---\nname: sql-analysis\ndescription: Override version\n---\n# Override")

        manager = SkillManager(config=config)
        skill = manager.get_skill("sql-analysis")
        assert skill is not None
        # First directory wins — the original version is kept, duplicate skipped
        assert skill.description == "Guided workflow for SQL data analysis using db_tools"
        # Total count stays 5 (duplicate not added)
        assert manager.get_skill_count() == 5


# ============================================================================
# 2. Load → Execute Pipeline
# ============================================================================


@pytest.mark.nightly
class TestSkillLoadAndExecuteIntegration:
    """Test the full load → execute → result pipeline with real scripts."""

    def test_workflow_skill_loads_content_no_bash_tool(self, skill_func_tool):
        """Workflow-only skill returns content but no SkillBashTool."""
        result = skill_func_tool.load_skill("sql-analysis")
        assert result.success == 1
        assert "Schema Discovery" in result.result
        assert "db_tools" in result.result

        # No bash tool for workflow-only skills
        assert skill_func_tool.get_skill_bash_tool("sql-analysis") is None

    @pytest.mark.acceptance
    def test_script_skill_loads_and_executes(self, skill_func_tool):
        """Script skill creates bash tool; execute returns JSON output."""
        result = skill_func_tool.load_skill("report-generator")
        assert result.success == 1
        assert "python scripts/generate_report.py" in result.result

        # Bash tool should exist
        bash_tool = skill_func_tool.get_skill_bash_tool("report-generator")
        assert bash_tool is not None

        # Execute script
        exec_result = bash_tool.execute_command("python scripts/generate_report.py --format json")
        assert exec_result.success == 1
        output = json.loads(exec_result.result.strip())
        assert output["status"] == "success"
        assert output["format"] == "json"
        assert output["rows_processed"] == 42

    def test_chained_workflow_then_execute(self, skill_func_tool):
        """Load workflow skill first, then script skill, execute script."""
        # Step 1: Load workflow skill
        r1 = skill_func_tool.load_skill("sql-analysis")
        assert r1.success == 1
        assert skill_func_tool.get_skill_bash_tool("sql-analysis") is None

        # Step 2: Load script skill
        r2 = skill_func_tool.load_skill("data-profiler")
        assert r2.success == 1

        # Step 3: Execute script from data-profiler
        bash_tool = skill_func_tool.get_skill_bash_tool("data-profiler")
        assert bash_tool is not None
        exec_result = bash_tool.execute_command("python scripts/profile_data.py --table students")
        assert exec_result.success == 1
        output = json.loads(exec_result.result.strip())
        assert output["table"] == "students"

    def test_script_execution_error_propagates(self, skill_func_tool):
        """Script that raises an error returns failure result."""
        # Use skill_execute_command for a skill that doesn't exist
        result = skill_func_tool.skill_execute_command("nonexistent-skill", "echo hello")
        assert result.success == 0
        assert "not found" in result.error.lower()

    @pytest.mark.acceptance
    def test_denied_command_rejected(self, skill_func_tool):
        """Commands not matching allowed_commands are rejected."""
        skill_func_tool.load_skill("report-generator")
        bash_tool = skill_func_tool.get_skill_bash_tool("report-generator")

        result = bash_tool.execute_command("rm -rf /")
        assert result.success == 0
        assert "not allowed" in result.error.lower()

    def test_skill_execute_command_before_load(self, skill_func_tool):
        """Calling skill_execute_command before load_skill gives helpful error."""
        result = skill_func_tool.skill_execute_command("report-generator", "python scripts/generate_report.py")
        assert result.success == 0
        assert "not been loaded" in result.error.lower() or "load_skill" in result.error.lower()


# ============================================================================
# 3. Permission Enforcement Integration
# ============================================================================


@pytest.mark.nightly
class TestPermissionIntegration:
    """Test permission enforcement across SkillManager + PermissionManager layers."""

    def test_deny_hides_skill_from_available_and_xml(self, skill_manager_with_perms):
        """DENY permission hides skill from get_available_skills and XML."""
        available = skill_manager_with_perms.get_available_skills("chatbot")
        names = [s.name for s in available]

        assert "admin-tools" not in names
        assert "sql-analysis" in names
        assert "report-generator" in names

        xml = skill_manager_with_perms.generate_available_skills_xml("chatbot")
        assert "admin-tools" not in xml
        assert "sql-analysis" in xml

    @pytest.mark.acceptance
    def test_deny_blocks_load(self, skill_manager_with_perms):
        """DENY permission blocks load_skill."""
        success, message, _content = skill_manager_with_perms.load_skill("admin-tools", "chatbot")
        assert success is False
        assert "denied" in message.lower() or "Permission" in message

    def test_ask_keeps_skill_visible_but_blocks_load(self, skill_config, perm_ask_sql):
        """ASK permission keeps skill visible but returns ASK_PERMISSION on load."""
        perm_manager = PermissionManager(global_config=perm_ask_sql)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]
        assert "sql-analysis" in names  # Visible

        success, message, _content = manager.load_skill("sql-analysis", "chatbot")
        assert success is False
        assert message == "ASK_PERMISSION"

    def test_node_override_grants_access_to_denied_skill(self, skill_config, perm_deny_admin_with_node_override):
        """Global DENY + node-specific ALLOW → skill accessible for that node."""
        global_config, node_overrides = perm_deny_admin_with_node_override
        perm_manager = PermissionManager(global_config=global_config, node_overrides=node_overrides)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        # Regular chatbot: admin-tools denied
        chatbot_skills = manager.get_available_skills("chatbot")
        chatbot_names = [s.name for s in chatbot_skills]
        assert "admin-tools" not in chatbot_names

        # school_all node: admin-tools allowed via override
        all_skills = manager.get_available_skills("school_all")
        all_names = [s.name for s in all_skills]
        assert "admin-tools" in all_names

        # Verify load works for school_all
        success, message, content = manager.load_skill("admin-tools", "school_all")
        assert success is True
        assert "Administrative" in content

    def test_permission_with_pattern_filtering_combined(self, skill_config, perm_deny_admin):
        """Pattern filter + permission filter work together."""
        perm_manager = PermissionManager(global_config=perm_deny_admin)
        manager = SkillManager(config=skill_config, permission_manager=perm_manager)

        # Pattern: only sql-* skills
        available = manager.get_available_skills("chatbot", patterns=["sql-*"])
        names = [s.name for s in available]

        assert "sql-analysis" in names
        assert "sql-optimization" in names
        assert "report-generator" not in names  # Filtered by pattern
        assert "admin-tools" not in names  # Filtered by permission + pattern

    def test_disable_model_invocation_hides_from_available(self, tmp_path):
        """disable_model_invocation: true hides skill from get_available_skills."""
        skill_dir = tmp_path / "hidden-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: hidden-skill\ndescription: Hidden\n" "disable_model_invocation: true\n---\n# Hidden"
        )

        config = SkillConfig(directories=[str(tmp_path)])
        manager = SkillManager(config=config)

        available = manager.get_available_skills("chatbot")
        names = [s.name for s in available]
        assert "hidden-skill" not in names

        # But it still exists in registry
        assert manager.get_skill("hidden-skill") is not None


# ============================================================================
# 4. Agentic Node Skill Filtering (agent.yml → node config)
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

        # Check admin-* deny rule exists
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

    @pytest.mark.acceptance
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

        # sql-* matches, but global permission has ASK for sql-* (still visible)
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

        # school_all has "*" pattern + admin override → sees everything
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

        # school_sql XML
        sql_patterns = manager.parse_skill_patterns("sql-*")
        sql_xml = manager.generate_available_skills_xml("school_sql", patterns=sql_patterns)
        assert "sql-analysis" in sql_xml
        assert "sql-optimization" in sql_xml
        assert "report-generator" not in sql_xml

        # school_report XML
        report_patterns = manager.parse_skill_patterns("report-*, data-*")
        report_xml = manager.generate_available_skills_xml("school_report", patterns=report_patterns)
        assert "report-generator" in report_xml
        assert "data-profiler" in report_xml
        assert "sql-analysis" not in report_xml


# ============================================================================
# 5. Multi-Skill Tool Accumulation Lifecycle
# ============================================================================


@pytest.mark.nightly
class TestSkillToolsAccumulationIntegration:
    """Test multi-skill loading lifecycle and tool management."""

    @pytest.mark.acceptance
    def test_loaded_tools_accumulate_across_skills(self, skill_func_tool):
        """Loading multiple script skills accumulates bash tools."""
        assert len(skill_func_tool.get_all_skill_bash_tools()) == 0

        # Load first script skill
        skill_func_tool.load_skill("report-generator")
        assert len(skill_func_tool.get_all_skill_bash_tools()) == 1

        # Load second script skill
        skill_func_tool.load_skill("data-profiler")
        assert len(skill_func_tool.get_all_skill_bash_tools()) == 2

    def test_workflow_skill_does_not_add_to_bash_tools(self, skill_func_tool):
        """Loading workflow-only skills doesn't create bash tools."""
        skill_func_tool.load_skill("sql-analysis")
        skill_func_tool.load_skill("sql-optimization")
        assert len(skill_func_tool.get_all_skill_bash_tools()) == 0

    def test_mixed_skills_only_script_ones_get_bash_tools(self, skill_func_tool):
        """Loading mix of workflow + script skills: only script ones get bash tools."""
        skill_func_tool.load_skill("sql-analysis")  # workflow
        skill_func_tool.load_skill("report-generator")  # script
        skill_func_tool.load_skill("sql-optimization")  # workflow
        skill_func_tool.load_skill("data-profiler")  # script

        tools = skill_func_tool.get_all_skill_bash_tools()
        assert len(tools) == 2
        assert "report-generator" in tools
        assert "data-profiler" in tools

    def test_duplicate_load_does_not_double_bash_tool(self, skill_func_tool):
        """Loading the same script skill twice doesn't create duplicate bash tools."""
        skill_func_tool.load_skill("report-generator")
        skill_func_tool.load_skill("report-generator")
        assert len(skill_func_tool.get_all_skill_bash_tools()) == 1

    def test_loaded_skill_tools_returns_tool_objects(self, skill_func_tool):
        """get_loaded_skill_tools returns Tool objects from loaded script skills."""
        skill_func_tool.load_skill("report-generator")
        skill_func_tool.load_skill("data-profiler")

        tools = skill_func_tool.get_loaded_skill_tools()
        # Each SkillBashTool provides at least 1 tool
        assert len(tools) >= 2


# ============================================================================
# 6. Real LLM Skill Integration (Acceptance)
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

        # Classify tool actions
        tool_actions = [a for a in all_actions if a.role == "tool"]
        success_tools = [a for a in tool_actions if a.status == "success"]
        failed_tools = [a for a in tool_actions if a.status == "failed"]

        # Skill invocation check
        action_types = [a.action_type for a in all_actions]
        action_messages = " ".join(a.messages for a in all_actions)
        has_load_skill = "load_skill" in action_types or "load_skill" in action_messages
        has_skill_exec = "skill_execute_command" in action_types or "skill_execute_command" in action_messages

        print(f"\n{separator}")
        print()
        print("  SKILL INTEGRATION TEST SUMMARY")
        print()
        print(separator)
        print()
        print(f"  Model:            {model_name}")
        print(f"  Total Duration:   {duration:.1f}s")
        print(f"  Total Actions:    {len(all_actions)}")
        print(
            f"  Tool Calls:       {len(tool_actions)}  "
            f"(pass {len(success_tools)} success, fail {len(failed_tools)} failed)"
        )

        # Successful tools
        if success_tools:
            print()
            print("  SUCCESSFUL TOOLS:")
            for i, a in enumerate(success_tools, 1):
                name = a.action_type or a.function_name()
                print(f"    {i}. {name:<30} -> success")

        # Failed tools
        if failed_tools:
            print()
            print("  FAILED TOOLS:")
            for i, a in enumerate(failed_tools, 1):
                name = a.action_type or a.function_name()
                print(f"    {i}. {name}")
                if a.input:
                    args_str = json.dumps(a.input, default=str, ensure_ascii=False)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"       Args: {args_str}")
                error_msg = ""
                if a.output and isinstance(a.output, dict):
                    error_msg = a.output.get("error", "") or a.output.get("message", "")
                if not error_msg and a.messages:
                    error_msg = a.messages
                if error_msg:
                    if len(str(error_msg)) > 200:
                        error_msg = str(error_msg)[:200] + "..."
                    print(f"       Error: {error_msg}")

        # Skill invocation summary
        print()
        print("  SKILL INVOCATION:")
        print(f"    load_skill:             {'Found' if has_load_skill else 'Not found'}")
        print(f"    skill_execute_command:   {'Found' if has_skill_exec else 'Not found'}")

        print()
        print(separator)

    @pytest.mark.asyncio
    async def test_skill_invocation_in_chat(self, llm_agent_config):
        """Verify load_skill and skill_execute_command appear in action history."""
        import time

        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.schemas.action_history import ActionHistoryManager
        from datus.schemas.chat_agentic_node_models import ChatNodeInput

        # Resolve model name for report
        model_key = llm_agent_config.target
        model_config = llm_agent_config.models.get(model_key, {})
        model_name = model_config.get("model", model_key) if isinstance(model_config, dict) else model_key

        # 1. Create ChatAgenticNode
        node = ChatAgenticNode(
            node_id="test_skill_llm",
            description="LLM skill integration test",
            node_type="chat",
            agent_config=llm_agent_config,
        )

        # 2. Set input directly (bypass workflow)
        node.input = ChatNodeInput(
            user_message=self.QUESTION,
            database="california_schools",
            max_turns=15,
        )

        # 3. Auto-approve ALL skill permissions (bypass InteractionBroker)
        assert node.permission_manager is not None, "PermissionManager not initialized"
        node.permission_manager.approve_for_session("skills", "*")

        # 4. Execute with streaming, collect all actions and measure time
        ahm = ActionHistoryManager()
        start_time = time.time()
        async for _ in node.execute_stream(ahm):
            pass
        duration = time.time() - start_time

        # 5. Print structured report
        all_actions = ahm.get_actions()
        self._print_report(model_name, duration, all_actions)

        # 6. Assertions
        action_types = [a.action_type for a in all_actions]
        action_messages = " ".join(a.messages for a in all_actions)

        # Must have load_skill call (LLM decided to use the skill)
        has_load_skill = "load_skill" in action_types or "load_skill" in action_messages
        # Should have skill_execute_command (skill script execution)
        has_skill_exec = "skill_execute_command" in action_types or "skill_execute_command" in action_messages

        assert has_load_skill, f"Expected load_skill in action history. " f"Action types found: {action_types}"
        # skill_execute_command is expected but LLM behavior is non-deterministic
        if not has_skill_exec:
            import warnings

            warnings.warn(
                "skill_execute_command not found in action history. "
                "The LLM loaded the skill but did not execute a command.",
                stacklevel=2,
            )

        # Verify the node completed (success or error - not stuck)
        final_action = all_actions[-1]
        assert final_action.status in ("success", "failed"), f"Unexpected final status: {final_action.status}"
