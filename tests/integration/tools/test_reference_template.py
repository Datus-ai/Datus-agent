# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for reference template tools.

Nightly tests require LLM API keys and run the full bootstrap → search → get → render pipeline.
CI tests verify tool availability using pre-built test data (from build_scripts/build_test_data.sh).
"""

import json
from pathlib import Path

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.tools.func_tool.reference_template_tools import ReferenceTemplateTools

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLE_TEMPLATE_DIR = str(PROJECT_ROOT / "sample_data" / "california_schools" / "reference_template")

# ============================================================================
# Nightly: Full bootstrap → tools pipeline
# ============================================================================


@pytest.mark.nightly
class TestReferenceTemplateBootstrap:
    """Nightly: Bootstrap J2 template files into vector DB via LLM summary generation."""

    def test_bootstrap_overwrite(self, agent_config: AgentConfig):
        """Bootstrap should process 3 J2 files (4 templates) and store to vector DB."""
        from datus.storage.reference_template.reference_template_init import init_reference_template
        from datus.storage.reference_template.store import ReferenceTemplateRAG

        storage = ReferenceTemplateRAG(agent_config)
        storage.truncate()

        result = init_reference_template(
            storage,
            agent_config,
            SAMPLE_TEMPLATE_DIR,
            build_mode="overwrite",
            subject_tree=[
                "california_schools/Free_Rate/Query",
                "california_schools/Charter/Zip",
                "california_schools/SAT_Score/Phone",
                "california_schools/Enrollment/Summary",
            ],
        )

        assert result["status"] == "success", f"Bootstrap failed: {result}"
        assert result["valid_entries"] == 4, (
            f"Expected 4 valid templates (3 files, 1 multi), got {result['valid_entries']}"
        )
        assert result["processed_entries"] >= 1, f"Expected at least 1 processed, got {result['processed_entries']}"
        assert storage.get_reference_template_size() >= 1, "Storage should have at least 1 template after bootstrap"

    def test_bootstrap_incremental_skips_existing(self, agent_config: AgentConfig):
        """Incremental mode should skip already-bootstrapped templates."""
        from datus.storage.reference_template.reference_template_init import init_reference_template
        from datus.storage.reference_template.store import ReferenceTemplateRAG

        storage = ReferenceTemplateRAG(agent_config)
        size_before = storage.get_reference_template_size()

        # Incremental should skip all existing entries
        result = init_reference_template(
            storage,
            agent_config,
            SAMPLE_TEMPLATE_DIR,
            build_mode="incremental",
        )

        assert result["status"] == "success"
        assert storage.get_reference_template_size() >= size_before


@pytest.mark.nightly
class TestReferenceTemplateToolsNightly:
    """Nightly: Verify tools work against bootstrapped data."""

    @pytest.fixture
    def tpl_tools(self, agent_config: AgentConfig):
        return ReferenceTemplateTools(agent_config)

    def test_has_reference_templates(self, tpl_tools):
        """After bootstrap, has_reference_templates should be True."""
        assert tpl_tools.has_reference_templates is True, "Should have templates after bootstrap"

    def test_search_reference_template(self, tpl_tools):
        """Search should find templates by natural language query."""
        result = tpl_tools.search_reference_template("free rate school")
        assert result.success == 1, f"Search failed: {result.error}"
        assert isinstance(result.result, list)
        assert len(result.result) > 0, "Should find at least one template"

        first = result.result[0]
        assert "name" in first, f"Result should have 'name', got keys: {list(first.keys())}"
        assert "template" in first, "Result should have 'template'"
        assert "parameters" in first, "Result should have 'parameters'"

        # Parameters should be a valid JSON string
        params = json.loads(first["parameters"])
        assert isinstance(params, list), "Parameters should be a list"
        assert all("name" in p for p in params), "Each parameter should have a 'name' field"

    def test_get_reference_template(self, tpl_tools):
        """Get should retrieve a specific template with full details."""
        # First search to find a valid subject_path + name
        search_result = tpl_tools.search_reference_template("school")
        assert search_result.success == 1 and len(search_result.result) > 0

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")
        assert subject_path and name, f"Need subject_path and name, got: {first}"

        get_result = tpl_tools.get_reference_template(subject_path=subject_path, name=name)
        assert get_result.success == 1, f"Get failed: {get_result.error}"
        assert get_result.result is not None
        assert "template" in get_result.result
        assert "parameters" in get_result.result

    def test_render_reference_template(self, tpl_tools):
        """Render should produce valid SQL from template + params."""
        search_result = tpl_tools.search_reference_template("school")
        assert search_result.success == 1 and len(search_result.result) > 0

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")
        params_json = first.get("parameters", "[]")

        # Build params dict with placeholder values
        params = json.loads(params_json)
        render_params = {}
        for p in params:
            param_name = p["name"]
            # Use reasonable placeholder values
            if "limit" in param_name or "top" in param_name or "count" in param_name or "taker" in param_name:
                render_params[param_name] = "10"
            elif "order" in param_name:
                render_params[param_name] = "ASC"
            elif "type" in param_name or "status" in param_name:
                render_params[param_name] = "Active"
            else:
                render_params[param_name] = "test_value"

        render_result = tpl_tools.render_reference_template(
            subject_path=subject_path,
            name=name,
            params=json.dumps(render_params),
        )
        assert render_result.success == 1, f"Render failed: {render_result.error}"
        rendered_sql = render_result.result["rendered_sql"]
        assert "SELECT" in rendered_sql.upper(), "Rendered SQL should contain SELECT"
        # Template variables should be replaced
        assert "{{" not in rendered_sql, "Rendered SQL should not contain {{ }} placeholders"

    def test_render_missing_params_gives_helpful_error(self, tpl_tools):
        """Render with missing params should return actionable error for LLM retry."""
        search_result = tpl_tools.search_reference_template("school")
        assert search_result.success == 1 and len(search_result.result) > 0

        first = search_result.result[0]
        subject_path = first.get("subject_path", [])
        name = first.get("name", "")
        params_json = first.get("parameters", "[]")
        params = json.loads(params_json)

        if len(params) > 1:
            # Only provide the first parameter, omit the rest
            partial_params = {params[0]["name"]: "test_value"}
            render_result = tpl_tools.render_reference_template(
                subject_path=subject_path,
                name=name,
                params=json.dumps(partial_params),
            )
            assert render_result.success == 0, "Render with missing params should fail"
            # Error should mention which params are missing
            assert "requires parameters" in render_result.error or "Missing" in render_result.error
            assert "retry" in render_result.error.lower()
        else:
            pytest.skip("No template with >1 parameter found to test missing params scenario")


@pytest.mark.nightly
class TestSubjectTreeIncludesTemplates:
    """Nightly: Verify reference_template appears in the subject tree."""

    def test_list_subject_tree_has_reference_template(self, agent_config: AgentConfig):
        """list_subject_tree should include reference_template entries."""
        ctx_tools = ContextSearchTools(agent_config)
        tree_result = ctx_tools.list_subject_tree()
        assert tree_result.success == 1, f"list_subject_tree failed: {tree_result.error}"

        # Walk tree to find reference_template leaves
        def find_key_in_tree(tree, target_key):
            for key, value in tree.items():
                if key == target_key:
                    return True
                if isinstance(value, dict):
                    if find_key_in_tree(value, target_key):
                        return True
            return False

        has_templates = find_key_in_tree(tree_result.result, "reference_template")
        assert has_templates, (
            f"Subject tree should contain 'reference_template' entries. Tree: {json.dumps(tree_result.result, indent=2)}"
        )


@pytest.mark.nightly
class TestNodeToolWiring:
    """Nightly: Verify reference_template_tools can be wired into agentic nodes."""

    def test_gen_sql_node_with_template_tools(self, agent_config: AgentConfig):
        """GenSQLAgenticNode should load reference_template_tools when configured."""
        from unittest.mock import patch

        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        # Inject a node config that includes reference_template_tools
        agent_config.agentic_nodes["test_template_node"] = {
            "model": "deepseek",
            "system_prompt": "gen_sql",
            "prompt_language": "en",
            "max_turns": 5,
            "tools": "db_tools.*, reference_template_tools.*",
        }

        with patch("datus.models.base.LLMBaseModel.create_model") as mock_create:
            mock_model = mock_create.return_value
            mock_model.model_config = type("Config", (), {"model": "test"})()
            mock_model.context_length.return_value = 8000

            node = GenSQLAgenticNode(
                node_id="test_tpl_node",
                description="Test template node",
                node_type=NodeType.TYPE_GENSQL,
                agent_config=agent_config,
                node_name="test_template_node",
            )

        assert node.reference_template_tools is not None, "reference_template_tools should be initialized"
        tool_names = [t.name for t in node.tools]
        assert "search_reference_template" in tool_names, f"Missing search tool, available: {tool_names}"
        assert "get_reference_template" in tool_names, f"Missing get tool, available: {tool_names}"
        assert "render_reference_template" in tool_names, f"Missing render tool, available: {tool_names}"


# ============================================================================
# CI: Quick sanity checks using pre-built data
# ============================================================================


@pytest.mark.ci
class TestReferenceTemplateToolsCI:
    """CI-level tests: verify tools are functional with pre-built test data.

    Pre-built data is created by build_scripts/build_test_data.sh.
    These tests skip gracefully if data hasn't been bootstrapped yet.
    """

    @pytest.fixture
    def tpl_tools(self, agent_config: AgentConfig):
        tools = ReferenceTemplateTools(agent_config)
        if not tools.has_reference_templates:
            pytest.skip("No reference template data available (run build_scripts/build_test_data.sh first)")
        return tools

    def test_search_returns_results(self, tpl_tools):
        """search_reference_template should return structured results."""
        result = tpl_tools.search_reference_template("school")
        assert result.success == 1
        assert isinstance(result.result, list)
        assert len(result.result) > 0

    def test_get_returns_template_with_params(self, tpl_tools):
        """get_reference_template should return template content and parameters."""
        search_result = tpl_tools.search_reference_template("school")
        assert search_result.success == 1, f"Search failed: {search_result.error}"
        assert len(search_result.result) > 0, "Search returned no results"
        first = search_result.result[0]
        get_result = tpl_tools.get_reference_template(first["subject_path"], first["name"])
        assert get_result.success == 1
        assert "template" in get_result.result
        assert "parameters" in get_result.result

    def test_render_produces_sql(self, tpl_tools):
        """render_reference_template should produce SQL without Jinja2 placeholders."""
        search_result = tpl_tools.search_reference_template("school")
        assert search_result.success == 1, f"Search failed: {search_result.error}"
        assert len(search_result.result) > 0, "Search returned no results"
        first = search_result.result[0]
        params = json.loads(first.get("parameters", "[]"))
        render_params = {p["name"]: "test_value" for p in params}

        render_result = tpl_tools.render_reference_template(
            first["subject_path"], first["name"], json.dumps(render_params)
        )
        assert render_result.success == 1
        assert "{{" not in render_result.result["rendered_sql"]
