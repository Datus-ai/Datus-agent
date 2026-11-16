"""Analytics nodes retain host tool scope through creation and rebuilds."""

import pytest

from datus.agent.node.node import Node
from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.configuration.node_type import NodeType
from datus.schemas.chat_agentic_node_models import ChatNodeInput


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize("factory", ["interactive", "workflow"])
def test_analytics_scope_and_read_only_sql(real_agent_config, mock_llm_create, kind, factory):
    real_agent_config.agentic_nodes = {
        "team_analyst": {
            "node_class": kind,
            "tools": "db_tools.list_tables,db_tools.execute_sql,filesystem_tools.read_file",
            "subagents": "",
            "system_prompt": kind,
        }
    }
    if factory == "interactive":
        node = create_interactive_node("team_analyst", real_agent_config, execution_mode="workflow")
    else:
        node = Node.new_instance(
            "analytics", "Analytics", kind, agent_config=real_agent_config, node_name="team_analyst"
        )
    assert node.get_node_name() == "team_analyst"
    assert isinstance(create_node_input("Explore the data", node), ChatNodeInput)
    assert isinstance(NodeType.type_input(kind, {"user_message": "Explore the data"}), ChatNodeInput)
    for _ in range(2):
        names = {tool.name for tool in node.tools}
        assert {"list_tables", "execute_sql", "read_file"} <= names
        assert not names & {"write_file", "edit_file", "validate_semantic", "task", "bash", "describe_table"}
        result = node.db_func_tool.execute_sql("CREATE TABLE forbidden_write (id INTEGER)")
        assert result.success == 0
        node._rebuild_tools()
    prompt = node._get_system_prompt()
    assert "Analytics Role" in prompt
    assert "read-only" in prompt.lower()


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize("mode", ["interactive", "workflow"])
def test_analytics_interaction_tools_survive_data_tool_filter(real_agent_config, mock_llm_create, kind, mode):
    real_agent_config.agentic_nodes = {"analyst": {"node_class": kind, "tools": "filesystem_tools.read_file"}}
    node = create_interactive_node("analyst", real_agent_config, execution_mode=mode)
    for _ in range(2):
        names = {tool.name for tool in node.tools}
        assert ("ask_user" in names) == (mode == "interactive")
        assert {"todo_list", "todo_read", "todo_write", "todo_update"} <= names
        assert "read_file" in names
        assert "write_file" not in names
        node._rebuild_tools()


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize(
    "prompt_config", [{}, {"system_prompt": None}, {"system_prompt": ""}, {"system_prompt": "chat"}]
)
def test_analytics_prompt_default_and_custom_template(real_agent_config, mock_llm_create, kind, prompt_config):
    real_agent_config.agentic_nodes = {
        "custom_analyst": {"node_class": kind, "tools": "filesystem_tools.read_file", **prompt_config}
    }
    node = create_interactive_node("custom_analyst", real_agent_config, execution_mode="workflow")
    prompt = node._get_system_prompt()
    expected = prompt_config.get("system_prompt") or kind
    assert node.node_config["system_prompt"] == expected
    assert bool(prompt)
    assert ("Analytics Role" in prompt) == (expected == kind)


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize(
    "tools",
    [
        " filesystem_tools.read_file , , filesystem_tools.glob ",
        [" filesystem_tools.read_file ", "", "  ", "filesystem_tools.glob"],
        (" filesystem_tools.read_file ", "", "  ", "filesystem_tools.glob"),
    ],
)
def test_analytics_tool_sequence_keeps_exact_filter(real_agent_config, mock_llm_create, kind, tools):
    real_agent_config.agentic_nodes = {"analyst": {"node_class": kind, "tools": tools}}
    node = create_interactive_node("analyst", real_agent_config, execution_mode="workflow")
    for _ in range(2):
        names = {tool.name for tool in node.tools}
        assert {"read_file", "glob"} <= names
        assert not names & {"write_file", "edit_file", "grep"}
        node._rebuild_tools()


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize(
    "tools", [None, "db_tools.*,semantic_tools.*,filesystem_tools.*,bash_tools.*,sub_agent_tools.*"]
)
def test_analytics_defaults_and_wildcards_respect_type_policy(real_agent_config, mock_llm_create, kind, tools):
    from datus.tools.registry.contracts import tool_definition
    from datus.tools.registry.subagent_policy import tool_reference

    config = {"node_class": kind}
    if tools is not None:
        config["tools"] = tools
    real_agent_config.agentic_nodes = {"analyst": config}
    node = create_interactive_node("analyst", real_agent_config, execution_mode="interactive")
    catalog = {
        f"{family}.{method}"
        for family, group in tool_reference(kind)["tool_types"].items()
        for method in group["tools"]
    }
    for _ in range(2):
        definitions = [tool_definition(tool) for tool in node.tools]
        data_tools = {d.tool_id for d in definitions if d.access != "session" and d.category != "skills"}
        assert data_tools <= catalog
        assert {"db_tools.execute_sql", "filesystem_tools.read_file", "semantic_tools.query_metrics"} <= data_tools
        assert {
            "semantic_tools.describe_concept",
            "semantic_tools.find_concept_paths",
            "semantic_tools.select_records",
        } <= data_tools
        assert ("filesystem_tools.write_file" in data_tools) == (kind == "analytics_builder")
        assert not data_tools & {"semantic_tools.validate_semantic", "filesystem_tools.delete_file"}
        assert "ask_user" in {tool.name for tool in node.tools}
        node._rebuild_tools()


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
def test_analytics_mounts_select_records_only_scope(real_agent_config, mock_llm_create, kind):
    from datus.tools.registry.contracts import tool_definition

    real_agent_config.agentic_nodes = {"analyst": {"node_class": kind, "tools": "semantic_tools.select_records"}}
    node = create_interactive_node("analyst", real_agent_config, execution_mode="workflow")
    for _ in range(2):
        definitions = [tool_definition(tool) for tool in node.tools]
        data_tools = {d.tool_id for d in definitions if d.access != "session" and d.category != "skills"}
        assert data_tools == {"semantic_tools.select_records"}
        node._rebuild_tools()


@pytest.mark.parametrize("kind", ["analytics_explorer", "analytics_builder"])
@pytest.mark.parametrize("tools", ["", " ", []])
def test_explicit_empty_host_scope_does_not_restore_defaults(real_agent_config, mock_llm_create, kind, tools):
    from datus.tools.registry.contracts import tool_definition

    real_agent_config.agentic_nodes = {"analyst": {"node_class": kind, "tools": tools}}
    node = create_interactive_node("analyst", real_agent_config, execution_mode="interactive")
    for _ in range(2):
        definitions = [tool_definition(tool) for tool in node.tools]
        assert all(d.access == "session" or d.category == "skills" for d in definitions)
        assert "ask_user" in {tool.name for tool in node.tools}
        node._rebuild_tools()
