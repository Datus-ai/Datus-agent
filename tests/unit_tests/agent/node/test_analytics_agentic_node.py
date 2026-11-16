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
