import os
from pathlib import Path
from typing import List, Optional

import yaml

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.node_models import SqlTask
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def load_builtin_workflow_config() -> dict:
    current_dir = Path(__file__).parent
    config_path = current_dir / "workflow.yml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Workflow configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Workflow configuration loaded: {config_path}")

    return config


def create_nodes_from_config(
    workflow_config: list, sql_task: SqlTask, agent_config: Optional[AgentConfig] = None
) -> List[Node]:
    nodes = []

    start_node = Node.new_instance(
        node_id="node_0",
        description=NodeType.get_description(NodeType.TYPE_BEGIN),
        node_type=NodeType.TYPE_BEGIN,
        input_data=sql_task,
        agent_config=agent_config,
    )
    nodes.append(start_node)

    for index, node_type in enumerate(workflow_config, start=1):
        node_id = f"node_{index}"
        description = NodeType.get_description(node_type)

        input_data = None
        if node_type == NodeType.TYPE_SCHEMA_LINKING:
            input_data = SchemaLinkingInput.from_sql_task(
                sql_task=sql_task,
                matching_rate=agent_config.schema_linking_rate if agent_config else "fast",
            )

        node = Node.new_instance(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
        )

        nodes.append(node)

    logger.info(f"Generated workflow with {len(nodes)} nodes")

    return nodes


def generate_workflow(
    task: SqlTask,
    plan_type: str = "reflection",
    agent_config: Optional[AgentConfig] = None,
) -> Workflow:
    logger.info(f"Generating workflow for task based on plan type '{plan_type}': {task}")

    if not plan_type and agent_config:
        plan_type = agent_config.workflow_plan
    elif not plan_type:
        plan_type = "reflection"  # fallback to default

    if agent_config and plan_type in agent_config.custom_workflows:
        logger.info(f"Using custom workflow '{plan_type}' from configuration")
        selected_workflow = agent_config.custom_workflows[plan_type]
    else:
        # Check builtin workflows
        config = load_builtin_workflow_config()
        workflows = config.get("workflow", {})

        if plan_type not in workflows:
            if agent_config and agent_config.custom_workflows:
                available_custom = list(agent_config.custom_workflows.keys())
                available_builtin = list(workflows.keys())
                raise ValueError(
                    f"Invalid plan type '{plan_type}'. "
                    f"Available builtin workflows: {available_builtin}, "
                    f"custom workflows: {available_custom}"
                )
            else:
                available_builtin = list(workflows.keys())
                raise ValueError(f"Invalid plan type '{plan_type}'. Available builtin workflows: {available_builtin}")

        selected_workflow = workflows[plan_type]

    workflow = Workflow(
        name=f"SQL Query Workflow ({plan_type})",
        task=task,
        agent_config=agent_config,
    )

    nodes = create_nodes_from_config(selected_workflow, task, agent_config)

    for node in nodes:
        workflow.add_node(node)

    logger.info(f"Generated workflow with {len(nodes)} nodes")
    return workflow
