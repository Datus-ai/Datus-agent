# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from pathlib import Path
from typing import List, Optional

import yaml
from agents import Tool

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
    workflow_config: list,
    sql_task: SqlTask,
    agent_config: Optional[AgentConfig] = None,
    tools: Optional[List[Tool]] = None,
) -> List[Node]:
    nodes = []

    start_node = Node.new_instance(
        node_id="node_0",
        description=NodeType.get_description(NodeType.TYPE_BEGIN),
        node_type=NodeType.TYPE_BEGIN,
        input_data=sql_task,
        agent_config=agent_config,
        tools=tools,
    )
    nodes.append(start_node)

    # Process workflow config that may contain nested structures
    processed_nodes = _process_workflow_config(workflow_config, sql_task, agent_config, tools=tools)
    nodes.extend(processed_nodes)

    logger.info(f"Generated workflow with {len(nodes)} nodes")

    return nodes


def _process_workflow_config(
    config: list,
    sql_task: SqlTask,
    agent_config: Optional[AgentConfig] = None,
    start_index: int = 1,
    node_id_prefix: str = "node",
    tools: Optional[List[Tool]] = None,
) -> List[Node]:
    """Process workflow configuration that may contain nested parallel structures"""
    nodes = []
    current_index = start_index

    for item in config:
        if isinstance(item, str):
            # Simple node type
            node_id = f"{node_id_prefix}_{current_index}"
            node = _create_single_node(item, node_id, sql_task, agent_config)
            nodes.append(node)
            current_index += 1

        elif isinstance(item, dict):
            # Handle nested structures like parallel
            for key, value in item.items():
                if key == "parallel" and isinstance(value, list):
                    # Create a parallel node
                    parallel_children = value
                    node_id = f"{node_id_prefix}_{current_index}"

                    from datus.schemas.parallel_node_models import ParallelInput

                    parallel_input = ParallelInput(
                        child_nodes=parallel_children, shared_input=None  # Will be set up during execution
                    )

                    parallel_node = Node.new_instance(
                        node_id=node_id,
                        description=NodeType.get_description(NodeType.TYPE_PARALLEL),
                        node_type=NodeType.TYPE_PARALLEL,
                        input_data=parallel_input,
                        agent_config=agent_config,
                        tools=tools,
                    )
                    nodes.append(parallel_node)
                    current_index += 1

                elif key == "selection":
                    # Create a selection node (if it's specified as dict with criteria)
                    node_id = f"{node_id_prefix}_{current_index}"

                    from datus.schemas.parallel_node_models import SelectionInput

                    selection_criteria = value if isinstance(value, str) else "best_quality"
                    selection_input = SelectionInput(
                        candidate_results={},  # Will be populated during execution
                        selection_criteria=selection_criteria,
                    )

                    selection_node = Node.new_instance(
                        node_id=node_id,
                        description=NodeType.get_description(NodeType.TYPE_SELECTION),
                        node_type=NodeType.TYPE_SELECTION,
                        input_data=selection_input,
                        agent_config=agent_config,
                    )
                    nodes.append(selection_node)
                    current_index += 1
                else:
                    # Handle other dict-based configurations if needed
                    logger.warning(f"Unknown configuration item: {key}")

        elif item == "selection":
            # Simple selection node
            node_id = f"{node_id_prefix}_{current_index}"

            from datus.schemas.parallel_node_models import SelectionInput

            selection_input = SelectionInput(
                candidate_results={}, selection_criteria="best_quality"  # Will be populated during execution
            )

            selection_node = Node.new_instance(
                node_id=node_id,
                description=NodeType.get_description(NodeType.TYPE_SELECTION),
                node_type=NodeType.TYPE_SELECTION,
                input_data=selection_input,
                agent_config=agent_config,
            )
            nodes.append(selection_node)
            current_index += 1

    return nodes


def _create_single_node(
    node_type: str, node_id: str, sql_task: SqlTask, agent_config: Optional[AgentConfig] = None
) -> Node:
    # normalize aliases from config
    normalized_type = node_type
    if node_type in {"reason_sql", "reasoning_sql", "reason"}:
        normalized_type = NodeType.TYPE_REASONING
    elif node_type in {"reflection", "reflect"}:
        normalized_type = NodeType.TYPE_REFLECT
    elif node_type == "execute":
        normalized_type = NodeType.TYPE_EXECUTE_SQL
    elif node_type == "chat":
        normalized_type = NodeType.TYPE_CHAT
    # Check if node_type is defined in agentic_nodes config - if so, map to gensql
    elif agent_config and hasattr(agent_config, "agentic_nodes") and node_type in agent_config.agentic_nodes:
        normalized_type = NodeType.TYPE_GENSQL

    description = NodeType.get_description(normalized_type)

    input_data = None
    if normalized_type == NodeType.TYPE_SCHEMA_LINKING:
        input_data = SchemaLinkingInput.from_sql_task(
            sql_task=sql_task,
            matching_rate=agent_config.schema_linking_rate if agent_config else "fast",
        )

    node = Node.new_instance(
        node_id=node_id,
        description=description,
        node_type=normalized_type,
        input_data=input_data,
        agent_config=agent_config,
        node_name=node_type if normalized_type == NodeType.TYPE_GENSQL else None,  # Pass original name for gensql nodes
    )

    return node


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

    # support { steps: [...] } structure for custom workflows
    workflow_steps = selected_workflow
    workflow_config = None
    if isinstance(selected_workflow, dict):
        if "steps" in selected_workflow:
            workflow_steps = selected_workflow["steps"]
            # Extract config if available
            if "config" in selected_workflow:
                workflow_config = selected_workflow["config"]

    workflow = Workflow(
        name=f"SQL Query Workflow ({plan_type})",
        task=task,
        agent_config=agent_config,
    )

    # Store workflow config in the workflow object if available
    if workflow_config:
        workflow.workflow_config = workflow_config

    nodes = create_nodes_from_config(workflow_steps, task, agent_config, workflow.tools)

    for node in nodes:
        workflow.add_node(node)
    if task.tables and agent_config is not None:
        from datus.storage.schema_metadata import SchemaWithValueRAG

        try:
            rag = SchemaWithValueRAG(agent_config=agent_config)
            schemas, values = rag.search_tables(
                task.tables, task.catalog_name, task.database_name, task.schema_name, dialect=task.database_type
            )
            if len(schemas) != len(task.tables):
                schema_table_names = [item.table_name for item in schemas]
                logger.warning(
                    f"The obtained table schema is: {schema_table_names}; "
                    f"The table required for the task is: {schemas}"
                )
            logger.debug(f"Use task tables: {schemas}")
            workflow.context.update_schema_and_values(schemas, values)
        except Exception as e:
            logger.warning(f"Failed to obtain the schema corresponding to {task.tables}: {e}")

    logger.info(f"Generated workflow with {len(nodes)} nodes")
    return workflow
