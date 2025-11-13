# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def setup_node_input(node, workflow):
    """Sets up the input for a node based on its type."""
    node_type = node.type

    if node_type in NodeType.ACTION_TYPES or node_type in NodeType.CONTROL_TYPES:
        return node.setup_input(workflow)
    else:
        logger.warning(f"Unknown node type for setup input: {node_type}")
        return {
            "success": False,
            "message": f"Unknown node type: {node_type}",
            "suggestions": [],
        }


def update_context_from_node(node: Node, workflow: Workflow) -> Dict:
    if (
        node.type in NodeType.ACTION_TYPES
        or node.type == NodeType.TYPE_REFLECT
        or node.type == NodeType.TYPE_PARALLEL
        or node.type == NodeType.TYPE_SELECTION
        or node.type == NodeType.TYPE_SUBWORKFLOW
    ):
        result = node.update_context(workflow)
        logger.info(f"update_context_from_node: node_type={node.type}, result={result}")
        return result
    else:
        logger.warning(f"Unknown node type for context updating: {node.type}")
        return {"success": False, "message": f"Unknown node type: {node.type}"}


def evaluate_result(node: Node, workflow: Workflow) -> Dict:
    """
    Evaluate the result of a node execution and setup input for the next node.

    Args:
        result: The result of the node execution
        node: The node that was executed
        workflow: The workflow contains all the context and next node

    Returns:
        Evaluation result with success flag and suggestions
    """
    try:
        # Update context from previous node
        update_result = update_context_from_node(node, workflow)

        # skip the error. continue. because it's not a blocker for the workflow
        if not update_result["success"]:
            logger.warning(f"Failed to update context from node {node.id}: {update_result['message']}")

        # Set up the next node input
        next_node = workflow.get_next_node()
        if next_node:
            return setup_node_input(next_node, workflow)
        else:
            return {"success": True, "message": "Last node, finished"}
    except Exception as e:
        logger.error(f"Failed to evaluate result: {e}")
        return {"success": False, "message": f"Evaluation failed: {str(e)}"}
