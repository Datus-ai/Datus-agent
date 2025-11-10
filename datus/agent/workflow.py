# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import Context, SQLContext, SqlTask
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from .node import Node

logger = get_logger(__name__)


class Workflow:
    """
    Represents a workflow of nodes for AI and human collaboration.
    """

    def __init__(
        self,
        name: str,
        task: Optional[SqlTask] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize a workflow with metadata.

        Args:
            name: Name of the workflow, for example "SQL-workload_id"
            task: SqlTask object containing task details
            db_conn: Database connection object that has been validated by the agent
        """
        self.name = name
        self.task = task
        self.workflow_config = None
        self.metadata = {}

        self.nodes = {}  # Map of node_id to Node objects
        self.node_order = []  # List of node IDs in execution order
        self.current_node_index = 0
        self.status = "pending"  # pending, running, completed, failed, paused
        self.reflection_round = 0
        self.creation_time = time.time()
        self.completion_time = None

        self.context = Context()
        self.llms = {}
        self._global_config = agent_config
        self._init_tools()

    @property
    def global_config(self) -> AgentConfig:
        return self._global_config

    @global_config.setter
    def global_config(self, value: AgentConfig):
        self._global_config = value
        if self.nodes.values():
            for node in self.nodes.values():
                node.global_config = value

    def add_node(self, node: "Node", position: Optional[int] = None) -> str:
        if position is not None:
            if not isinstance(position, int):
                raise ValueError(f"Position must be integer, got {type(position)}")

        self.nodes[node.id] = node
        node.workflow = self

        if position is not None and 0 <= position <= len(self.node_order):
            self.node_order.insert(position, node.id)
        else:
            self.node_order.append(node.id)

        logger.info(f"Added node '{node.description}' to workflow")
        return node.id

    def get_last_sqlcontext(self) -> SQLContext:
        """Get the last SQL context from the workflow.

        Returns:
            SQLContext: The last SQL context

        Raises:
            DatusException: If no SQL context is available
        """
        if len(self.context.sql_contexts) > 0:
            return self.context.sql_contexts[-1]
        raise DatusException(ErrorCode.NODE_NO_SQL_CONTEXT)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the workflow.

        Args:
            node_id: ID of the node to remove

        Returns:
            True if the node was removed, False otherwise
        """
        if node_id in self.nodes:
            # Remove the node from the node map
            del self.nodes[node_id]

            # Remove the node from the node order
            if node_id in self.node_order:
                self.node_order.remove(node_id)

            # Update dependencies in other nodes
            for node in self.nodes.values():
                if node_id in node.dependencies:
                    node.dependencies.remove(node_id)

            logger.info(f"Removed node '{node_id}' from workflow")
            return True

        return False

    def move_node(self, node_id: str, new_position: int) -> bool:
        """
        Move a node to a new position in the node order.

        Args:
            node_id: ID of the node to move
            new_position: New position in the node order

        Returns:
            True if the node was moved, False otherwise
        """
        if node_id in self.node_order and 0 <= new_position < len(self.node_order):
            # Remove the node from its current position
            self.node_order.remove(node_id)

            # Insert the node at the new position
            self.node_order.insert(new_position, node_id)

            logger.info(f"Moved node '{node_id}' to position {new_position}")
            return True

        return False

    def get_node(self, node_id: str) -> Optional["Node"]:
        """
        Get a node by its ID.

        Args:
            node_id: ID of the node to get

        Returns:
            The node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_last_node_by_type(self, node_type: str) -> Optional["Node"]:
        """
        Get the last node of a specific type.
        Args:
            node_type: Type of the node to get
        Returns:
            The last node of the specified type if found, None otherwise
        """
        for nid in reversed(range(self.current_node_index)):
            node = self.nodes[self.node_order[nid]]
            if node.type == node_type:
                return node
        return None

    def get_current_node(self) -> Optional["Node"]:
        """
        Get the current node to execute.

        Returns:
            The current node if available, None otherwise
        """
        if self.current_node_index < len(self.node_order):
            node_id = self.node_order[self.current_node_index]
            return self.nodes[node_id]
        return None

    def get_next_node(self) -> Optional["Node"]:
        """
        Get the next node to execute.

        Returns:
            The next node if available, None otherwise
        """
        if len(self.node_order) > 0 and self.current_node_index + 1 < len(self.node_order):
            node_id = self.node_order[self.current_node_index + 1]
            return self.nodes.get(node_id)
        return None

    def advance_to_next_node(self) -> Optional["Node"]:
        """
        Advance to the next node in the workflow and set the next node input from the previous node result.

        Returns:
            The next node if available, None otherwise.
        """
        if self.current_node_index >= len(self.node_order) - 1:
            self.status = "completed"
            self.completion_time = time.time()
            return None
        self.current_node_index += 1
        next_node_id = self.node_order[self.current_node_index]
        return self.nodes.get(next_node_id)

    def is_complete(self) -> bool:
        """
        Check if the workflow is complete.

        Returns:
            True if the workflow is complete, False otherwise
        """
        return self.status == "completed" or self.current_node_index >= len(self.node_order)

    def pause(self, checkpoint_path: str = ""):
        """
        Pause the workflow execution.
        """
        self.status = "paused"
        if checkpoint_path:
            self.save(checkpoint_path)
        logger.info("Workflow paused")

    def resume(self, checkpoint_path: str = ""):
        """
        Resume the workflow execution.
        """
        self.status = "running"
        if checkpoint_path:
            self.load(checkpoint_path, self.global_config)
        logger.info("Workflow resumed")

    def reset(self):
        """
        Reset the workflow to its initial state.
        """
        self.current_node_index = 0
        self.status = "pending"

        # Reset all nodes
        for node in self.nodes.values():
            node.status = "pending"
            node.result = None
            node.start_time = None
            node.end_time = None

        logger.info("Workflow reset")

    def adjust_nodes(self, suggestions: List[Dict]):
        """
        Adjust the workflow based on suggestions.

        Args:
            suggestions: List of node adjustment suggestions
        """
        for suggestion in suggestions:
            action = suggestion.get("action")
            node_id = suggestion.get("node_id")

            if action == "add" and "node" in suggestion:
                # Add a new node
                node_data = suggestion["node"]
                node = Node.new_instance(
                    node_id=node_data.get("id"),
                    description=node_data.get("description"),
                    node_type=node_data.get("type"),
                    input_data=node_data.get("input"),
                    agent_config=self.global_config,
                    tools=self.tools,
                )
                position = suggestion.get("position")
                self.add_node(node, position)

            elif action == "remove" and node_id:
                # Remove a node
                self.remove_node(node_id)

            elif action == "move" and node_id and "position" in suggestion:
                # Move a node
                self.move_node(node_id, suggestion["position"])

            elif action == "modify" and node_id and "modifications" in suggestion:
                # Modify a node
                node = self.get_node(node_id)
                if node:
                    modifications = suggestion["modifications"]

                    if "description" in modifications:
                        node.description = modifications["description"]

                    if "type" in modifications:
                        node.type = modifications["type"]

                    if "input" in modifications:
                        node.input = modifications["input"]

                    if "dependencies" in modifications:
                        node.dependencies = modifications["dependencies"]

        logger.info(f"Adjusted workflow with {len(suggestions)} suggestions")

    def get_final_result(self) -> Dict:
        """
        Get the final result of the workflow.

        Returns:
            Dictionary containing the workflow results
        """
        results = {
            "name": self.name,
            "description": self.task.task,
            "status": self.status,
            "nodes": {},
        }

        # Collect results from all completed nodes
        for node_id, node in self.nodes.items():
            if node.status == "completed":
                results["nodes"][node_id] = {
                    "description": node.description,
                    "result": node.result,
                }

        # If the last node is completed, use its result as the main result
        if self.node_order and self.nodes[self.node_order[-1]].status == "completed":
            results["final_result"] = self.nodes[self.node_order[-1]].result

        return results

    def display(self):
        """
        Display the workflow as a mindmap.
        This is a placeholder implementation that logs the workflow structure.
        In a real implementation, this would generate a visual representation.
        """
        logger.info(f"Workflow: {self.name} - {self.task.task if self.task else 'No task defined'}")
        logger.info(f"Status: {self.status}")

        MAX_LENGTH = 1000

        def truncate_str(s: Any) -> str:
            if s is None:
                return "None"

            s_str = ""
            if hasattr(s, "compact_result") and callable(s.compact_result):
                s_str = s.compact_result()
            else:
                s_str = str(s)
            if len(s_str) > MAX_LENGTH:
                return s_str[:MAX_LENGTH] + "... (truncated)"
            return s_str

        for i, node_id in enumerate(self.node_order):
            node = self.nodes[node_id]
            status_marker = "→" if i == self.current_node_index else " "
            completion_marker = "✓" if node.status == "completed" else "✗" if node.status == "failed" else " "

            # Define ANSI color codes for terminal output
            CYAN = "\033[96m"
            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            BOLD = "\033[1m"

            # Set color based on node status
            status_color = {
                "completed": GREEN,  # Green for completed nodes
                "failed": RED,  # Red for failed nodes
                "pending": YELLOW,  # Yellow for pending nodes
                "running": CYAN,  # Cyan for running nodes
            }.get(node.status, RESET)

            # Format and output node information
            logger.info(
                f"{BOLD}{status_marker}{RESET} "  # Current node indicator
                f"[{i:02d}] "  # Node index (2-digit format)
                f"{completion_marker} "  # Completion marker (✓/✗)
                f"{BOLD}{node.description}{RESET} "  # Node description (bold)
                f"({CYAN}{node.type}{RESET}) "  # Node type (cyan)
                f"\n    {YELLOW}Status:{RESET} {status_color}{node.status}{RESET}"  # Node status
                f"\n    {YELLOW}Input:{RESET} {truncate_str(node.input)}"  # Node input
                f"\n    {YELLOW}Result:{RESET} {truncate_str(node.result)}"  # Node result
            )

        # In a real implementation, this would generate and display a mindmap visualization
        # For example, using a library like graphviz or a web-based visualization

    def to_dict(self) -> Dict:
        """
        Convert the workflow to a dictionary representation.

        Returns:
            Dictionary representation of the workflow
        """
        return {
            "name": self.name,
            "description": self.task.task if self.task else "",
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "node_order": self.node_order,
            "task": self.task.to_dict() if self.task else None,
            "current_node_index": self.current_node_index,
            "status": self.status,
            "creation_time": self.creation_time,
            "completion_time": self.completion_time,
            "metadata": self.metadata,
            "context": self.context.to_dict(),
            # "db_conn": self.db_conn.to_dict() if self.db_conn else None,
        }

    def get_task(self) -> Optional[str]:
        """
        Get the task description associated with the workflow.
        Returns:
            The task if available, None otherwise
        """
        return self.task.task if self.task else None

    def save(self, file_path: str):
        """
        Save the workflow to a YAML file.

        Args:
            file_path: Path to save the YAML file
        """
        import yaml

        workflow_dict = self.to_dict()

        # Convert nodes to list format for better YAML readability
        workflow_dict["nodes"] = [{"id": node_id, **node_data} for node_id, node_data in workflow_dict["nodes"].items()]

        with open(file_path, "w") as f:
            # Convert Pydantic models to dictionaries before YAML serialization
            from enum import Enum

            def convert_pydantic(obj):
                if hasattr(obj, "model_dump"):  # Pydantic v2
                    return obj.model_dump()
                elif hasattr(obj, "dict"):  # Pydantic v1
                    return obj.dict()
                elif isinstance(obj, Enum):  # Handle enum objects
                    return obj.value
                elif isinstance(obj, (list, tuple)):
                    return [convert_pydantic(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_pydantic(v) for k, v in obj.items()}
                return obj

            yaml.safe_dump(
                {"workflow": convert_pydantic(workflow_dict)},
                f,
                default_flow_style=False,
                indent=2,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def load(cls, file_path: str, agent_config: Optional[AgentConfig] = None) -> "Workflow":
        """
        Load a workflow from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            The loaded Workflow instance
        """
        import yaml

        from datus.schemas.node_models import SqlTask

        from .node import Node

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        workflow_data = data["workflow"]

        # Convert nodes list back to dict format
        nodes = {node["id"]: Node.from_dict(node, agent_config) for node in workflow_data["nodes"]}
        # FIXME global_config

        # Create task from description if there's no task
        task = None
        if workflow_data.get("task"):
            task = SqlTask.from_dict(workflow_data["task"])
        elif workflow_data.get("description"):
            task = SqlTask(task=workflow_data["description"])

        # Initialize workflow with name and task
        workflow = cls(
            name=workflow_data["name"],
            task=task,
            # db_conn=workflow_data.get("db_conn")
        )

        workflow.nodes = nodes
        workflow.node_order = workflow_data["node_order"]
        workflow.current_node_index = workflow_data["current_node_index"]
        workflow.status = workflow_data["status"]
        workflow.creation_time = workflow_data["creation_time"]
        workflow.completion_time = workflow_data.get("completion_time")
        workflow.metadata = workflow_data.get("metadata", {})

        workflow.context = Context.from_dict(workflow_data["context"])

        return workflow

    def _init_tools(self):
        from datus.tools.func_tool import db_function_tools

        self.tools = db_function_tools(self._global_config, self.task.database_name)
