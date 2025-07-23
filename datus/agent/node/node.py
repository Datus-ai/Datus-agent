from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.models.base import LLMBaseModel
from datus.schemas.fix_node_models import FixInput
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.schemas.node_models import (
    BaseInput,
    BaseResult,
    ExecuteSQLInput,
    ExecuteSQLResult,
    GenerateSQLInput,
    GenerateSQLResult,
    OutputInput,
    OutputResult,
    ReflectionResult,
)
from datus.schemas.reason_sql_node_models import ReasoningResult
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class Workflow:
    pass


class Node(ABC):
    """
    Represents a single node in a workflow.
    """

    @classmethod
    def new_instance(
        cls,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        from datus.agent.node import (
            BeginNode,
            CompareNode,
            DocSearchNode,
            ExecuteSQLNode,
            FixNode,
            GenerateMetricsNode,
            GenerateSemanticModelNode,
            GenerateSQLNode,
            HitlNode,
            OutputNode,
            ParallelNode,
            ReasonSQLNode,
            ReflectNode,
            SchemaLinkingNode,
            SearchMetricsNode,
            SelectionNode,
        )

        if node_type == NodeType.TYPE_SCHEMA_LINKING:
            return SchemaLinkingNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_GENERATE_SQL:
            return GenerateSQLNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_EXECUTE_SQL:
            return ExecuteSQLNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_REASONING:
            return ReasonSQLNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_DOC_SEARCH:
            return DocSearchNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_OUTPUT:
            return OutputNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_FIX:
            return FixNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_REFLECT:
            return ReflectNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_HITL:
            return HitlNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_BEGIN:
            return BeginNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_GENERATE_METRICS:
            return GenerateMetricsNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_GENERATE_SEMANTIC_MODEL:
            return GenerateSemanticModelNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_SEARCH_METRICS:
            return SearchMetricsNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_PARALLEL:
            return ParallelNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_SELECTION:
            return SelectionNode(node_id, description, node_type, input_data, agent_config)
        elif node_type == NodeType.TYPE_COMPARE:
            return CompareNode(node_id, description, node_type, input_data, agent_config)
        else:
            raise ValueError(f"Invalid node type: {node_type}")

    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: BaseInput = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize a node with its metadata.

        Args:
            node_id: Unique identifier for the node
            description: Human-readable description of the node
            node_type: Type of node (e.g., sql_generation, data_validation)
            input_data: Input data for the node
        """
        if node_type not in NodeType.ACTION_TYPES and node_type not in NodeType.CONTROL_TYPES:
            raise ValueError(f"Invalid node type: {node_type}")

        self.id = node_id
        self.description = description
        self.type = node_type
        self.input = input_data
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.start_time = None
        self.end_time = None
        self.dependencies = []  # IDs of nodes that must complete before this one
        self.metadata = {}
        self.agent_config = agent_config
        self.model = None

    def _initialize(self):
        """Initialize the model for this node"""
        model_name = None
        nodes_config = {}
        if not self.model:
            model_name = None
            nodes_config = self.agent_config.nodes

        if self.type in nodes_config:
            node_config = nodes_config[self.type]
            model_name = node_config.model
            node_input = node_config.input
            for attr, value in node_input.__dict__.items():
                if value is not None:
                    setattr(self.input, attr, value)

        llm_model = LLMBaseModel.create_model(model_name=model_name, agent_config=self.agent_config)
        logger.info(
            f"Initializing model type: {llm_model.model_config.type}"
            f", model name {llm_model.model_config.model} for node {self.type}"
        )

        if (
            hasattr(llm_model, "set_context")
            and hasattr(self, "workflow")
            and self.workflow
            and llm_model.model_config.save_llm_trace
        ):
            llm_model.set_context(workflow=self.workflow, current_node=self)

        self.model = llm_model

    @abstractmethod
    def update_context(self, workflow: Workflow) -> Dict:
        pass

    @abstractmethod
    def setup_input(self, workflow: Workflow) -> Dict:
        pass

    def start(self):
        """
        Mark the node as started.
        """
        self.status = "running"
        self.start_time = time.time()

    def complete(self, result: BaseResult):
        """
        Mark the node as completed with a result.

        Args:
            result: The result of the node execution
        """
        self.status = "completed" if result.success else "failed"
        self.result = result
        self.end_time = time.time()

    def fail(self, error: str = None):
        """
        Mark the node as failed with an error message.

        Args:
            error: The error message explaining the failure
        """
        self.status = "failed"
        if error:
            self.result = BaseResult(success=False, error=error)
        self.end_time = time.time()

    @abstractmethod
    def execute(self) -> BaseResult:
        pass

    def run(self):
        """Execute the node based on its type and update status."""
        try:
            self._initialize()
            self.start()

            if self.type in NodeType.ACTION_TYPES:
                try:
                    self.execute()

                    if self.result is not None and self.result.success:
                        self.complete(self.result)
                    else:
                        logger.error(f"Node execution failed: {self.result}")
                        self.fail(f"Node execution failed: {self.result}")
                except Exception as e:
                    logger.error(f"Node execution failed: {str(e)}")
                    self.fail(str(e))
            elif self.type in NodeType.CONTROL_TYPES:
                if self.type == NodeType.TYPE_REFLECT:
                    self.execute()
                    self.complete(self.result)
                    # strategy_result = execute_strategy(self, strategy, details, model)
                    # logger.info(f"Execute strategy result: {strategy_result}")
                elif self.type == NodeType.TYPE_PARALLEL:
                    self.execute()
                    if self.result is not None and self.result.success:
                        self.complete(self.result)
                    else:
                        logger.error(f"Parallel node execution failed: {self.result}")
                        self.fail(f"Parallel node execution failed: {self.result}")
                elif self.type == NodeType.TYPE_SELECTION:
                    self.execute()
                    if self.result is not None and self.result.success:
                        self.complete(self.result)
                    else:
                        logger.error(f"Selection node execution failed: {self.result}")
                        self.fail(f"Selection node execution failed: {self.result}")
                else:
                    pass
            else:
                raise ValueError(f"Invalid node type: {self.type}")
        except Exception as e:
            logger.error(f"Node execution failed: {str(e)}")
            self.fail(str(e))

        return self.result

    def add_dependency(self, node_id: str):
        """
        Add a dependency to this node.

        Args:
            node_id: ID of the node that must complete before this one
        """
        if node_id not in self.dependencies:
            self.dependencies.append(node_id)

    def _sql_connector(self) -> BaseSqlConnector:
        return db_manager_instance(self.agent_config.namespaces).get_conn(
            self.agent_config.current_namespace,
            self.agent_config.db_type,
            self.input.database_name,
        )

    def to_dict(self) -> Dict:
        """
        Convert the node to a dictionary representation.

        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "input": (
                dict(self.input) if isinstance(self.input, BaseInput) else self.input
            ),  # try to use BaseInput for all input data
            "status": self.status,
            "result": dict(self.result) if isinstance(self.result, BaseResult) else self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any], agent_config: Optional[AgentConfig] = None) -> Node:
        """Create a Node instance from dictionary representation."""
        # Convert input data based on a node type
        input_data = node_dict["input"]
        if isinstance(input_data, dict):
            try:
                if node_dict["type"] == NodeType.TYPE_SCHEMA_LINKING:
                    input_data = SchemaLinkingInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SQL:
                    input_data = GenerateSQLInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_EXECUTE_SQL:
                    input_data = ExecuteSQLInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_OUTPUT:
                    input_data = OutputInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_FIX:
                    input_data = FixInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_METRICS:
                    input_data = GenerateMetricsInput(**input_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SEMANTIC_MODEL:
                    input_data = GenerateSemanticModelInput(**input_data)
            except Exception as e:
                logger.warning(f"Failed to convert input data for {node_dict['type']}: {e}")
                input_data = None

        # Create node instance
        node = cls(
            node_id=node_dict["id"],
            description=node_dict["description"],
            node_type=node_dict["type"],
            input_data=input_data,
            agent_config=agent_config,
        )

        # Convert result data based on node type
        result_data = node_dict["result"]
        if isinstance(result_data, dict):
            try:
                # TODO: use factory pattern to create the result data
                if node_dict["type"] == NodeType.TYPE_SCHEMA_LINKING:
                    result_data = SchemaLinkingResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SQL:
                    result_data = GenerateSQLResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_EXECUTE_SQL:
                    result_data = ExecuteSQLResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_OUTPUT:
                    result_data = OutputResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_REFLECT:
                    result_data = ReflectionResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_REASONING:
                    result_data = ReasoningResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_METRICS:
                    result_data = GenerateMetricsResult(**result_data)
                elif node_dict["type"] == NodeType.TYPE_GENERATE_SEMANTIC_MODEL:
                    result_data = GenerateSemanticModelResult(**result_data)
                elif "success" in result_data:
                    result_data = BaseResult(**result_data)
            except Exception as e:
                logger.warning(f"Failed to convert result data for {node_dict['type']}: {e}")
                result_data = None

        # Set additional attributes
        node.status = node_dict["status"]
        node.result = result_data
        node.start_time = node_dict["start_time"]
        node.end_time = node_dict["end_time"]
        node.dependencies = node_dict["dependencies"]
        node.metadata = node_dict["metadata"]

        return node
