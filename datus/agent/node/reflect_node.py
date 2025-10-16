# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.reflect import evaluate_with_model
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.node_models import STRATEGY_LIST, ReflectionInput, ReflectionResult, SQLContext, StrategyType
from datus.utils.env import get_env_int
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ReflectNode(Node):
    def update_context(self, workflow: Workflow) -> Dict:
        """Update reflection results to workflow context."""
        result = self.result
        try:
            workflow.reflection_round += 1
            if "keywords" in result.details:
                workflow.context.doc_search_keywords = result.details["keywords"]

            last_record = workflow.context.sql_contexts[-1]
            if last_record.sql_query == self.input.sql_context[-1].sql_query:
                strategy = result.strategy
                if strategy not in STRATEGY_LIST:
                    error_msg = f"Unknown reflection strategy: {strategy}"
                    logger.error(error_msg)
                    return {"success": False, "message": error_msg}

                last_record.reflection_strategy = strategy
                last_record.reflection_explanation = self.result.details.get("explanation", "")

                # change the workflow as needed
                strategy_result = self._execute_reflection_strategy(strategy, result.details, workflow)
                return strategy_result
            else:
                error_msg = "SQL query mismatch in reflection"
                logger.warning(f"{error_msg}: {last_record.sql_query}")
                return {"success": False, "message": error_msg}
        except Exception as e:
            logger.error(f"Failed to update reflection context: {str(e)}")
            return {"success": False, "message": f"Reflection context update failed: {str(e)}"}

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = ReflectionInput(
            task_description=workflow.task,
            sql_context=workflow.context.sql_contexts,
        )
        self.input = next_input
        return {"success": True, "message": "Node input appears valid", "suggestions": [next_input]}

    def execute(self):
        self.result = self._execute_reflect()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute reflection with streaming support."""
        async for action in self._reflect_stream(action_history_manager):
            yield action

    def _execute_reflect(self) -> ReflectionResult:
        if not self.model:
            # TODO: Add a manual evaluation function
            raise ValueError("Model is required for reflection")

        task = self.input.task_description

        if len(self.input.sql_context) == 0:
            return ReflectionResult(success=False, error="No SQL context provided", strategy="UNKNOWN", details={})

        # TODO: use all the sql_contexts to evaluate
        eval_result = evaluate_with_model(task, self.input, self.model)

        # Explicitly create ReflectionResult with required fields
        return ReflectionResult(
            success=eval_result.get("success", False),
            error=eval_result.get("error", ""),
            strategy=eval_result.get("strategy", "UNKNOWN"),
            details=eval_result.get("details", {}),
        )

    def _execute_reflection_strategy(
        self,
        strategy: str,
        details: Dict,
        workflow: Workflow,
    ) -> Dict:
        """
        Execute the recommended optimization strategy.

        Args:
            node: The reflection node
            workflow: The workflow need to be modified
            strategy: The strategy to execute
            details: Details about the error and original SQL

        Returns:
            Result of the strategy execution
        """
        strategy = strategy.upper()
        if strategy == StrategyType.SUCCESS:
            return {"success": True, "message": "go on to output"}

        max_round = get_env_int("MAX_REFLECTION_ROUNDS", 3)
        if workflow.reflection_round == max_round:
            logger.info("Max reflection rounds reached, execute reasoning")
            return self._execute_strategy(details, workflow, StrategyType.REASONING)
        elif workflow.reflection_round > max_round:
            logger.info("Max reflection rounds exceeded, exit")
            return {"success": True, "message": "Max reflection rounds exceeded"}
        if strategy in [
            StrategyType.DOC_SEARCH,
            StrategyType.SCHEMA_LINKING,
            StrategyType.SIMPLE_REGENERATE,
            StrategyType.REASONING,
        ]:
            return self._execute_strategy(details, workflow, strategy)
        else:
            return {"success": False, "message": f"Unknown strategy: {strategy}"}

    def _execute_strategy(self, details: Dict, workflow: Workflow, strategy: str) -> Dict:
        """
            Execute the reflection strategy to add relative nodes to workflow
            Args:
                node: The reflection node
                details: Details about the error and original SQL
            Returns:
        Result of the strategy execution
        """

        try:
            if strategy == StrategyType.SIMPLE_REGENERATE:
                if "sql" in self.result.details:
                    new_record = SQLContext(
                        sql_query=self.result.details.get("sql", ""),
                        explanation=self.result.details.get("explanation", ""),
                    )
                    workflow.context.sql_contexts.append(new_record)
                else:
                    logger.warning("{strategy} strategy requires 'sql_query' in node.result.details")

            current_position = workflow.current_node_index
            strategy = strategy.lower()
            nodes_added = []
            reflection_nodes = workflow._global_config.reflection_nodes(strategy).copy()
            reflection_nodes.reverse()
            for node_type in reflection_nodes:
                new_node = Node.new_instance(
                    node_id=f"reflect_{workflow.reflection_round}_{node_type}",
                    description=NodeType.NODE_TYPE_DESCRIPTIONS.get(node_type, ""),
                    node_type=node_type,
                    input_data=None,
                    agent_config=workflow._global_config,
                    tools=workflow.tools,
                )
                workflow.add_node(new_node, current_position + 1)
                nodes_added.insert(0, new_node.id)

            return {
                "success": True,
                "message": f"Added {strategy} workflow sequence",
                "workflow_modified": True,
                "nodes_added": nodes_added,
            }
        except Exception as e:
            logger.error(f"Error during {strategy} workflow modification: {e}")
            return {"success": False, "message": f"{strategy} workflow modification failed: {str(e)}"}

    async def _reflect_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute reflection with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for reflection")
            return

        try:
            # Reflection analysis action
            reflection_action = ActionHistory(
                action_id="reflection_analysis",
                role=ActionRole.WORKFLOW,
                messages="Analyzing SQL execution results and determining next steps",
                action_type="reflection_analysis",
                input={
                    "sql_contexts_count": len(self.input.sql_context) if hasattr(self.input, "sql_context") else 0,
                    "task_description": getattr(self.input.task_description, "task", "")
                    if hasattr(self.input, "task_description")
                    else "",
                },
                status=ActionStatus.PROCESSING,
            )
            yield reflection_action

            # Execute reflection analysis
            result = self._execute_reflect()

            reflection_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
            reflection_action.output = {
                "success": result.success,
                "strategy": result.strategy,
                "has_details": bool(result.details),
                "error": result.error if hasattr(result, "error") and result.error else None,
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield reflection_action

        except Exception as e:
            logger.error(f"Reflection streaming error: {str(e)}")
            raise
