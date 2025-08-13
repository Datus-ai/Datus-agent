from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.plan import generate_workflow
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.base import BaseResult
from datus.schemas.subworkflow_node_models import SubworkflowInput, SubworkflowResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SubworkflowNode(Node):
    """Node that executes a predefined workflow as a step"""

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context with subworkflow results"""
        if self.result and self.result.success and self.result.node_results:
            # Update context with all node results from the subworkflow
            workflow.context.update({f"subworkflow_{self.result.workflow_name}": self.result.node_results})
            logger.info(f"Updated workflow context with results from subworkflow '{self.result.workflow_name}'")

        return {"success": True, "message": "Subworkflow node context updated"}

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for subworkflow node execution"""
        if not isinstance(self.input, SubworkflowInput):
            return {"success": False, "message": "Invalid input type for SubworkflowNode"}

        # Validate that workflow name is provided
        if not self.input.workflow_name:
            return {"success": False, "message": "Workflow name not specified for subworkflow execution"}

        return {"success": True, "message": "Subworkflow node input setup complete"}

    def execute(self) -> SubworkflowResult:
        """Execute the subworkflow"""
        if not self.input or not self.input.workflow_name:
            return SubworkflowResult(
                success=False,
                error="No workflow specified for subworkflow execution",
                workflow_name="unknown",
                node_results={},
                execution_order=[],
            )

        logger.info(f"Starting subworkflow execution of '{self.input.workflow_name}'")

        try:
            # Get parent workflow from node context
            parent_workflow = getattr(self, "workflow", None)
            if not parent_workflow:
                raise ValueError("Parent workflow context not available for subworkflow node execution")

            # Get task from parent workflow
            task = parent_workflow.task

            # Find the workflow in the agent config
            agent_config = self.agent_config
            if not agent_config:
                raise ValueError("Agent configuration not available for subworkflow execution")

            # Check if workflow exists in custom workflows
            if self.input.workflow_name not in agent_config.custom_workflows:
                raise ValueError(f"Workflow '{self.input.workflow_name}' not found in agent configuration")

            # Create a new workflow instance for the subworkflow
            subworkflow = generate_workflow(
                task=task,
                plan_type=self.input.workflow_name,
                agent_config=agent_config,
            )

            # Pass context from parent workflow if requested
            if self.input.pass_context:
                subworkflow.context = parent_workflow.context.copy()

            # Override node parameters if provided
            if self.input.node_params:
                self._apply_node_params(subworkflow)

            # Execute the subworkflow
            node_results = {}
            execution_order = []

            # Skip the first node (Start Node)
            start_node = subworkflow.get_current_node()
            if start_node:
                start_node.complete(BaseResult(success=True))
                next_node = subworkflow.advance_to_next_node()
                if next_node:
                    # lazy import to avoid circular import at module level
                    from datus.agent.evaluate import setup_node_input  # type: ignore

                    setup_node_input(next_node, subworkflow)

            # Execute all nodes in sequence with loop protection
            max_iterations = 50  # Prevent infinite loops
            iteration_count = 0

            while not subworkflow.is_complete() and iteration_count < max_iterations:
                iteration_count += 1
                current_node = subworkflow.get_current_node()
                if not current_node:
                    logger.warning(f"No current node found in subworkflow at iteration {iteration_count}")
                    break

                node_id = current_node.id
                logger.info(
                    f"Executing subworkflow node ({iteration_count}/{max_iterations}): {current_node.description}"
                )

                try:
                    current_node._initialize()
                    current_node.start()
                    current_node.execute()

                    # finalize node status based on its result
                    exec_result = current_node.result
                    if exec_result is not None and getattr(exec_result, "success", False):
                        current_node.complete(exec_result)
                    else:
                        error_msg = getattr(exec_result, "error", "Execution failed") if exec_result else "No result"
                        current_node.fail(error_msg)

                    # Record the result
                    node_results[node_id] = {
                        "success": current_node.status == "completed",
                        "result": current_node.result,
                        "node_type": current_node.type,
                        "description": current_node.description,
                    }
                    execution_order.append(node_id)

                    # Update the subworkflow context
                    if current_node.status == "completed":
                        current_node.update_context(subworkflow)
                    else:
                        logger.warning(f"Subworkflow node {node_id} failed")
                        result = SubworkflowResult(
                            success=False,
                            error=f"Subworkflow node {node_id} failed",
                            workflow_name=self.input.workflow_name,
                            node_results=node_results,
                            execution_order=execution_order,
                        )
                        self.result = result
                        return result

                except Exception as e:
                    logger.error(f"Subworkflow node {node_id} execution failed: {str(e)}")
                    node_results[node_id] = {
                        "success": False,
                        "error": str(e),
                        "node_type": current_node.type,
                        "description": current_node.description,
                    }
                    execution_order.append(node_id)

                    result = SubworkflowResult(
                        success=False,
                        error=f"Subworkflow node {node_id} execution failed: {str(e)}",
                        workflow_name=self.input.workflow_name,
                        node_results=node_results,
                        execution_order=execution_order,
                    )
                    self.result = result
                    return result

                # Advance to the next node and setup its input
                next_node = subworkflow.advance_to_next_node()
                if next_node:
                    from datus.agent.evaluate import setup_node_input  # type: ignore

                    try:
                        setup_result = setup_node_input(next_node, subworkflow)
                        if not setup_result.get("success", False):
                            logger.error(
                                f"Failed to setup input for next node {next_node.id}: "
                                f"{setup_result.get('message', 'Unknown error')}"
                            )
                            result = SubworkflowResult(
                                success=False,
                                error=f"Failed to setup input for node {next_node.id}: "
                                f"{setup_result.get('message', 'Unknown error')}",
                                workflow_name=self.input.workflow_name,
                                node_results=node_results,
                                execution_order=execution_order,
                            )
                            self.result = result
                            return result
                    except Exception as e:
                        logger.error(f"Exception during setup_node_input for {next_node.id}: {str(e)}")
                        result = SubworkflowResult(
                            success=False,
                            error=f"Exception during setup for node {next_node.id}: {str(e)}",
                            workflow_name=self.input.workflow_name,
                            node_results=node_results,
                            execution_order=execution_order,
                        )
                        self.result = result
                        return result

            # Check if we hit the iteration limit
            if iteration_count >= max_iterations:
                logger.error(
                    f"Subworkflow execution exceeded maximum iterations ({max_iterations}), possible infinite loop"
                )
                result = SubworkflowResult(
                    success=False,
                    error=f"Subworkflow execution exceeded maximum iterations ({max_iterations}), "
                    f"possible infinite loop",
                    workflow_name=self.input.workflow_name,
                    node_results=node_results,
                    execution_order=execution_order,
                )
                self.result = result
                return result

            # If we got here, all nodes executed successfully
            result = SubworkflowResult(
                success=True,
                workflow_name=self.input.workflow_name,
                node_results=node_results,
                execution_order=execution_order,
            )
            self.result = result
            return result

        except Exception as e:
            logger.error(f"Subworkflow execution failed: {str(e)}")
            result = SubworkflowResult(
                success=False,
                error=f"Subworkflow execution failed: {str(e)}",
                workflow_name=self.input.workflow_name if hasattr(self, "input") and self.input else "unknown",
                node_results={},
                execution_order=[],
            )
            self.result = result
            return result

    def _apply_node_params(self, workflow: Workflow) -> None:
        """Apply node parameter overrides to the subworkflow nodes"""
        if not self.input.node_params:
            return

        for node_type, params in self.input.node_params.items():
            # Find all nodes of this type
            for node in workflow.nodes.values():
                if node.type == node_type:
                    # Apply parameters
                    for param_name, param_value in params.items():
                        if hasattr(node.input, param_name):
                            setattr(node.input, param_name, param_value)
                            logger.info(f"Applied parameter override '{param_name}' to node {node.id}")
                        else:
                            logger.warning(f"Parameter '{param_name}' not found in node {node.id}")

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the subworkflow node with streaming support.

        Args:
            action_history_manager: Manager for tracking action history

        Yields:
            ActionHistory: Progress updates during node execution
        """
        # Create initial action history
        action_id = f"{self.id}_stream"
        if action_history_manager:
            action_history = action_history_manager.create(
                action_id=action_id,
                action_type="subworkflow_execution",
                status="running",
                message=f"Starting subworkflow execution of '{self.input.workflow_name if self.input else 'unknown'}'",
            )
            yield action_history

        # Execute the main logic (non-streaming)
        result = self.execute()

        # Generate final action history
        if action_history_manager:
            status = "completed" if result.success else "failed"
            message = (
                f"Subworkflow execution completed with {len(result.node_results)} nodes"
                if result.success
                else result.error
            )
            action_history = action_history_manager.update(
                action_id=action_id,
                status=status,
                message=message,
                result={
                    "success": result.success,
                    "workflow_name": result.workflow_name,
                    "node_count": len(result.node_results),
                },
            )
            yield action_history
