# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncGenerator, Dict, List, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.parallel_node_models import ParallelInput, ParallelResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ParallelNode(Node):
    """Node that executes multiple child nodes in parallel"""

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context after parallel execution"""
        if self.result and self.result.success and self.result.child_results:
            # Store parallel results in workflow context
            workflow.context.update_parallel_results(self.result.child_results)
            logger.info(f"Updated workflow context with {len(self.result.child_results)} parallel results")

        return {"success": True, "message": "Parallel node context updated"}

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for parallel node execution"""
        if not isinstance(self.input, ParallelInput):
            return {"success": False, "message": "Invalid input type for ParallelNode"}

        # Validate that child node types are supported
        for child_node_type in self.input.child_nodes:
            # allow action types directly
            if child_node_type in NodeType.ACTION_TYPES:
                continue
            # allow subworkflow names defined in config
            if (
                isinstance(child_node_type, str)
                and self.agent_config
                and child_node_type in getattr(self.agent_config, "custom_workflows", {})
            ):
                continue
            # allow explicit subworkflow marker
            if child_node_type == NodeType.TYPE_SUBWORKFLOW:
                continue
            return {
                "success": False,
                "message": f"Unsupported child entry: {child_node_type}. "
                f"Must be an action node type or a subworkflow name.",
            }

        return {"success": True, "message": "Parallel node input setup complete"}

    def execute(self) -> ParallelResult:
        """Execute child nodes in parallel"""
        if not self.input or not self.input.child_nodes:
            return ParallelResult(
                success=False,
                error="No child nodes specified for parallel execution",
                child_results={},
                execution_order=[],
            )

        logger.info(f"Starting parallel execution of {len(self.input.child_nodes)} child nodes")

        child_results = {}
        execution_order = []

        try:
            # Get workflow from node context (should be set during execution)
            workflow = getattr(self, "workflow", None)
            if not workflow:
                raise ValueError("Workflow context not available for parallel node execution")

            # Create child nodes
            child_nodes = self._create_child_nodes(workflow)

            # Execute child nodes in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(child_nodes)) as executor:
                # Submit all child node executions
                future_to_node = {executor.submit(self._execute_child_node, node): node.id for node in child_nodes}

                # Collect results as they complete
                for future in as_completed(future_to_node):
                    node_id = future_to_node[future]
                    try:
                        result = future.result()
                        child_results[node_id] = result
                        execution_order.append(node_id)
                        logger.info(f"Child node {node_id} completed")
                    except Exception as e:
                        logger.error(f"Child node {node_id} failed: {str(e)}")
                        child_results[node_id] = {"success": False, "error": str(e)}
                        execution_order.append(node_id)

            # Evaluate children success
            all_success = all(
                result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False)
                for result in child_results.values()
            )
            any_success = any(
                result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False)
                for result in child_results.values()
            )

            logger.info(
                f"Child results evaluation: {[(k, v.get('success', 'unknown')) for k, v in child_results.items()]}"
            )
            logger.info(f"All success determined as: {all_success}")

            # Do not block the workflow if some children failed; continue to selection as long as any child succeeded.
            result = ParallelResult(success=any_success, child_results=child_results, execution_order=execution_order)

            if not any_success:
                result.error = "All child nodes failed during parallel execution"
                logger.error(f"Parallel execution marked as failed: {result.error}")
            else:
                if not all_success:
                    logger.warning("Parallel execution partial success: some child nodes failed")
                logger.info(f"Parallel execution completed with {len(child_results)} child results")

            self.result = result
            logger.info(f"Final parallel result: success={result.success}, error={result.error}")
            return result

        except Exception as e:
            logger.error(f"Parallel execution failed: {str(e)}")
            result = ParallelResult(
                success=False,
                error=f"Parallel execution failed: {str(e)}",
                child_results=child_results,
                execution_order=execution_order,
            )
            self.result = result
            return result

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the node with streaming support.

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
                action_type="parallel_execution",
                status="running",
                message=f"Starting parallel execution of {len(self.input.child_nodes) if self.input else 0} child "
                f"nodes",
            )
            yield action_history

        # Execute the main logic (non-streaming)
        result = self.execute()

        # Generate final action history
        if action_history_manager:
            status = "completed" if result.success else "failed"
            message = (
                f"Parallel execution completed with {len(result.child_results)} results"
                if result.success
                else result.error
            )
            action_history = action_history_manager.update(
                action_id=action_id,
                status=status,
                message=message,
                result={"success": result.success, "child_count": len(result.child_results)},
            )
            yield action_history

    def _create_child_nodes(self, workflow: Workflow) -> List[Node]:
        """Create child node instances for parallel execution"""
        child_nodes = []

        for i, child_node_type in enumerate(self.input.child_nodes):
            # Support `child` as the subworkflow name (a string that exists in `custom_workflows`),
            # in which case the type is `TYPE_SUBWORKFLOW`.
            is_subworkflow_name = (
                isinstance(child_node_type, str)
                and self.agent_config
                and child_node_type in getattr(self.agent_config, "custom_workflows", {})
            )

            node_type_to_use = NodeType.TYPE_SUBWORKFLOW if is_subworkflow_name else child_node_type
            child_node_id = f"{self.id}_child_{i}_{child_node_type}"
            child_description = (
                f"Parallel child subworkflow: {child_node_type}"
                if is_subworkflow_name
                else f"Parallel child: {NodeType.get_description(child_node_type)}"
            )

            input_data = None
            if is_subworkflow_name:
                try:
                    from datus.schemas.subworkflow_node_models import SubworkflowInput

                    input_data = SubworkflowInput(workflow_name=child_node_type, pass_context=True)
                except Exception as e:
                    logger.error(f"Create SubworkflowInput failed for {child_node_type}: {e}")

            child_node = Node.new_instance(
                node_id=child_node_id,
                description=child_description,
                node_type=node_type_to_use,
                input_data=input_data,
                agent_config=self.agent_config,
                tools=workflow.tools,
            )

            child_node.workflow = workflow
            if not is_subworkflow_name:
                setup_result = child_node.setup_input(workflow)
                if not setup_result.get("success", False):
                    logger.warning(
                        f"Failed to setup input for child node {child_node_id}: "
                        f"{setup_result.get('message', 'Unknown error')}"
                    )

            child_nodes.append(child_node)

        return child_nodes

    def _execute_child_node(self, child_node: Node) -> Dict:
        """Execute a single child node"""
        try:
            logger.info(f"Executing child node: {child_node.id}")

            child_node._initialize()

            child_node.start()
            child_node.execute()  # This sets child_node.result

            result = child_node.result

            success = False
            if result is not None:
                # Check if result has success attribute and it's True
                if hasattr(result, "success") and result.success:
                    success = True
                    child_node.complete(result)
                else:
                    # Result exists but indicates failure
                    error_msg = getattr(result, "error", "Execution failed")
                    child_node.fail(f"Child node execution failed: {error_msg}")
            else:
                # No result returned
                child_node.fail("Child node execution returned no result")

            return {
                "success": success,
                "status": child_node.status,
                "result": result,
                "node_id": child_node.id,
                "node_type": child_node.type,
                "start_time": child_node.start_time,
                "end_time": child_node.end_time,
            }

        except Exception as e:
            logger.error(f"Child node {child_node.id} execution failed: {str(e)}")
            child_node.fail(str(e))
            return {
                "success": False,
                "status": "failed",
                "error": str(e),
                "node_id": child_node.id,
                "node_type": child_node.type,
                "start_time": child_node.start_time,
                "end_time": child_node.end_time,
            }
