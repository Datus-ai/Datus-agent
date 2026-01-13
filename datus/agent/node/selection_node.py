# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any, AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.prompts.selection import create_selection_prompt
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.parallel_node_models import SelectionResult
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SelectionNode(Node):
    """Node that selects the best result from multiple parallel candidates"""

    def update_context(self, workflow: Workflow) -> Dict:
        """Update workflow context with selected result"""
        if self.result and self.result.success and self.result.selected_result:
            # Update context with the selected result
            workflow.context.update_selection_result(
                self.result.selected_result,
                {
                    "selected_source": self.result.selected_source,
                    "selection_reason": self.result.selection_reason,
                    "score_analysis": self.result.score_analysis,
                },
            )

            # Extract and add SQL context from the selected result
            selected_data = self.result.selected_result
            if isinstance(selected_data, dict) and "result" in selected_data:
                # Get the actual result object from the child node result
                child_result = selected_data["result"]

                # Check if it's a SQL generation result
                if hasattr(child_result, "sql_query") and hasattr(child_result, "explanation"):
                    from datus.schemas.node_models import SQLContext

                    # Create SQL context from the selected result
                    sql_context = SQLContext(
                        sql_query=child_result.sql_query, explanation=child_result.explanation or ""
                    )

                    # Add to workflow context
                    workflow.context.sql_contexts.append(sql_context)
                    logger.info(f"Added selected SQL to context: {child_result.sql_query[:100]}...")

                    return {"success": True, "message": "Selection node context updated with SQL"}

        return {"success": True, "message": "Selection node context updated"}

    def setup_input(self, workflow: Workflow) -> Dict:
        """Setup input for selection node"""
        logger.info(
            f"SelectionNode.setup_input called: current_index={workflow.current_node_index}, "
            f"node_order={workflow.node_order}"
        )

        # Get parallel results from workflow context
        parallel_results = workflow.context.parallel_results
        logger.info(f"SelectionNode.setup_input: parallel_results available = {parallel_results is not None}")

        if not parallel_results:
            # If no parallel results, look for any completed reasoning node
            reasoning_node = None
            # Check all nodes for a reasoning node that has completed successfully
            for i, node_id in enumerate(workflow.node_order):
                node = workflow.nodes.get(node_id)
                logger.info(
                    f"SelectionNode.setup_input: checking node at index {i}: node_id={node_id}, "
                    f"node_type={node.type if node else None}, status={node.status if node else None}"
                )
                if (
                    node
                    and node.type == "reasoning"
                    and node.status == "completed"
                    and node.result
                    and node.result.success
                ):
                    reasoning_node = node
                    logger.info(f"Found completed reasoning node: {node_id}")
                    break

            if reasoning_node:
                logger.info(f"No parallel results found, using reasoning node {reasoning_node.id} result as candidate")
                # Create a single candidate result from the reasoning node
                reasoning_result = {
                    "reasoning_node": {
                        "success": True,
                        "result": reasoning_node.result,
                        "node_id": reasoning_node.id,
                        "node_type": reasoning_node.type,
                    }
                }
                parallel_results = reasoning_result
                logger.info(
                    f"SelectionNode.setup_input: created reasoning_result candidate with {len(parallel_results)} "
                    f"entries"
                )
            else:
                logger.error("SelectionNode.setup_input: No completed reasoning node found with successful result")
                return {"success": False, "message": "No parallel results available in workflow context"}

        # Create or update the SelectionInput with parallel results
        from datus.schemas.parallel_node_models import SelectionInput

        # If input already exists and is SelectionInput, update it; otherwise create new
        if isinstance(self.input, SelectionInput):
            self.input.candidate_results = parallel_results
        else:
            self.input = SelectionInput(candidate_results=parallel_results, selection_criteria="best_quality")

        logger.info(f"Setup selection input with {len(parallel_results)} candidates")
        return {"success": True, "message": "Selection node input setup complete"}

    def execute(self) -> SelectionResult:
        """Select the best result from candidates"""
        if not self.input or not self.input.candidate_results:
            result = SelectionResult(
                success=False,
                error="No candidate results provided for selection",
                selected_result=None,
                selected_source="",
                selection_reason="No candidates available",
                all_candidates={},
            )
            self.result = result
            return result

        candidates = self.input.candidate_results

        logger.info(f"Starting selection from {len(candidates)} candidates")

        try:
            # If only one candidate, select it directly
            if len(candidates) == 1:
                candidate_id = list(candidates.keys())[0]
                candidate_result = candidates[candidate_id]

                result = SelectionResult(
                    success=True,
                    selected_result=candidate_result,
                    selected_source=candidate_id,
                    selection_reason="Only one candidate available",
                    all_candidates=candidates,
                    score_analysis={candidate_id: {"score": 10, "reason": "Single candidate"}},
                )
                self.result = result
                return result

            # Use LLM-based selection for multiple candidates
            if self.model:
                result = self._llm_based_selection(candidates)
            else:
                # Fallback to rule-based selection
                result = self._rule_based_selection(candidates)

            self.result = result
            return result

        except Exception as e:
            logger.error(f"Selection failed: {str(e)}")
            result = SelectionResult(
                success=False,
                error=f"Selection failed: {str(e)}",
                selected_result=None,
                selected_source="",
                selection_reason="Selection process failed",
                all_candidates=candidates,
            )
            self.result = result
            return result

    def _llm_based_selection(self, candidates: Dict[str, Any]) -> SelectionResult:
        """Use LLM to select the best candidate"""
        prompt_version = self.input.prompt_version if self.input else None
        prompt = create_selection_prompt(candidates, prompt_version=prompt_version)

        try:
            logger.info("Calling LLM for candidate selection...")
            response = self.model.generate_with_json_output(prompt)

            best_candidate_id = response.get("best_candidate", "")
            selection_reason = response.get("reason", "LLM-based selection")
            score_analysis = response.get("score_analysis", {})

            if best_candidate_id not in candidates:
                # Fallback if LLM returns invalid candidate
                logger.warning(f"LLM returned invalid candidate: {best_candidate_id}")
                return self._rule_based_selection(candidates)

            selected_result = candidates[best_candidate_id]

            return SelectionResult(
                success=True,
                selected_result=selected_result,
                selected_source=best_candidate_id,
                selection_reason=selection_reason,
                all_candidates=candidates,
                score_analysis=score_analysis,
            )

        except Exception as e:
            logger.error(f"LLM-based selection failed: {str(e)}")
            # Fallback to rule-based selection
            return self._rule_based_selection(candidates)

    def _rule_based_selection(self, candidates: Dict[str, Any]) -> SelectionResult:
        """Fallback rule-based selection"""
        logger.info("Using rule-based selection as fallback")

        # Score candidates based on simple rules
        scores = {}

        for candidate_id, candidate in candidates.items():
            score = 0
            reasons = []

            # Check if candidate has a successful result
            if isinstance(candidate, dict):
                if candidate.get("success", False):
                    score += 5
                    reasons.append("Successful execution")

                # Check if result exists and has content
                result = candidate.get("result")
                if result:
                    score += 3
                    reasons.append("Has result content")

                    # For SQL results, prefer results with data
                    if hasattr(result, "sql_result_final") and result.sql_result_final:
                        score += 2
                        reasons.append("Has SQL result")

                # Check error status
                if not candidate.get("error"):
                    score += 2
                    reasons.append("No errors")

            scores[candidate_id] = {"score": score, "reason": "; ".join(reasons) if reasons else "Basic scoring"}

        # Select candidate with highest score
        best_candidate_id = max(scores.keys(), key=lambda x: scores[x]["score"])
        best_score = scores[best_candidate_id]["score"]

        return SelectionResult(
            success=True,
            selected_result=candidates[best_candidate_id],
            selected_source=best_candidate_id,
            selection_reason=f"Rule-based selection (score: {best_score})",
            all_candidates=candidates,
            score_analysis=scores,
        )

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """
        Execute the selection node with streaming support.

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
                action_type="selection",
                status="running",
                message="Selecting the best result from candidates",
            )
            yield action_history

        # Execute the main logic (non-streaming)
        result = self.execute()

        # Generate final action history
        if action_history_manager:
            status = "completed" if result.success else "failed"
            message = (
                f"Selection completed. Selected source: {result.selected_source}" if result.success else result.error
            )
            action_history = action_history_manager.update(
                action_id=action_id,
                status=status,
                message=message,
                result={
                    "success": result.success,
                    "selected_source": result.selected_source if result.success else None,
                    "reason": result.selection_reason if result.success else None,
                },
            )
            yield action_history
