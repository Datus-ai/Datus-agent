# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
from typing import AsyncGenerator, Dict, Optional

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistory, ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.tools.search_tools import SearchTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DocSearchNode(Node):
    def __init__(
        self,
        node_id: str,
        description: str,
        node_type: str,
        input_data: DocSearchInput = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            node_id=node_id,
            description=description,
            node_type=node_type,
            input_data=input_data,
            agent_config=agent_config,
        )

    def execute(self):
        self.result = self._execute_document()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute document search with streaming support."""
        async for action in self._doc_search_stream(action_history_manager):
            yield action

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = DocSearchInput(keywords=workflow.context.doc_search_keywords, top_n=3, method="internal")
        self.input = next_input
        return {"success": True, "message": "Document appears valid", "suggestions": [next_input]}

    def update_context(self, workflow: Workflow) -> Dict:
        """Update document search results to workflow context."""
        result = self.result
        try:
            logger.info(f"Updating document search context: {result}")
            workflow.context.document_result = result
            return {"success": True, "message": "Updated document search context"}
        except Exception as e:
            logger.error(f"Failed to update document search context: {str(e)}")
            return {"success": False, "message": f"Document search context update failed: {str(e)}"}

    def _execute_document(self) -> DocSearchResult:
        """Execute document search based on method"""
        return SearchTool(self.agent_config).execute(self.input)

    async def _doc_search_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute document search with streaming support and action history tracking."""
        try:
            # Document search action
            search_action = ActionHistory(
                action_id="document_search",
                role=ActionRole.WORKFLOW,
                messages="Searching for relevant documentation",
                action_type="document_search",
                input={
                    "keywords": getattr(self.input, "keywords", []),
                    "top_n": getattr(self.input, "top_n", 3),
                },
                status=ActionStatus.PROCESSING,
            )
            yield search_action

            # Execute document search
            result = self._execute_document()

            search_action.status = ActionStatus.SUCCESS if result.success else ActionStatus.FAILED
            search_action.output = {
                "success": result.success,
                "documents_found": len(result.docs) if result.docs else 0,
            }

            # Store result for later use
            self.result = result

            # Yield the updated action with final status
            yield search_action

        except Exception as e:
            logger.error(f"Document search streaming error: {str(e)}")
            raise
