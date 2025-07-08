from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.doc_search_node_models import DocSearchInput, DocSearchResult
from datus.tools.search_tools import SearchTool
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DocSearchNode(Node):
    def execute(self):
        self.result = self._execute_document()

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = DocSearchInput(keywords=workflow.context.doc_search_keywords, top_n=3, method="external")
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
        tool = SearchTool()
        if not tool:
            return DocSearchResult(success=False, error="Document search tool not found", docs={})

        return tool.execute(self.input)
