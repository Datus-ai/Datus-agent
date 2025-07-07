from typing import Dict

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult
from datus.tools.llms_tools import LLMTool
from datus.utils.loggings import get_logger

logger = get_logger(__file__)


class GenerateMetricsNode(Node):
    def execute(self):
        self.result = self._generate_metrics()

    def validate_input(self):
        if not isinstance(self.input, GenerateMetricsInput):
            raise ValueError("Input must be a GenerateMetricsInput instance")
        return True

    def _generate_metrics(self) -> GenerateMetricsResult:
        """Generate metrics for the given SQL query."""
        if not self.model:
            return GenerateMetricsResult(
                success=False,
                error="Metrics generation model not provided",
                sql_queries=[],
                metrics=[],
            )
        try:
            tool = LLMTool(self.model)
            logger.debug(f"Generate metrics input: {type(self.input)} {self.input}")
            return tool.generate_metrics(self.input, self.agent_config.current_db_config())
        except Exception as e:
            logger.error(f"Metrics generation execution error: {str(e)}")
            return GenerateMetricsResult(
                success=False,
                error=str(e),
                sql_queries=[],
                metrics=[],
            )

    def update_context(self, workflow: Workflow) -> Dict:
        pass

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = GenerateMetricsInput(
            sql_task=workflow.task,
            sql_query=workflow.get_last_sqlcontext().sql_query,
        )
        self.input = next_input
        return {"success": True, "message": "Metrics generated", "suggestions": [next_input]}
