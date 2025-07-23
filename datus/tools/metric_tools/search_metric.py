from typing import Optional

from datus.configuration.agent_config import AgentConfig
from datus.models.base import LLMBaseModel
from datus.schemas.node_models import Metrics
from datus.schemas.search_metrics_node_models import SearchMetricsInput, SearchMetricsResult
from datus.storage.metric.store import SemanticMetricsRAG, rag_by_configuration
from datus.tools import BaseTool


class SearchMetricsTool(BaseTool):
    """Tool for managing and querying metric information."""

    def __init__(
        self,
        db_path: str = "data/datus_db",
        store: Optional[SemanticMetricsRAG] = None,
        agent_config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if store:
            self.store = store
        elif agent_config:
            self.store = rag_by_configuration(agent_config)
        else:
            self.store = SemanticMetricsRAG(db_path)

    def execute(self, input_param: SearchMetricsInput, model: Optional[LLMBaseModel] = None) -> SearchMetricsResult:
        """Execute search metrics operations."""
        return self._search_hybrid_metrics(input_param)

    def _search_hybrid_metrics(self, input_param: SearchMetricsInput) -> SearchMetricsResult:
        metric_results = self.store.search_hybrid_metrics(input_param, input_param.top_n_by_rate())

        # Convert dictionaries to proper model instances
        metric_list = [Metrics.from_dict(metric) for metric in metric_results]

        return SearchMetricsResult(
            success=True,
            error=None,
            semantic_model_meta=input_param.semantic_model_meta,
            metrics=metric_list,
            metrics_count=len(metric_list),
        )
