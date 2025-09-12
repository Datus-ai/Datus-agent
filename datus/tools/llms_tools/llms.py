from typing import List, Optional, Union

from agents import Tool

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.models.base import LLMBaseModel
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput, GenerateSemanticModelResult
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult
from datus.storage.schema_metadata.store import SchemaStorage
from datus.tools.base import BaseTool
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

from ...schemas.compare_node_models import CompareInput, CompareResult
from ...schemas.fix_node_models import FixInput, FixResult
from ...schemas.reason_sql_node_models import ReasoningInput, ReasoningResult
from ...schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from .autofix_sql import autofix_sql
from .compare_sql import compare_sql
from .generate_metrics import generate_metrics_with_mcp
from .generate_semantic_model import generate_semantic_model_with_mcp
from .generate_sql import generate_sql
from .match_schema import MatchSchemaTool
from .reasoning_sql import reasoning_sql_with_mcp

logger = get_logger(__name__)


class LLMTool(BaseTool):
    def __init__(
        self,
        model: Optional[LLMBaseModel] = None,
        model_name: str = "deepseek-v3",
        agent_config: Optional[AgentConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if model is not None:
            self.model = model
        else:
            self.model = LLMBaseModel.create_model(model_name=model_name, agent_config=agent_config)

    def generate_sql(self, input_data: GenerateSQLInput) -> GenerateSQLResult:
        return generate_sql(self.model, input_data)

    def autofix_sql(self, input_data: FixInput, docs: list[str]) -> FixResult:
        return autofix_sql(self.model, input_data, docs)

    def reasoning_sql(self, input_data: ReasoningInput, tools: List[Tool]) -> ReasoningResult:
        tool_config = {"max_turns": input_data.max_turns}
        return reasoning_sql_with_mcp(model=self.model, input_data=input_data, tools=tools, tool_config=tool_config)

    def test(self, input: str) -> GenerateSQLResult:
        return self.model.generate(input)

    def match_schema(
        self,
        input_data: SchemaLinkingInput,
        rag_storage: Union[SchemaStorage, None] = None,
        storage_path: Union[str, None] = None,
    ) -> SchemaLinkingResult:
        if not rag_storage and not storage_path:
            raise DatusException(
                ErrorCode.TOOL_INVALID_INPUT,
                message="Schema linking by LLM requires LanceDB storage",
            )
        if not rag_storage:
            rag_storage = SchemaStorage(storage_path)
        match_schema_tool = MatchSchemaTool(self.model, rag_storage)
        return match_schema_tool.execute(input_data)

    def generate_metrics(
        self,
        input_data: GenerateMetricsInput,
        db_config: DbConfig,
        tools: List[Tool],
        tool_config=None,
    ) -> GenerateMetricsResult:
        if tool_config is None:
            tool_config = {}
        return generate_metrics_with_mcp(
            self.model, input_data, tools=tools, db_config=db_config, tool_config=tool_config
        )

    def generate_semantic_model(
        self, table_definition: str, input_data: GenerateSemanticModelInput, db_config: DbConfig, tool_config=None
    ) -> GenerateSemanticModelResult:
        if tool_config is None:
            tool_config = {}
        return generate_semantic_model_with_mcp(
            self.model, table_definition, input_data, db_config=db_config, tool_config=tool_config
        )

    def compare_sql(self, input_data: CompareInput) -> CompareResult:
        return compare_sql(self.model, input_data)
