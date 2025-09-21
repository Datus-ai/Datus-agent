from typing import Optional, get_type_hints

from pydantic import BaseModel, create_model

from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.schemas.compare_node_models import CompareInput
from datus.schemas.date_parser_node_models import DateParserInput
from datus.schemas.doc_search_node_models import DocSearchInput
from datus.schemas.fix_node_models import FixInput
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput
from datus.schemas.node_models import ExecuteSQLInput, GenerateSQLInput, OutputInput, ReflectionInput
from datus.schemas.parallel_node_models import ParallelInput, SelectionInput
from datus.schemas.reason_sql_node_models import ReasoningInput
from datus.schemas.schema_linking_node_models import SchemaLinkingInput
from datus.schemas.search_metrics_node_models import SearchMetricsInput
from datus.schemas.subworkflow_node_models import SubworkflowInput


class NodeType:
    # Workflow control node types
    TYPE_BEGIN = "start"
    # TYPE_EVALUATE = "evaluate"
    TYPE_HITL = "hitl"
    TYPE_REFLECT = "reflect"
    TYPE_PARALLEL = "parallel"
    TYPE_SELECTION = "selection"
    TYPE_SUBWORKFLOW = "subworkflow"

    # Control node types list
    CONTROL_TYPES = [TYPE_BEGIN, TYPE_HITL, TYPE_REFLECT, TYPE_PARALLEL, TYPE_SELECTION, TYPE_SUBWORKFLOW]

    # SQL workflow action types
    TYPE_SCHEMA_LINKING = "schema_linking"  # For database schema analysis
    TYPE_GENERATE_SQL = "generate_sql"  # For SQL query generation
    TYPE_EXECUTE_SQL = "execute_sql"  # For SQL query execution
    TYPE_OUTPUT = "output"  # For result presentation
    TYPE_REASONING = "reasoning"  # For result presentation
    TYPE_DOC_SEARCH = "doc_search"  # For document search
    TYPE_FIX = "fix"  # For fixing the SQL query
    TYPE_GENERATE_METRICS = "generate_metrics"  # For generating metrics
    TYPE_GENERATE_SEMANTIC_MODEL = "generate_semantic_model"  # For generating semantic models
    TYPE_SEARCH_METRICS = "search_metrics"  # For search metrics
    TYPE_COMPARE = "compare"  # For comparing SQL with expectations
    TYPE_DATE_PARSER = "date_parser"  # For parsing temporal expressions

    # Agentic node types
    TYPE_CHAT = "chat"  # For conversational AI interactions
    TYPE_CHATBOT = "chatbot"  # For SQL generation with conversational AI

    ACTION_TYPES = [
        TYPE_SCHEMA_LINKING,
        TYPE_GENERATE_SQL,
        TYPE_EXECUTE_SQL,
        TYPE_OUTPUT,
        TYPE_REASONING,
        TYPE_DOC_SEARCH,
        TYPE_FIX,
        TYPE_GENERATE_METRICS,
        TYPE_GENERATE_SEMANTIC_MODEL,
        TYPE_SEARCH_METRICS,
        TYPE_COMPARE,
        TYPE_DATE_PARSER,
    ]

    # Agentic node types list
    AGENTIC_TYPES = [TYPE_CHAT, TYPE_CHATBOT]

    NODE_TYPE_DESCRIPTIONS = {
        TYPE_BEGIN: "Beginning of the workflow",
        TYPE_SCHEMA_LINKING: "Understand the query and find related schemas",
        TYPE_GENERATE_SQL: "Generate SQL query",
        TYPE_EXECUTE_SQL: "Execute SQL query",
        TYPE_REFLECT: "evaluation and self-reflection",
        TYPE_OUTPUT: "Return the results to the user",
        TYPE_REASONING: "Reasoning analysis",
        TYPE_DOC_SEARCH: "Search related documents",
        TYPE_HITL: "Human in the loop",
        TYPE_FIX: "Fix the SQL query",
        TYPE_GENERATE_METRICS: "Generate metrics",
        TYPE_GENERATE_SEMANTIC_MODEL: "Generate semantic model",
        TYPE_SEARCH_METRICS: "Search metrics",
        TYPE_PARALLEL: "Execute child nodes in parallel",
        TYPE_SELECTION: "Select best result from multiple candidates",
        TYPE_SUBWORKFLOW: "Execute a nested workflow",
        TYPE_COMPARE: "Compare SQL with expectations",
        TYPE_DATE_PARSER: "Parse temporal expressions in queries",
        TYPE_CHAT: "Conversational AI interactions with tool calling",
        TYPE_CHATBOT: "SQL generation with conversational AI and tool calling",
    }

    @classmethod
    def get_description(cls, node_type: str) -> str:
        return cls.NODE_TYPE_DESCRIPTIONS.get(node_type, f"Unknown node type: {node_type} for workflow")

    @classmethod
    def type_input(cls, node_type: str, input_data: dict, ignore_require_check: bool = False):
        # TODO: use factory pattern to create the input data
        if node_type == NodeType.TYPE_SCHEMA_LINKING:
            input_data_cls = SchemaLinkingInput
        elif node_type == NodeType.TYPE_GENERATE_SQL:
            input_data_cls = GenerateSQLInput
        elif node_type == NodeType.TYPE_EXECUTE_SQL:
            input_data_cls = ExecuteSQLInput
        elif node_type == NodeType.TYPE_REFLECT:
            input_data_cls = ReflectionInput
        elif node_type == NodeType.TYPE_REASONING:
            input_data_cls = ReasoningInput
        elif node_type == NodeType.TYPE_OUTPUT:
            input_data_cls = OutputInput
        elif node_type == NodeType.TYPE_FIX:
            input_data_cls = FixInput
        elif node_type == NodeType.TYPE_DOC_SEARCH:
            input_data_cls = DocSearchInput
        elif node_type == NodeType.TYPE_GENERATE_METRICS:
            input_data_cls = GenerateMetricsInput
        elif node_type == NodeType.TYPE_GENERATE_SEMANTIC_MODEL:
            input_data_cls = GenerateSemanticModelInput
        elif node_type == NodeType.TYPE_SEARCH_METRICS:
            input_data_cls = SearchMetricsInput
        elif node_type == NodeType.TYPE_PARALLEL:
            input_data_cls = ParallelInput
        elif node_type == NodeType.TYPE_SELECTION:
            input_data_cls = SelectionInput
        elif node_type == NodeType.TYPE_SUBWORKFLOW:
            input_data_cls = SubworkflowInput
        elif node_type == NodeType.TYPE_COMPARE:
            input_data_cls = CompareInput
        elif node_type == NodeType.TYPE_DATE_PARSER:
            input_data_cls = DateParserInput
        elif node_type == NodeType.TYPE_CHAT:
            input_data_cls = ChatNodeInput
        elif node_type == NodeType.TYPE_CHATBOT:
            input_data_cls = GenSQLNodeInput
        else:
            raise NotImplementedError(f"node_type {node_type} not implemented")

        if ignore_require_check:
            input_data_cls = cls.make_optional_model(input_data_cls)

        return input_data_cls(**input_data)

    # By default, Pydantic v2 validates required fields, but since we are using it as a config,
    # we don't need that strict validation. Therefore, we introduce this to relax the checks.
    def make_optional_model(base_model: type[BaseModel], name_suffix="_Relaxed"):
        # Get field types from class annotations
        type_hints = get_type_hints(base_model)

        fields = {name: (Optional[typ], None) for name, typ in type_hints.items()}

        new_model = create_model(base_model.__name__ + name_suffix, __base__=base_model, **fields)
        return new_model
