import json

from datus.models.base import LLMBaseModel
from datus.prompts.gen_sql import get_sql_prompt
from datus.schemas.node_models import GenerateSQLInput, GenerateSQLResult
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.time_utils import get_default_current_date
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
def generate_sql(model: LLMBaseModel, input_data: GenerateSQLInput) -> GenerateSQLResult:
    """Generate SQL query using the provided model."""
    if not isinstance(input_data, GenerateSQLInput):
        raise ValueError("Input must be a GenerateSQLInput instance")

    try:
        # Format the prompt with schema list
        prompt = get_sql_prompt(
            database_type=input_data.get("database_type", DBType.SQLITE),
            table_schemas=input_data.table_schemas,
            data_details=input_data.data_details,
            metrics=input_data.metrics,
            question=input_data.sql_task.task,
            external_knowledge=input_data.external_knowledge,
            prompt_version=input_data.prompt_version,
            context=[sql_context.to_str() for sql_context in input_data.contexts],
            max_table_schemas_length=input_data.max_table_schemas_length,
            max_data_details_length=input_data.max_data_details_length,
            max_context_length=input_data.max_context_length,
            max_value_length=input_data.max_value_length,
            max_text_mark_length=input_data.max_text_mark_length,
            database_docs=input_data.database_docs,
            current_date=get_default_current_date(input_data.sql_task.current_date),
            date_ranges=getattr(input_data.sql_task, "date_ranges", ""),
        )

        logger.debug(f"Generated SQL prompt:  {type(model)}, {prompt}")
        # Generate SQL using the provided model
        sql_query = model.generate_with_json_output(prompt)
        logger.debug(f"Generated SQL: {sql_query}")

        # Clean and parse the response
        if isinstance(sql_query, str):
            # Remove markdown code blocks if present
            sql_query = sql_query.strip().replace("```json\n", "").replace("\n```", "")
            # Remove SQL comments
            cleaned_lines = []
            for line in sql_query.split("\n"):
                line = line.strip()
                if line and not line.startswith("--"):
                    cleaned_lines.append(line)
            cleaned_sql = " ".join(cleaned_lines)
            try:
                sql_query_dict = json.loads(cleaned_sql)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse cleaned SQL: {cleaned_sql}")
                return GenerateSQLResult(success=False, error="Invalid JSON format", sql_query=sql_query)
        else:
            sql_query_dict = sql_query

        # Return result as GenerateSQLResult
        if sql_query_dict and isinstance(sql_query_dict, dict):
            return GenerateSQLResult(
                success=True,
                error=None,
                sql_query=sql_query_dict.get("sql", ""),
                tables=sql_query_dict.get("tables", []),
                explanation=sql_query_dict.get("explanation"),
            )
        else:
            return GenerateSQLResult(success=False, error="sql generation failed, no result", sql_query=sql_query)
    except json.JSONDecodeError as e:
        logger.error(f"SQL json decode failed: {str(e)} SQL: {sql_query}")
        return GenerateSQLResult(success=False, error=str(e), sql_query={sql_query})
    except Exception as e:
        logger.error(f"SQL generation failed: {str(e)}")
        return GenerateSQLResult(success=False, error=str(e), sql_query="")
