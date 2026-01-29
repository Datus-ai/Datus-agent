# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

import pyarrow as pa

from datus.models.base import LLMBaseModel
from datus.prompts.schema_lineage import gen_prompt, gen_summary_prompt
from datus.schemas.node_models import TableSchema
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.storage.schema_metadata import SchemaStorage
from datus.tools.base import BaseTool
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.json_utils import llm_result2json
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import metadata_identifier

logger = get_logger(__name__)


class MatchSchemaTool(BaseTool):
    def __init__(self, model: LLMBaseModel, storage: SchemaStorage, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.storage = storage

    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, SchemaLinkingInput):
            return False
        return True

    def execute(self, input_data: SchemaLinkingInput) -> SchemaLinkingResult:
        table_metadata = self.storage.search_all(database_name=input_data.database_name)
        if table_metadata.num_rows == 0:
            return SchemaLinkingResult(
                success=False,
                error=f"No table metadata found in {input_data.database_name}",
                table_schemas=[],
                schema_count=0,
                table_values=[],
                value_count=0,
            )
        try:
            all_tables = gen_all_table_dict(table_metadata)
            match_result = self.match_schema(input_data, table_metadata, all_tables)
            return self._process_match_result(input_data, match_result, input_data.database_name, all_tables)
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_EXECUTION_FAILED,
                message=f"Schema linking by LLM execution error cause by {str(e)}",
            ) from e

    def _process_match_result(
        self,
        input_data: SchemaLinkingInput,
        match_result: Dict[str, Any],
        database_name: str,
        all_tables: Dict[str, Dict[str, Any]],
    ) -> SchemaLinkingResult:
        if not match_result:
            return SchemaLinkingResult(
                success=False,
                error="No match result found",
                table_schemas=[],
                schema_count=0,
                table_values=[],
                value_count=0,
            )

        schema_result = []
        for table_schema in match_result:
            table_name = table_schema["table"]
            if isinstance(table_name, str):
                table_name = [table_name]
            for sub_table_name in table_name:
                sub_table_name = sub_table_name.split(".")
                full_table_name = f"{database_name}.{sub_table_name[-2]}.{sub_table_name[-1]}"
                if full_table_name in all_tables:
                    matched_table = all_tables[full_table_name]
                    if "identifier" in matched_table:
                        identifier = matched_table["identifier"]
                    else:
                        identifier = metadata_identifier(
                            dialect=input_data.database_type,
                            catalog_name=matched_table["catalog_name"],
                            database_name=database_name,
                            schema_name=matched_table["schema_name"],
                            table_name=matched_table["table_name"],
                        )
                    schema_result.append(
                        TableSchema(
                            identifier=identifier,
                            catalog_name=matched_table["catalog_name"],
                            database_name=database_name,
                            schema_name=matched_table["schema_name"],
                            table_name=matched_table["table_name"],
                            definition=matched_table["definition"],
                            table_type=matched_table["table_type"],
                        )
                    )
                else:
                    logger.warning(f"Table {full_table_name} not found  metadata in {database_name}")
        return SchemaLinkingResult(
            success=True,
            error=None,
            table_schemas=schema_result,
            schema_count=len(schema_result),
            table_values=[],
            value_count=0,
        )

    def map_reduce_match_schema(
        self,
        input_data: SchemaLinkingInput,
        table_metadata: pa.Table,
        all_table_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Extract schema names from PyArrow Table using batch processing
        if DBType.support_schema(input_data.database_type):
            all_schemas = set(table_metadata["schema_name"].unique().to_pylist())

            logger.debug(
                f"Current task should merge by schema: table_size={table_metadata.num_rows}, schemas={len(all_schemas)}"
                f", question={input_data.input_text}"
            )
            should_match_tables = self.storage.search_top_tables_by_every_schema(
                input_data.input_text,
                database_name=input_data.database_name,
                catalog_name=input_data.catalog_name,
                all_schemas=all_schemas,
                top_n=20,
            )
            return self._match_schema(input_data, all_table_dict, should_match_tables)
        else:
            return self._match_schema(input_data, all_table_dict, table_metadata)

    def match_schema(
        self,
        input_data: SchemaLinkingInput,
        table_metadata: pa.Table,
        all_table_dict: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Match the schema of the input database with the table metadata by LLM

        Args:
            input_data: Database name, user query and top_n
            table_metadata: The table metadata as PyArrow Table
            all_table_dict: Dictionary mapping full table names to their metadata

        Returns:
            json string: matched tables and their scores. Example:
           [
                {"table": "table1", "score": 0.95, "reasons": ["reason1", "reason2"]},
                {"table": "table2", "score": 0.9, "reasons": ["reason1", "reason2"]},
            ] //...
        """
        # Check if we need to use map-reduce approach based on row count
        if table_metadata.num_rows > 200:
            return self.map_reduce_match_schema(input_data, table_metadata, all_table_dict)
        else:
            # Convert PyArrow Table to list for compatibility with existing methods
            # Using batch processing to handle large datasets efficiently

            match_result = self._match_schema(input_data, all_table_dict, table_metadata)
            return match_result

    def _match_schema(
        self,
        input_data: SchemaLinkingInput,
        all_table_dict: Dict[str, Dict[str, Any]],
        table_metadata: pa.Table,
    ) -> Dict[str, Any]:
        messages = gen_prompt(
            input_data.database_type,
            input_data.database_name,
            input_data.input_text,
            table_metadata.to_pylist(),
            input_data.prompt_version,
            # input_data.top_n,
        )
        task_size = self.count_task_size(self.model, input_data.input_text, str(messages))
        if task_size == 1:
            return llm_result2json(self.model.generate(messages, temperature=0.3))
        else:
            return self.split_and_match_schema(input_data, all_table_dict, table_metadata, task_size)

    def count_task_size(self, model: LLMBaseModel, user_question: str, prompt: Union[str, List[Dict[str, str]]]) -> int:
        tokens_count = model.token_count(prompt)
        max_tokens = model.max_tokens()
        if tokens_count > max_tokens:
            task_size = math.ceil(tokens_count / max_tokens)
            logger.info(
                f"""query ```{user_question}``` ; tokens count: {tokens_count}, will split into {task_size} tasks"""
            )
            return task_size
        else:
            return 1

    def split_and_match_schema(
        self,
        input_data: SchemaLinkingInput,
        all_table_dict: Dict[str, Dict[str, Any]],
        table_metadata: List[Dict[str, Any]],
        task_size: int,
    ) -> Dict[str, Any]:
        futures = []

        with ThreadPoolExecutor(max_workers=min(task_size, 10)) as executor:
            step = len(table_metadata) // task_size
            for i in range(task_size):
                start = i * step
                if start >= len(table_metadata):
                    break
                end = start + step
                if end > len(table_metadata):
                    end = len(table_metadata)
                futures.append(
                    executor.submit(self._match_schema, input_data, all_table_dict, table_metadata[start:end])
                )
        # reduce the result
        summary_metadata = []
        for future in futures:
            matched_tables = future.result()

            if isinstance(matched_tables, dict):
                summary_metadata = parse_matched_tables(input_data.database_name, matched_tables, all_table_dict)
            elif isinstance(matched_tables, list):
                for mt in matched_tables:
                    summary_metadata.extend(parse_matched_tables(input_data.database_name, mt, all_table_dict))

            else:
                logger.warning(
                    f"Invalid matched_tables type, excepted: list or dict, current type: {type(matched_tables)},"
                    f"matched_tables: {matched_tables}"
                )
        logger.info(f"user query: {input_data.input_text}, child tasks result: {len(summary_metadata)}")

        summary_prompt = gen_summary_prompt(
            input_data.database_type,
            input_data.database_name,
            input_data.input_text,
            summary_metadata,
            # input_data.top_n,
        )
        # todo truncate
        summary_response = self.model.generate(summary_prompt)
        return llm_result2json(summary_response)


def parse_matched_tables(
    database_name: str, tables: Dict[str, Any], all_table_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    summary_metadata = []
    table = tables["table"]
    if isinstance(table, str):
        table = [table]
    for sub_table_name in table:
        tb_arry = sub_table_name.split(".")
        sm = {
            "database_name": database_name,
            "schema_name": tb_arry[-2],
            "table_name": tb_arry[-1],
            "score": tables["score"],
            "reasons": tables["reasons"],
        }
        full_table_name = f"{sm['database_name']}.{sm['schema_name']}.{sm['table_name']}"
        if full_table_name in all_table_dict:
            sm["schema_text"] = all_table_dict[full_table_name]["schema_text"]
            summary_metadata.append(sm)
        else:
            logger.warning(f"Table {full_table_name} not found  metadata in {database_name}")
    return summary_metadata


def gen_all_table_dict(all_tables: pa.Table) -> Dict[str, Any]:
    """
    Convert PyArrow Table to dictionary with batch processing to handle large datasets efficiently.

    Args:
        database_name: Name of the database
        all_tables: PyArrow Table containing table metadata

    Returns:
        Dictionary mapping full table names to their metadata
    """
    batch_size = 1000
    table_metadata = {}

    # Process table in batches to handle large datasets efficiently
    for i in range(0, all_tables.num_rows, batch_size):
        current_batch_size = min(batch_size, all_tables.num_rows - i)
        batch = all_tables.slice(i, current_batch_size)

        # Convert batch to list of dictionaries for processing
        batch_records = batch.to_pylist()

        for record in batch_records:
            table_metadata[record["identifier"]] = record

    return table_metadata
