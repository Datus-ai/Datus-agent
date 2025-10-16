# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
import re
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from datus.agent.node import Node
from datus.agent.workflow import Workflow
from datus.schemas.action_history import ActionHistory, ActionHistoryManager
from datus.schemas.generate_metrics_node_models import GenerateMetricsInput, GenerateMetricsResult
from datus.schemas.generate_semantic_model_node_models import GenerateSemanticModelInput
from datus.tools.db_tools.db_manager import db_manager_instance
from datus.tools.llms_tools import LLMTool
from datus.tools.llms_tools.generate_metrics import generate_metrics_with_mcp_stream
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names, parse_table_name_parts

logger = get_logger(__name__)


def _quote_pair_for_db(db_type: str) -> Tuple[str, str]:
    """Return the opening and closing quote characters for a DB type."""

    normalized = (db_type or "").lower()
    if normalized in {DBType.MYSQL.value, DBType.STARROCKS.value}:
        return "`", "`"
    if normalized in {DBType.MSSQL.value, DBType.SQLSERVER.value}:
        return "[", "]"
    return '"', '"'


_STRING_LITERAL_PATTERN = re.compile(r"""('(?:''|[^'])*'|"(?:""|[^"])*")""")
_HYPHEN_IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*-[A-Za-z0-9_-]*)\b")


def _quote_segment(segment: str, left_quote: str, right_quote: str) -> str:
    """Quote hyphenated identifiers inside a non-string segment."""

    def repl(match: re.Match[str]) -> str:
        identifier = match.group(1)
        start, end = match.span(1)
        prev_char = segment[start - 1] if start > 0 else ""
        next_char = segment[end] if end < len(segment) else ""

        # Check for already-quoted identifiers (handle boundaries gracefully)
        if prev_char == left_quote and next_char == right_quote:
            return identifier
        if prev_char in {"`", '"'} and next_char == prev_char:
            return identifier
        if prev_char == "[" and next_char == "]":
            return identifier

        return f"{left_quote}{identifier}{right_quote}"

    return _HYPHEN_IDENTIFIER_PATTERN.sub(repl, segment)


def quote_hyphenated_identifiers(sql: str, db_type: str) -> str:
    """Quote unquoted identifiers that contain hyphens; strings are left untouched."""

    if not sql or "-" not in sql:
        return sql

    left_quote, right_quote = _quote_pair_for_db(db_type)
    segments = _STRING_LITERAL_PATTERN.split(sql)

    for idx in range(0, len(segments), 2):
        segments[idx] = _quote_segment(segments[idx], left_quote, right_quote)

    return "".join(segments)


class GenerateMetricsNode(Node):
    def _check_semantic_model_files_exist(self, table_names: List[str]) -> List[str]:
        """Check which semantic model files are missing for given table names.

        Returns:
            List of table names that don't have semantic model files
        """
        model_path = os.getenv("MF_MODEL_PATH")
        if not model_path:
            logger.warning("MF_MODEL_PATH environment variable is not set")
            return table_names

        if not os.path.exists(model_path):
            logger.warning(f"Model path does not exist: {model_path}")
            return table_names

        missing_tables = []
        for table_name in table_names:
            table_name = table_name.split(".")[-1]
            yaml_file = os.path.join(model_path, f"{table_name}.yaml")
            yml_file = os.path.join(model_path, f"{table_name}.yml")

            if not (os.path.exists(yaml_file) or os.path.exists(yml_file)):
                missing_tables.append(table_name)
                logger.info(f"Semantic model file missing for table: {table_name}")
            else:
                logger.debug(f"Semantic model file exists for table: {table_name}")

        return missing_tables

    async def _generate_missing_semantic_models(self, missing_tables: List[str]) -> Optional[str]:
        """Generate semantic models for missing tables.

        Returns:
            Error message if any generation fails, None if all succeed
        """
        if not missing_tables:
            return None

        logger.info(f"Generating semantic models for {len(missing_tables)} missing tables: {missing_tables}")

        # Get database manager and connector
        db_manager = db_manager_instance(self.agent_config.namespaces)
        current_namespace = self.agent_config.current_namespace

        for table_name in missing_tables:
            try:
                # Parse table name parts
                table_parts = parse_table_name_parts(table_name, self.input.sql_task.database_type)

                # Use parsed parts or fallback to sql_task values
                catalog_name = table_parts["catalog_name"] or self.input.sql_task.catalog_name
                database_name = table_parts["database_name"] or self.input.sql_task.database_name
                schema_name = table_parts["schema_name"] or self.input.sql_task.schema_name
                actual_table_name = table_parts["table_name"]

                logger.debug(f"Generating semantic model for table: {actual_table_name}")

                # Get database connector
                connector = None
                try:
                    connector = db_manager.get_conn(current_namespace, database_name)

                    # Get table DDL
                    tables_with_ddl = connector.get_tables_with_ddl(
                        tables=[actual_table_name],
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                    )

                    if not tables_with_ddl:
                        return f"No DDL found for table: {table_name}"

                    # Create semantic model input
                    semantic_input = GenerateSemanticModelInput(
                        sql_task=self.input.sql_task,
                        table_name=table_name,
                        prompt_version=self.input.prompt_version,
                    )

                    # Generate semantic model using direct MCP call to avoid asyncio issues
                    try:
                        # Use the model's generate_with_mcp directly with proper async handling
                        from datus.prompts.generate_semantic_model import get_generate_semantic_model_prompt
                        from datus.prompts.prompt_manager import prompt_manager
                        from datus.tools.mcp_server import MCPServer
                        from datus.utils.json_utils import extract_json_str

                        # Setup for direct MCP call
                        prompt = get_generate_semantic_model_prompt(
                            database_type=self.agent_config.db_type,
                            table_definition=tables_with_ddl[0]["definition"],
                            prompt_version=semantic_input.prompt_version,
                        )

                        instruction = prompt_manager.get_raw_template(
                            "generate_semantic_model_system", semantic_input.prompt_version
                        )

                        filesystem_mcp_server = MCPServer.get_filesystem_mcp_server(path=os.getenv("MF_MODEL_PATH"))
                        metricflow_mcp_server = MCPServer.get_metricflow_mcp_server(
                            database_name=database_name, db_config=self.agent_config.current_db_config(database_name)
                        )

                        # Use await instead of asyncio.run since we're already in async context
                        exec_result = await self.model.generate_with_tools(
                            prompt=prompt,
                            mcp_servers={
                                "filesystem_mcp_server": filesystem_mcp_server,
                                "metricflow_mcp_server": metricflow_mcp_server,
                            },
                            instruction=instruction,
                            output_type=str,
                            max_turns=20,
                        )

                        # Parse result
                        try:
                            json.loads(extract_json_str(exec_result["content"]))
                            result_success = True
                            result_error = ""
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse semantic model result: {e}")
                            result_success = False
                            result_error = f"Failed to parse result: {e}"

                        if not result_success:
                            return f"Failed to generate semantic model for table {table_name}: {result_error}"

                    except Exception as e:
                        return f"Failed to generate semantic model for table {table_name}: {str(e)}"

                    logger.info(f"Successfully generated semantic model for table: {table_name}")

                finally:
                    if connector:
                        try:
                            connector.close()
                        except Exception as e:
                            logger.warning(f"Error closing database connector: {e}")

            except Exception as e:
                logger.error(f"Error generating semantic model for table {table_name}: {e}")
                return f"Error generating semantic model for table {table_name}: {str(e)}"

        return None

    def execute(self):
        self.result = self._generate_metrics()

    async def execute_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Execute metrics generation with streaming support."""
        async for action in self._generate_metrics_stream(action_history_manager):
            yield action

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
                metrics=[],
            )

        try:
            # Check and generate semantic models if needed
            db_type = self.input.sql_task.database_type
            sql_query = quote_hyphenated_identifiers(self.input.sql_query, db_type)
            table_names = extract_table_names(sql_query, self.input.sql_task.database_type)
            logger.info(f"Extracted table names from SQL: {table_names}")

            if table_names:
                missing_tables = self._check_semantic_model_files_exist(table_names)

                if missing_tables:
                    logger.info(f"Missing semantic model files for tables: {missing_tables}")

                    # Run async method properly
                    import asyncio
                    import concurrent.futures

                    error_msg = None
                    try:
                        # Check if we're in an event loop
                        asyncio.get_running_loop()
                        # We're in an event loop, run the async code in a new thread with its own loop
                        logger.info("Running semantic model generation in separate thread")

                        def run_in_thread():
                            return asyncio.run(self._generate_missing_semantic_models(missing_tables))

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            error_msg = future.result()

                    except RuntimeError:
                        # No running loop, safe to use asyncio.run directly
                        error_msg = asyncio.run(self._generate_missing_semantic_models(missing_tables))

                    if error_msg:
                        logger.error(f"Semantic model generation failed: {error_msg}")
                        return GenerateMetricsResult(
                            success=False,
                            error=f"Semantic model generation failed: {error_msg}",
                            metrics=[],
                        )
                    else:
                        logger.info("Successfully generated all missing semantic models")
                else:
                    logger.info("All required semantic model files exist")

            # Generate metrics
            tool = LLMTool(self.model)
            logger.debug(f"Generate metrics input: {type(self.input)} {self.input}")
            return tool.generate_metrics(
                self.input,
                self.agent_config.current_db_config(self.input.sql_task.database_name),
                self.tools,
            )
        except Exception as e:
            logger.error(f"Metrics generation execution error: {str(e)}")
            return GenerateMetricsResult(
                success=False,
                error=str(e),
                metrics=[],
            )

    async def _generate_metrics_stream(
        self, action_history_manager: Optional[ActionHistoryManager] = None
    ) -> AsyncGenerator[ActionHistory, None]:
        """Generate metrics with streaming support and action history tracking."""
        if not self.model:
            logger.error("Model not available for metrics generation")
            return

        try:
            # Check and generate semantic models if needed
            db_type = self.input.sql_task.database_type
            sql_query = quote_hyphenated_identifiers(self.input.sql_query, db_type)
            table_names = extract_table_names(sql_query, self.input.sql_task.database_type)
            logger.info(f"Extracted table names from SQL: {table_names}")

            if table_names:
                missing_tables = self._check_semantic_model_files_exist(table_names)

                if missing_tables:
                    logger.info(f"Missing semantic model files for tables: {missing_tables}")

                    # Yield semantic model generation action
                    yield ActionHistory(
                        action_id="semantic_model_check",
                        role="workflow",
                        messages=f"Checking semantic model files for {len(table_names)} tables",
                        action_type="semantic_model_generation",
                        input={"table_names": table_names},
                        output={"missing_tables": missing_tables},
                        status="success",
                    )

                    error_msg = await self._generate_missing_semantic_models(missing_tables)
                    if error_msg:
                        logger.error(f"Semantic model generation failed: {error_msg}")
                        yield ActionHistory(
                            action_id="semantic_model_generation",
                            role="workflow",
                            messages=f"Semantic model generation failed: {error_msg}",
                            action_type="semantic_model_generation",
                            input={"missing_tables": missing_tables},
                            output={"success": False, "error": error_msg},
                            status="failed",
                        )
                        return
                    else:
                        logger.info("Successfully generated all missing semantic models")
                        yield ActionHistory(
                            action_id="semantic_model_generation",
                            role="workflow",
                            messages=f"Generated semantic models for {len(missing_tables)} tables",
                            action_type="semantic_model_generation",
                            input={"missing_tables": missing_tables},
                            output={"success": True, "generated_count": len(missing_tables)},
                            status="success",
                        )
                else:
                    logger.info("All required semantic model files exist")
                    yield ActionHistory(
                        action_id="semantic_model_check",
                        role="workflow",
                        messages="All required semantic model files exist",
                        action_type="semantic_model_generation",
                        input={"table_names": table_names},
                        output={"missing_tables": []},
                        status="success",
                    )

            # Stream the metrics generation
            async for action in generate_metrics_with_mcp_stream(
                model=self.model,
                input_data=self.input,
                tools=self.tools,
                db_config=self.agent_config.current_db_config(self.input.sql_task.database_name),
                tool_config={},
                action_history_manager=action_history_manager,
            ):
                yield action

        except Exception as e:
            logger.error(f"Metrics generation streaming error: {str(e)}")
            yield ActionHistory(
                action_id="semantic_model_error",
                role="workflow",
                messages=f"Semantic model check failed: {str(e)}",
                action_type="semantic_model_generation",
                input={"sql_query": self.input.sql_query},
                output={"success": False, "error": str(e)},
                status="failed",
            )
            raise

    def update_context(self, workflow: Workflow) -> Dict:
        pass

    def setup_input(self, workflow: Workflow) -> Dict:
        next_input = GenerateMetricsInput(
            sql_task=workflow.task,
            sql_query="",
        )
        self.input = next_input
        return {"success": True, "message": "Metrics generated", "suggestions": [next_input]}
