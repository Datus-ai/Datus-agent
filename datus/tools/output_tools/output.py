# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import json
import os
from io import StringIO
from typing import Any, Optional, Tuple

import pandas as pd

from datus.models.base import LLMBaseModel
from datus.prompts.output_checking import gen_prompt
from datus.schemas.node_models import OutputInput, OutputResult
from datus.tools.base import BaseTool
from datus.tools.db_tools.base import BaseSqlConnector
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class OutputTool(BaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_input(self, input_data: Any):
        """"""

    def execute(
        self,
        input_data: OutputInput,
        sql_connector: BaseSqlConnector,
        model: Optional[LLMBaseModel] = None,
    ) -> OutputResult:
        target_dir = input_data.output_dir
        os.makedirs(target_dir, exist_ok=True)
        if input_data.finished and not input_data.error:
            final_sql_query, final_sql_result = self.check_sql(input_data, sql_connector, model)
            if final_sql_query and final_sql_result is None:
                final_sql_result = input_data.sql_result
                final_sql_query = input_data.gen_sql

            if input_data.file_type == "sql":
                result_file = save_sql(
                    target_dir, input_data.task_id, final_sql_query if final_sql_query else input_data.gen_sql
                )
            elif input_data.file_type == "json":
                result_file = save_json(target_dir, input_data, final_sql_query, final_sql_result)
            elif input_data.file_type == "csv":
                result_file = save_csv(
                    target_dir,
                    input_data.task_id,
                    final_sql_result if final_sql_result else input_data.sql_result,
                )
            else:
                result_file = save_csv(
                    target_dir,
                    input_data.task_id,
                    final_sql_result if final_sql_result else input_data.sql_result,
                )
                save_sql(target_dir, input_data.task_id, final_sql_query)
                save_json(target_dir, input_data, final_sql_query, final_sql_result, result_file)
            return OutputResult(
                success=True,
                output=result_file,
                sql_query=input_data.gen_sql,
                sql_result=input_data.sql_result,
                sql_query_final=final_sql_query,
                sql_result_final=final_sql_result,
            )
        else:
            file_name = f"{input_data.task_id}.json" if input_data.task_id else "result.json"
            with open(os.path.join(target_dir, file_name), "w") as f:
                json.dump(
                    {
                        "finished": False,
                        "instance_id": input_data.task_id,
                        "instruction": input_data.task,
                        "database_name": input_data.database_name,
                        "error": input_data.error,
                        "gen_sql": input_data.gen_sql,
                        "sql_result": input_data.sql_result,
                    },
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
            return OutputResult(
                success=False,
                output=input_data.error,
                sql_query=input_data.gen_sql,
                sql_result=input_data.sql_result,
            )

    @optional_traceable()
    def check_sql(
        self,
        input_data: OutputInput,
        sql_connector: BaseSqlConnector,
        model: Optional[LLMBaseModel] = None,
    ) -> Tuple[str, str]:
        if not input_data.check_result:
            return input_data.gen_sql, input_data.sql_result
        if not model:
            logger.info("No model provided, return the original SQL and result.")
            return input_data.gen_sql, input_data.sql_result
        prompt = gen_prompt(
            user_question=input_data.task,
            table_schemas=input_data.table_schemas,
            sql_query=input_data.gen_sql,
            sql_execution_result=input_data.sql_result,
            metrics=input_data.metrics,
            external_knowledge=input_data.external_knowledge,
            prompt_version=input_data.prompt_version,
        )
        llm_result = model.generate_with_json_output(prompt)
        if llm_result.get("is_correct", True):
            return input_data.gen_sql, input_data.sql_result
        if "revised_sql" not in llm_result:
            logger.warning(f"No revised SQL in the result: {llm_result}")
            final_sql = input_data.gen_sql
        else:
            final_sql = llm_result.get("revised_sql")
        if not input_data.sql_result:
            if final_sql == input_data.gen_sql:
                return input_data.gen_sql, input_data.sql_result
            else:
                final_result = sql_connector.execute({"sql_query": final_sql})
                if final_result.success:
                    return final_sql, final_result.sql_return
                logger.warning(f"Execute check_sql failed, sql={final_sql}, error={final_result.error}")
                return input_data.gen_sql, input_data.sql_result

        try:
            if "final_columns" in llm_result and llm_result.get("final_columns") and input_data.sql_result:
                final_columns = llm_result.get("final_columns")
                csv_result = input_data.sql_result
                df = pd.read_csv(StringIO(csv_result))
                src_columns = set(df.columns)
                if set(final_columns).issubset(src_columns):
                    df = df[final_columns]
                    final_result = df.to_csv(index=False)
                else:
                    logger.warning(
                        f"The final columns are not subset of the source columns: "
                        f"{final_columns} is not subset of {src_columns}. "
                        "Execute the sql directly."
                    )
                    exe_result = sql_connector.execute({"sql_query": final_sql})
                    if exe_result.success:
                        final_result = exe_result.sql_return
                    else:
                        logger.warning(f"Execute check_sql failed, sql={final_sql}, error={exe_result.error}")
                        return input_data.gen_sql, input_data.sql_result
            else:
                logger.warning(f"No final columns in the result: {llm_result}. Execute the sql directly.")
                final_result = sql_connector.execute({"sql_query": final_sql}).sql_return
            return final_sql, final_result
        except Exception as e:
            logger.error(f"Failed execution based on new sql and results. new_sql=[{final_sql}], error: {e}")
            return input_data.gen_sql, input_data.sql_result


def save_sql(target_dir: str, file_name: str, sql_query: str) -> str:
    sql_file = f"{target_dir}/{file_name}.sql"
    with open(sql_file, "w") as f:
        f.write(sql_query)
    return sql_file


def save_csv(target_dir: str, file_name: str, query_result: str) -> str:
    csv_file = f"{target_dir}/{file_name}.csv"
    with open(csv_file, "w") as f:
        f.write(query_result)
    return csv_file


def save_json(
    target_dir: str,
    input_data: OutputInput,
    final_sql: str,
    final_query_result: str,
    result_file_name: Optional[str] = None,
) -> str:
    json_file = f"{target_dir}/{input_data.task_id}.json"
    result_json: dict[str, Any] = {
        "finished": True,
        "instance_id": input_data.task_id,
        "database_name": input_data.database_name,
        "gen_sql": input_data.gen_sql,
        "result": result_file_name if result_file_name else json_file,
    }
    if input_data.gen_sql != final_sql:
        result_json["gen_sql_final"] = final_sql
    if input_data.sql_result:
        result_json["sql_result"] = input_data.sql_result
        result_json["row_count"] = input_data.row_count

        if input_data.sql_result != final_query_result:
            result_json["sql_result_final"] = final_query_result
    if input_data.task:
        result_json["instruction"] = input_data.task

    with open(json_file, "w") as json_f:
        json.dump(
            result_json,
            json_f,
            ensure_ascii=False,
            indent=4,
        )
    return json_file
