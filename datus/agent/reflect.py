# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Dict

from datus.models.base import LLMBaseModel
from datus.prompts.reflection import get_evaluation_prompt
from datus.schemas.base import BaseInput
from datus.schemas.node_models import STRATEGY_LIST, SqlTask
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


@optional_traceable()
def evaluate_with_model(task: SqlTask, node_input: BaseInput, model: LLMBaseModel) -> Dict:
    """
    Use a language model to evaluate SQL execution results.
    """
    # Get evaluation prompt
    sql_context = node_input.sql_context[-1]
    prompt_version = node_input.prompt_version
    sample_line = node_input.sql_return_sample_line
    sample_return = sql_context.sql_return
    if sample_line != -1 and sql_context.sql_return:
        sample_return_list = sql_context.sql_return.split("\n")[: sample_line + 1]
        sample_return = "\n".join(sample_return_list)

    evaluate_template = get_evaluation_prompt(
        task_description=task.to_str(),
        sql_generation_result=sql_context.sql_query,
        sql_execution_result=(
            f"\nSAMPLE ROWS RETURN: \n{sample_return}\n"
            f"ERROR: {sql_context.sql_error}\n"
            f"Rows_returned: {sql_context.row_count}"
        ),
        prompt_version=prompt_version,
    )

    try:
        # Generate evaluation using the model
        evaluation = model.generate_with_json_output(evaluate_template)
        logger.info(f"Model evaluation: {evaluation}")

        # Get classification or use "UNKNOWN" as default
        classification = evaluation.get("classification", "UNKNOWN")
        if classification not in STRATEGY_LIST:
            logger.warning(f"Invalid strategy: {classification}, using UNKNOWN instead")
            classification = "UNKNOWN"

        return {
            "success": True,
            "error": "",
            "strategy": classification,
            "details": {k: v for k, v in evaluation.items() if k != "classification"},
        }
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return {
            "success": False,
            "error": f"Model evaluation failed, {str(e)}",
            "strategy": "UNKNOWN",
        }
