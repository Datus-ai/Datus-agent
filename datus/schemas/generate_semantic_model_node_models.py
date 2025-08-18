from pydantic import Field

from datus.schemas.base import BaseInput, BaseResult
from datus.schemas.node_models import SqlTask


class GenerateSemanticModelInput(BaseInput):
    """
    Input model for generating semantic model.
    """

    sql_task: SqlTask = Field(..., description="The SQL task to generate semantic model from")
    table_name: str = Field(..., description="The table name to generate semantic model from")
    prompt_version: str = Field(default="1.0", description="Version for prompt")


class GenerateSemanticModelResult(BaseResult):
    """
    Result model for generating semantic model.
    """

    table_name: str = Field(..., description="The table name of the semantic model")
    semantic_model_file: str = Field(..., description="The semantic model file of this request")
