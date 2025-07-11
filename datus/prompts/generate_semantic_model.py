from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def get_generate_semantic_model_prompt(
    database_type: str,
    table_definition: str,
    prompt_version: str = "1.0",
) -> str:
    user_content = prompt_manager.render_template(
        "generate_semantic_model_user",
        version=prompt_version,
        database_type=database_type,
        table_definition=table_definition,
    )

    return user_content
