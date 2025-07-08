from datus.utils.loggings import get_logger

from .prompt_manager import prompt_manager

logger = get_logger(__name__)


def get_generate_metrics_prompt(
    database_type: str,
    sql_query: str,
    description: str,
    prompt_version: str = "1.0",
) -> str:
    user_content = prompt_manager.render_template(
        "generate_metrics_user",
        version=prompt_version,
        database_type=database_type,
        sql_query=sql_query,
        description=description,
    )

    return user_content
