"""The active semantic modeling prompt uses the Dosi authoring contract."""

import pytest

from datus.prompts.prompt_manager import get_prompt_manager


@pytest.mark.parametrize(
    "datasource_dialect, expected_dialect",
    [
        ("duckdb", "DUCKDB"),
        ("postgresql", "POSTGRESQL"),
        ("snowflake", "SNOWFLAKE"),
        ("sqlite", "ANSI_SQL"),
        ("hive", "ANSI_SQL"),
        ("  duckdb  ", "DUCKDB"),
    ],
)
def test_semantic_modeling_template_maps_dialect_to_engine_enum(datasource_dialect: str, expected_dialect: str):
    text = get_prompt_manager().render_template(
        template_name="semantic_modeling_system",
        authoring_scope="full",
        current_datasource="warehouse",
        current_datasource_dialect=datasource_dialect,
        kind_subdir="subject/semantic_models/warehouse",
        has_ask_user_tool=False,
    )
    assert f"Dosi expression dialect: `{expected_dialect}`" in text
    assert 'scope="all"' in text
    assert 'scope="semantic_model"' in text
    assert "<required_skill>" in text
