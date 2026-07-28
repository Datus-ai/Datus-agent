# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import MagicMock, patch

from datus_db_core import connector_registry

from datus.prompts.database_notes import get_database_notes
from datus.prompts.gen_sql import get_sql_prompt
from datus.prompts.reasoning_sql_with_mcp import get_reasoning_prompt


def test_adapter_sql_generation_notes_are_used(monkeypatch):
    monkeypatch.setattr(
        connector_registry,
        "get_sql_generation_notes",
        lambda dialect: "Use project.table or project.schema.table." if dialect == "flexdb" else None,
        raising=False,
    )

    assert "project.schema.table" in get_database_notes("flexdb")


def test_adapter_notes_reach_both_sql_prompt_builders(monkeypatch):
    notes = "Use project.table or project.schema.table."
    monkeypatch.setattr(
        connector_registry,
        "get_sql_generation_notes",
        lambda dialect: notes if dialect == "flexdb" else None,
        raising=False,
    )

    sql_prompt_manager = MagicMock()
    sql_prompt_manager.render_template.return_value = "rendered"
    with patch("datus.prompts.gen_sql.get_prompt_manager", return_value=sql_prompt_manager):
        get_sql_prompt(
            database_type="flexdb",
            table_schemas="",
            data_details=[],
            metrics=[],
            question="question",
        )
    sql_user_call = next(
        call for call in sql_prompt_manager.render_template.call_args_list if call.args[0] == "gen_sql_user"
    )
    assert notes in sql_user_call.kwargs["database_notes"]

    reasoning_prompt_manager = MagicMock()
    reasoning_prompt_manager.render_template.return_value = "rendered"
    with patch(
        "datus.prompts.reasoning_sql_with_mcp.get_prompt_manager",
        return_value=reasoning_prompt_manager,
    ):
        get_reasoning_prompt(
            database_type="flexdb",
            table_schemas=[],
            data_details=[],
            metrics="",
            question="question",
            context=[],
        )
    reasoning_call = reasoning_prompt_manager.render_template.call_args
    assert notes in reasoning_call.kwargs["database_notes"]


def test_snowflake_legacy_notes_are_preserved(monkeypatch):
    monkeypatch.setattr(
        connector_registry,
        "get_sql_generation_notes",
        lambda _dialect: None,
        raising=False,
    )
    notes = get_database_notes("snowflake")
    assert "double quotes" in notes
    assert "database_name and schema_name" in notes
