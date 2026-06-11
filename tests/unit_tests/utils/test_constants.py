# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for :mod:`datus.utils.constants`."""

from __future__ import annotations

import argparse

from datus.utils.constants import DBType, EmbeddingProvider, LLMProvider, SQLType


def test_dbtype_str_returns_value() -> None:
    # Python 3.11+ changed str() on (str, Enum) to return "ClassName.MEMBER"
    # instead of the value. StrEnum restores the correct behaviour: str() == the
    # value, which is what argparse uses when building the choices display string.
    assert str(DBType.SQLITE) == "sqlite"
    assert str(DBType.DUCKDB) == "duckdb"


def test_dbtype_equality_with_plain_string() -> None:
    assert DBType.SQLITE == "sqlite"
    assert DBType.DUCKDB == "duckdb"


def test_dbtype_argparse_choices_display_and_accept() -> None:
    # Regression: before the StrEnum fix, argparse displayed "DBType.SQLITE" in
    # the choices column (str() was broken). Input like "sqlite" was still
    # accepted (argparse uses ==, and "sqlite" == DBType.SQLITE is True), but
    # the help text was misleading. StrEnum makes str() return the value so the
    # displayed choices match what users actually type.
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_type", choices=[DBType.SQLITE, "snowflake", DBType.DUCKDB])

    args = parser.parse_args(["--db_type", "sqlite"])
    assert args.db_type == "sqlite"

    args = parser.parse_args(["--db_type", "snowflake"])
    assert args.db_type == "snowflake"


def test_llmprovider_str_returns_value() -> None:
    assert str(LLMProvider.OPENAI) == "openai"
    assert str(LLMProvider.CLAUDE) == "claude"


def test_embeddingprovider_str_returns_value() -> None:
    assert str(EmbeddingProvider.OPENAI) == "openai"
    assert str(EmbeddingProvider.FASTEMBED) == "fastembed"


def test_sqltype_str_returns_value() -> None:
    assert str(SQLType.SELECT) == "select"
    assert str(SQLType.DDL) == "ddl"
