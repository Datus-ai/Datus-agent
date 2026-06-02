# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.agent.agent import _resolve_benchmark_sql_context
from datus.configuration.agent_config import BenchmarkConfig


class _Connector:
    catalog_name = "connector_catalog"
    database_name = "connector_database"
    schema_name = "connector_schema"

    def get_current_context(self):
        return {
            "catalog_name": "default_catalog",
            "database_name": "ac_manage",
            "schema_name": "",
        }


def test_resolve_benchmark_sql_context_uses_explicit_coordinate_keys():
    cfg = BenchmarkConfig(
        catalog_key="__datus_catalog",
        database_key="__datus_database",
        schema_key="__datus_schema",
    )
    row = {
        "__datus_datasource": "starrocks",
        "__datus_catalog": "custom_catalog",
        "__datus_database": "analytics",
        "__datus_schema": "public",
    }

    assert _resolve_benchmark_sql_context(cfg, row, _Connector()) == {
        "catalog_name": "custom_catalog",
        "database_name": "analytics",
        "schema_name": "public",
    }


def test_resolve_benchmark_sql_context_falls_back_to_connector_context():
    cfg = BenchmarkConfig(
        datasource_key="__datus_datasource",
        catalog_key="__datus_catalog",
        database_key="__datus_database",
        schema_key="__datus_schema",
    )
    row = {"__datus_datasource": "starrocks"}

    assert _resolve_benchmark_sql_context(cfg, row, _Connector()) == {
        "catalog_name": "default_catalog",
        "database_name": "ac_manage",
        "schema_name": "",
    }
