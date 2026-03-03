# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Built-in URI builders and context resolvers for dialects without external adapter packages.

Covers: BigQuery, MSSQL, Oracle
"""

from typing import Dict, Optional, Tuple, Union

from sqlalchemy.engine.url import URL, make_url

from datus.utils.exceptions import DatusException, ErrorCode

# ---------------------------------------------------------------------------
# Tiny helpers (duplicated from db_manager to avoid circular imports)
# ---------------------------------------------------------------------------


def _clean_str(value: Optional[Union[str, int]]) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item:
                return str(item).strip()
        return ""
    return str(value).strip()


def _value_or_none(value: Optional[Union[str, int]]) -> Optional[str]:
    cleaned = _clean_str(value)
    return cleaned or None


def _port_or_none(port_value: Optional[Union[str, int]]) -> Optional[int]:
    cleaned = _clean_str(port_value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------


def build_bigquery_uri(db_config) -> str:
    project = _clean_str(db_config.catalog) or _clean_str(db_config.host)
    dataset = _clean_str(db_config.database) or _clean_str(db_config.schema)
    if not project or not dataset:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message="BigQuery configuration requires `catalog` (project) and `database` (dataset)",
        )
    return str(URL.create(drivername="bigquery", host=project, database=dataset))


def resolve_bigquery_context(db_config, uri: str) -> Tuple[str, str, str, str]:
    url = make_url(uri)
    query_params: Dict[str, str] = {k: _clean_str(v) for k, v in url.query.items()}
    database = _clean_str(url.database)

    catalog = _clean_str(url.host) or _clean_str(db_config.catalog)
    dataset = database or _clean_str(db_config.database) or _clean_str(db_config.schema)
    schema = query_params.get("schema") or dataset
    return "bigquery", catalog, dataset, schema


# ---------------------------------------------------------------------------
# MSSQL
# ---------------------------------------------------------------------------


def build_mssql_uri(db_config) -> str:
    query: Dict[str, str] = {"driver": "ODBC Driver 17 for SQL Server"}
    if db_config.schema:
        query["schema"] = _clean_str(db_config.schema)
    return str(
        URL.create(
            drivername="mssql+pyodbc",
            username=_value_or_none(db_config.username),
            password=_value_or_none(db_config.password),
            host=_value_or_none(db_config.host),
            port=_port_or_none(db_config.port),
            database=_value_or_none(db_config.database),
            query=query,
        )
    )


def resolve_mssql_context(db_config, uri: str) -> Tuple[str, str, str, str]:
    url = make_url(uri)
    query_params: Dict[str, str] = {k: _clean_str(v) for k, v in url.query.items()}
    database = _clean_str(url.database) or _clean_str(db_config.database)
    schema = query_params.get("schema") or _clean_str(db_config.schema) or "dbo"
    return "mssql", "", database, schema


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def build_oracle_uri(db_config) -> str:
    query: Dict[str, str] = {}
    service = _clean_str(db_config.database)
    sid = _clean_str(db_config.schema)
    if service:
        query["service_name"] = service
    elif sid:
        query["sid"] = sid
    return str(
        URL.create(
            drivername="oracle+cx_oracle",
            username=_value_or_none(db_config.username),
            password=_value_or_none(db_config.password),
            host=_value_or_none(db_config.host),
            port=_port_or_none(db_config.port),
            query=query or None,
        )
    )


def resolve_oracle_context(db_config, uri: str) -> Tuple[str, str, str, str]:
    url = make_url(uri)
    query_params: Dict[str, str] = {k: _clean_str(v) for k, v in url.query.items()}
    database = _clean_str(url.database)
    service = query_params.get("service_name") or query_params.get("sid")
    database = service or database or _clean_str(db_config.database)
    schema = query_params.get("schema") or _clean_str(db_config.schema) or _clean_str(url.username)
    return "oracle", "", database, schema
