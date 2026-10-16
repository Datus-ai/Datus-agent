# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Real CLI -> LLM -> HANA query -> answer acceptance tests.

Uses the same terminal-input harness as ``test_chat_agentic.py``. Neither the
model nor the adapter/driver is mocked. Fixtures and an independent SQL oracle
use hdbcli directly, in uniquely named tables removed on teardown.

Environment:
    HANA_HOST/PORT/USER/PASSWORD/SCHEMA: dedicated live test database.
    HANA_ENCRYPT: defaults to true; false is for a local/SSH-forwarded Express.
    HANA_SSL_TRUST_STORE/HANA_SSL_HOST_NAME_IN_CERTIFICATE: optional TLS settings.
    HANA_AGENT_CONFIG: model config source (default tests/conf/agent.yml).
    HANA_AGENT_MODEL: model alias (default the source config's target).
    HANA_REQUIRE_AGENT_E2E=1: missing environment is a failure rather than a skip.
    HANA_AGENT_E2E_ARTIFACT_DIR: optional directory for actions and SQL evidence.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from datus.cli.repl import DatusCLI
from datus.storage.schema_metadata import create_metadata_rag
from datus.tools.db_tools import db_manager
from datus.utils.path_manager import get_path_manager, reset_path_manager, set_current_path_manager
from tests.integration.conftest import wait_for_agent

pytestmark = [pytest.mark.nightly, pytest.mark.product_e2e, pytest.mark.timeout(600)]


def _unavailable(reason):
    if os.getenv("HANA_REQUIRE_AGENT_E2E") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture
def hana_chat_environment(tmp_path, monkeypatch):
    if not os.getenv("HANA_PASSWORD") or not os.getenv("HANA_HOST"):
        _unavailable("HANA_HOST and HANA_PASSWORD are required for HANA Agent E2E")
    try:
        import datus_hana  # noqa: F401 - verify the installed entry-point package
        from hdbcli import dbapi
    except ImportError:
        _unavailable("Install datus-hana and its official hdbcli dependency")

    source = Path(os.getenv("HANA_AGENT_CONFIG", "tests/conf/agent.yml"))
    source_config = yaml.safe_load(source.read_text())["agent"]
    target = os.getenv("HANA_AGENT_MODEL", source_config["target"])
    model = source_config["models"][target]
    for variable in re.findall(r"\$\{([A-Z_][A-Z_0-9]*)\}", str(model)):
        if not os.getenv(variable):
            _unavailable(f"Model environment variable {variable} is required")

    schema = os.getenv("HANA_SCHEMA", os.environ.get("HANA_USER", "DATUS_TEST"))
    prefix = "E2E_" + uuid.uuid4().hex[:10].upper()
    customers, orders = prefix + "_CUSTOMERS", prefix + "_ORDERS"

    def quote(name):
        return '"' + name.replace('"', '""') + '"'

    customer_ref, order_ref = (f"{quote(schema)}.{quote(name)}" for name in (customers, orders))
    connection_args = {
        "address": os.environ["HANA_HOST"],
        "port": int(os.getenv("HANA_PORT", "23941")),
        "user": os.getenv("HANA_USER", "DATUS_TEST"),
        "password": os.environ["HANA_PASSWORD"],
        "encrypt": os.getenv("HANA_ENCRYPT", "true").lower() == "true",
        "sslValidateCertificate": True,
        "connectTimeout": 5000,
    }
    for env_name, sdk_name in (
        ("HANA_SSL_TRUST_STORE", "sslTrustStore"),
        ("HANA_SSL_HOST_NAME_IN_CERTIFICATE", "sslHostNameInCertificate"),
        ("HANA_DATABASE", "databaseName"),
    ):
        if os.getenv(env_name):
            connection_args[sdk_name] = os.environ[env_name]
    try:
        connection = dbapi.connect(**connection_args)
    except dbapi.Error as error:
        # Do not include a driver exception that could contain connection secrets.
        _unavailable(f"HANA connection unavailable ({type(error).__name__})")

    created = []
    path_token = set_current_path_manager(get_path_manager())

    def query(sql):
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            return cursor.fetchall()
        finally:
            cursor.close()

    try:
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"CREATE COLUMN TABLE {customer_ref} "
                "(CUSTOMER_ID INTEGER PRIMARY KEY, CUSTOMER_NAME NVARCHAR(80), REGION NVARCHAR(16))"
            )
            created.append(customer_ref)
            cursor.execute(
                f"CREATE COLUMN TABLE {order_ref} "
                "(ORDER_ID INTEGER PRIMARY KEY, CUSTOMER_ID INTEGER, AMOUNT DECIMAL(12,2), STATUS NVARCHAR(16))"
            )
            created.append(order_ref)
            cursor.executemany(
                f"INSERT INTO {customer_ref} VALUES (?, ?, ?)",
                [(1, "北辰商贸", "East"), (2, "海星科技", "East"), (3, "西山科技", "West")],
            )
            cursor.executemany(
                f"INSERT INTO {order_ref} VALUES (?, ?, ?, ?)",
                [
                    (101, 1, Decimal("128.50"), "PAID"),
                    (102, 1, Decimal("17.25"), "PAID"),
                    (103, 2, Decimal("203.40"), "PAID"),
                    (104, 2, Decimal("61.10"), "CANCELLED"),
                    (105, 3, Decimal("84.75"), "PAID"),
                    (106, 3, Decimal("9.50"), "PAID"),
                ],
            )
            connection.commit()
        finally:
            cursor.close()

        # Oracle SQL and its results never enter the model's config or prompt.
        oracle = dict(
            query(
                f"SELECT c.REGION, SUM(o.AMOUNT) FROM {customer_ref} c "
                f"JOIN {order_ref} o ON c.CUSTOMER_ID = o.CUSTOMER_ID "
                "WHERE o.STATUS = 'PAID' GROUP BY c.REGION"
            )
        )
        assert oracle == {"East": Decimal("349.15"), "West": Decimal("94.25")}
        home, project = tmp_path / "home", tmp_path / "workspace"
        project.mkdir()
        config = {
            "agent": {
                "home": str(home),
                "project_root": str(project),
                "project_name": prefix.lower(),
                "target": target,
                "models": {target: model},
                "sql_read_only": True,
                "permissions": {"default": "allow"},
                "observability": {"tracing": {"enabled": False}},
                "agentic_nodes": {"chat": {"model": target, "max_turns": 15, "tools": "db_tools.*"}},
                "services": {
                    "datasources": {
                        "hana_e2e": {
                            "type": "hana",
                            "host": os.environ["HANA_HOST"],
                            "port": connection_args["port"],
                            "username": connection_args["user"],
                            "password": "${HANA_PASSWORD}",
                            "schema": schema,
                            "database": os.getenv("HANA_DATABASE", ""),
                            "encrypt": connection_args["encrypt"],
                            "ssl_trust_store": os.getenv("HANA_SSL_TRUST_STORE") or None,
                            "ssl_host_name_in_certificate": os.getenv("HANA_SSL_HOST_NAME_IN_CERTIFICATE") or None,
                        }
                    }
                },
            }
        }
        config_path = tmp_path / "agent.yml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
        config_path.chmod(0o600)
        monkeypatch.setenv("DATUS_TUI", "0")
        db_manager._cli_cache.clear()
        yield SimpleNamespace(
            args=Namespace(
                config=str(config_path),
                datasource="hana_e2e",
                schema=schema,
                database="",
                history_file=str(home / "history"),
                debug=False,
                storage_path=str(home / "data"),
                home=str(home),
                project_root=str(project),
                reload=True,
                force=True,
                yes=True,
            ),
            schema=schema,
            customers=customers,
            orders=orders,
            oracle=oracle,
            query=query,
        )
    finally:
        db_manager._cli_cache.clear()
        reset_path_manager(path_token)
        try:
            for table in reversed(created):
                cursor = connection.cursor()
                try:
                    cursor.execute(f"DROP TABLE {table}")
                finally:
                    cursor.close()
            connection.commit()
        finally:
            connection.close()


def _run_chat(environment, questions, case_name):
    cli = None
    try:
        with (
            patch("datus.cli.repl.PromptSession.prompt", side_effect=[*questions, EOFError]),
            patch("datus.cli.repl.DatusCLI.prompt_input", return_value="n"),
            patch("datus.cli.repl.AtReferenceCompleter.parse_at_context", return_value=([], [], [], None)),
        ):
            cli = DatusCLI(args=environment.args)
            wait_for_agent(cli)
            cli.run()
        # Exercise the normal startup index as well as direct metadata tools.
        # Missing database namespaces must not silently produce an empty index.
        pending_sync = cli.bg_sync._current_future
        if pending_sync is not None:
            pending_sync.result(timeout=60)
        metadata = create_metadata_rag(cli.agent_config).search_all_schemas(
            schema_name=environment.schema,
            table_type="table",
            select_fields=["database_name", "schema_name", "table_name"],
        )
        indexed = {row["table_name"]: row for row in metadata.to_pylist()}
        for table in (environment.customers, environment.orders):
            assert indexed[table] == {
                "database_name": "",
                "schema_name": environment.schema,
                "table_name": table,
            }
        actions = cli.actions.get_actions()
        responses = [a for a in actions if a.action_type == "chat_response"]
        assert len(responses) == len(questions), f"Expected {len(questions)} responses; got {len(responses)}"
        assert all(a.output.get("success") is True for a in responses)
        return responses
    finally:
        if cli is not None:
            actions = cli.actions.get_actions()
            artifact_dir = os.getenv("HANA_AGENT_E2E_ARTIFACT_DIR")
            if artifact_dir:
                path = Path(artifact_dir)
                path.mkdir(parents=True, exist_ok=True)
                text = json.dumps([a.model_dump(mode="json") for a in actions], ensure_ascii=False, indent=2)
                text = text.replace(os.environ["HANA_PASSWORD"], "<redacted>")
                (path / f"{case_name}-actions.json").write_text(text)
            cli.bg_sync.shutdown()
            cli._bg_loop.call_soon_threadsafe(cli._bg_loop.stop)
            cli._bg_loop_thread.join(timeout=5)


def _successful_queries(response):
    queries = []
    for action in response.output.get("action_history", []):
        if action.get("action_type") not in {"execute_sql", "read_query"}:
            continue
        if action.get("status") != "success" or not (action.get("output") or {}).get("success"):
            continue
        arguments = (action.get("input") or {}).get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if arguments.get("sql"):
            queries.append(arguments["sql"])
    assert queries, "The real model must successfully execute a database query"
    return queries


def _assert_answer_amount(response, amount):
    text = str(response.output.get("response", "")).replace(",", "")
    numbers = {Decimal(n) for n in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text)}
    assert amount in numbers, f"Answer must contain the independently verified amount {amount}: {text}"


def test_hana_chat_join_aggregate(hana_chat_environment):
    """Discover metadata, generate a JOIN aggregate, execute it and explain correct totals."""
    env = hana_chat_environment
    question = (
        f"请先列出当前 schema {env.schema} 中的表，找到客户和订单表并查看它们的表结构。"
        "然后根据客户所属 REGION，统计 STATUS 为 PAID 的订单金额合计，按合计从高到低排列。"
        "请实际查询数据库，列出每个区域的金额并说明哪个区域最高。"
    )
    response = _run_chat(env, [question], "join-aggregate")[0]
    queries = _successful_queries(response)
    tool_names = response.output.get("execution_stats", {}).get("tools_used", [])
    assert "list_tables" in tool_names, f"Agent must discover the real tables: {tool_names}"
    assert "describe_table" in tool_names, f"Agent must discover real column metadata: {tool_names}"
    results = [env.query(sql) for sql in queries]
    expected = [(region, amount) for region, amount in sorted(env.oracle.items(), key=lambda item: -item[1])]
    # Extra useful measures (for example order count) and projection order are
    # model choices; verify the requested regions, amounts and sorting instead.
    assert any(
        len(rows) == len(expected)
        and all(region in row and amount in row for row, (region, amount) in zip(rows, expected))
        for rows in results
    ), f"Generated SQL must match the independent oracle: {results}"
    for amount in env.oracle.values():
        _assert_answer_amount(response, amount)


def test_hana_chat_multi_turn(hana_chat_environment):
    """A follow-up keeps the PAID filter while changing region; Chinese names round-trip."""
    env = hana_chat_environment
    questions = [
        f"请查询 {env.schema} 中的 {env.customers} 和 {env.orders}，"
        "计算 East 区域 STATUS 为 PAID 的订单总金额。请实际执行 SQL 并回答。",
        "沿用刚才的订单状态条件，改查 West 区域的总金额，并查出该区域客户的中文名称。请实际查询数据库。",
    ]
    responses = _run_chat(env, questions, "multi-turn")
    for region, response in zip(("East", "West"), responses):
        queries = _successful_queries(response)
        rows = [row for sql in queries for row in env.query(sql)]
        values = [value for row in rows for value in row]
        assert env.oracle[region] in values, f"Generated SQL must preserve region and PAID filters: {rows}"
        _assert_answer_amount(response, env.oracle[region])
    assert "西山科技" in responses[-1].output["response"]
