# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Opt-in real-Qwen /init tests on empty synthetic SQLite projects.

Run with DATUS_RUN_LIVE_INIT=1 and DASHSCOPE_API_KEY (or QWEN_API_KEY).
No LLM, tool, node, storage or database implementation is mocked. This tests
the current scope, a continuous chat session, live metadata and supplemental business
knowledge across separate calls. DATUS_RUN_LIVE_INIT_SCALE=1 additionally enables a 20,000-table
catalog with ten selected tables. Neither fixture contains customer data.
"""

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pytest

from datus.cli.chat_commands import ChatCommands
from datus.cli.init_commands import InitCommands
from datus.configuration.agent_config import AgentConfig
from datus.utils.path_manager import reset_path_manager
from tests.unit_tests.cli.test_chat_commands import MinimalCLI


def observe_live_calls(monkeypatch):
    """Collect metadata while preserving every real model call and response."""
    from datus.observability.model_call import ModelCall

    model_calls = []
    finish_call = ModelCall.finish

    def record_call(call, error=None):
        already_finished = call._finished
        try:
            return finish_call(call, error)
        finally:
            if not already_finished:
                fields = call.fields
                model_calls.append(
                    {
                        k: v
                        for k, v in fields.items()
                        if k.startswith("capacity_")
                        or k
                        in {
                            "model_call_id",
                            "model",
                            "status",
                            "duration_ms",
                            "input_tokens",
                            "output_tokens",
                            "error_type",
                        }
                    }
                )

    monkeypatch.setattr(ModelCall, "finish", record_call)
    return model_calls


def observe_catalog_io(monkeypatch):
    """Time real catalog reads and result archival without replacing their behavior."""
    from datus.tools.db_tools.sqlite_connector import SQLiteConnector
    from datus.utils.tool_result_store import ToolResultStore

    measurements = {"catalog_queries": [], "archive_writes": []}
    get_tables, save = SQLiteConnector.get_tables, ToolResultStore._save

    def read_catalog(connector, *args, **kwargs):
        started = time.monotonic()
        result = get_tables(connector, *args, **kwargs)
        measurements["catalog_queries"].append(
            {"table_count": len(result), "seconds": round(time.monotonic() - started, 6)}
        )
        return result

    def save_result(store, result_id, wire, payload):
        started = time.monotonic()
        result = save(store, result_id, wire, payload)
        elapsed = time.monotonic() - started
        body = payload.get("result") if isinstance(payload, dict) else payload
        measurements["archive_writes"].append(
            {
                "result_id": result_id,
                "bytes": len(wire.encode("utf-8")),
                "items": len(body) if isinstance(body, list) else None,
                "seconds": round(elapsed, 6),
            }
        )
        return result

    monkeypatch.setattr(SQLiteConnector, "get_tables", read_catalog)
    monkeypatch.setattr(ToolResultStore, "_save", save_result)
    return measurements


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.timeout(900)
@pytest.mark.skipif(os.environ.get("DATUS_RUN_LIVE_INIT") != "1", reason="Explicit live init opt-in required")
def test_qwen_scoped_init_supplements_existing_documents(tmp_path, monkeypatch, request):
    _run_qwen_init(tmp_path, monkeypatch, request)


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.timeout(1200)
@pytest.mark.skipif(os.environ.get("DATUS_RUN_LIVE_INIT_SCALE") != "1", reason="Explicit live scale opt-in required")
def test_qwen_init_large_catalog_in_one_loop(tmp_path, monkeypatch, request):
    _run_qwen_init(tmp_path, monkeypatch, request, scale=True)


def _run_qwen_init(tmp_path, monkeypatch, request, *, scale=False):
    key_name = next((name for name in ("DASHSCOPE_API_KEY", "QWEN_API_KEY") if os.environ.get(name)), None)
    if key_name is None:
        pytest.skip("Set DASHSCOPE_API_KEY or QWEN_API_KEY")
    # Tests are isolated from the user's workspace and external tracing.
    monkeypatch.chdir(tmp_path)
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    reset_path_manager()
    from agents import set_tracing_disabled

    set_tracing_disabled(True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "fixture.sqlite"
    order_tables = ["orders", *([f"orders_{i:02d}" for i in range(1, 9)] if scale else [])]
    core_tables = [*order_tables, "customers"]
    catalog_size = 20000 if scale else 3
    fixture_started = time.monotonic()
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), "
            "amount NUMERIC, status TEXT, created_at TEXT);"
            "CREATE TABLE unrelated_events(id INTEGER PRIMARY KEY, payload TEXT);"
        )
        for table in order_tables[1:]:
            wide_columns = "," + ",".join(f"attribute_{i:03d} TEXT" for i in range(501)) if table == "orders_08" else ""
            connection.execute(
                f"CREATE TABLE {table}(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), "
                f"amount NUMERIC, status TEXT, created_at TEXT{wide_columns})"
            )
        for i in range(catalog_size - len(core_tables) - 1):
            connection.execute(
                f"CREATE TABLE tenant_{i:05d}_archive_ledger_event_snapshots"
                "(id INTEGER PRIMARY KEY, tenant_id INTEGER, entity_code TEXT, amount NUMERIC, "
                "status TEXT, created_at TEXT, updated_at TEXT, source_system TEXT)"
            )
    (workspace / "README.md").write_text(
        "# Synthetic order reporting test project\n"
        "This is a constructed empty database, not customer data.\n"
        "orders records booked customer purchases for sales reporting.\n"
        "orders grain: one row per order according to the documented contract.\n"
        "customers is the customer directory used to label orders.\n"
        "customers grain: one row per customer according to the documented contract.\n"
        "All business timestamps use UTC.\n"
        "The booked revenue metric is defined by metrics.sql; its alias is order_revenue.\n"
        f"Only {', '.join(order_tables)} (sales domain) and customers (crm domain) are core scope.\n"
        "The unrelated_events table is outside this init scope.\n"
        + "".join(
            f"{table} records booked customer purchases for sales reporting.\n"
            f"{table} grain: one row per order according to the documented contract.\n"
            for table in order_tables[1:]
        )
    )
    (workspace / "metrics.sql").write_text(
        "SELECT SUM(o.amount) AS order_revenue FROM orders o WHERE o.status = 'booked';\n"
        "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id;\n"
        + "".join(
            f"SELECT SUM(o.amount) AS order_revenue_{table[-2:]} FROM {table} o WHERE o.status = 'booked';\n"
            for table in order_tables[1:]
        )
    )
    (workspace / "AGENTS.md").write_text("# User instructions\nKeep this user-owned instruction verbatim.\n")
    model = os.environ.get("DATUS_LIVE_INIT_MODEL", "qwen3.5-27b")
    config = AgentConfig(
        nodes={},
        home=str(tmp_path / "home"),
        project_name="synthetic-init-live",
        project_root=str(workspace),
        target="live",
        models={
            "live": {
                "type": "qwen",
                "model": model,
                "api_key": "${" + key_name + "}",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "context_length": 262144,
                "max_tokens": 8192,
                "tokenizer_ref": "Qwen/Qwen3.5-27B",
                "enable_thinking": False,
                "max_retry": 1,
                "temperature": 0,
            }
        },
        services={
            "datasources": {
                "fixture": {
                    "type": "sqlite",
                    "uri": str(database),
                    "name": "fixture",
                    "default": True,
                    "read_only": True,
                }
            }
        },
        agentic_nodes={
            "chat": {
                "model": "live",
                "system_prompt": "chat",
                "workspace_root": str(workspace),
                "tools": "db_tools.*,filesystem_tools.*,skills.*",
                "max_turns": 20 if scale else 15,
            }
        },
        observability={"tracing": {"enabled": False}},
    )
    from datus.tools.permission.profiles import get_profile

    config.permissions_config = get_profile("auto")
    config.current_datasource = "fixture"
    from datus.utils.loggings import configure_logging

    configure_logging(
        level="WARNING",
        console_output=False,
        log_dir=str(tmp_path / "logs"),
        agent_config=config,
        redact={"enabled": True, "patterns": [re.escape(os.environ[key_name])]},
    )
    model_calls = observe_live_calls(monkeypatch)
    catalog_io = observe_catalog_io(monkeypatch)
    runner = asyncio.Runner()

    def cleanup():
        from datus.storage.registry import clear_storage_registry
        from datus.tools.db_tools.db_manager import set_db_manager_factory

        runner.close()
        set_db_manager_factory(None)
        clear_storage_registry()
        reset_path_manager()

    request.addfinalizer(cleanup)

    class PersistentCLI(MinimalCLI):
        """Match the real CLI's persistent background loop across ordinary chat turns."""

        def run_on_bg_loop(self, coro, **_kwargs):
            return runner.run(coro)

    cli = PersistentCLI(config)
    chat = ChatCommands(cli)
    cli.chat_commands = chat
    original_factory, nodes = chat._create_new_node, []

    def record_node(*args, **kwargs):
        node = original_factory(*args, **kwargs)
        nodes.append(node)
        return node

    chat._create_new_node = record_node
    contexts = []
    original_execute = chat.execute_chat_command

    def capture_context(*args, **kwargs):
        if kwargs.get("init_context") is not None:
            contexts.append(kwargs["init_context"])
        return original_execute(*args, **kwargs)

    chat.execute_chat_command = capture_context
    started = time.monotonic()
    InitCommands(cli).cmd_init(
        (
            "First call list_tables with include_views=false once to confirm the catalog. "
            "Do not paginate the unrelated tables. Initialize all of " + ", ".join(order_tables) + " in sales "
            "and customers in crm. Check the structure of every named table and document relevant differences. "
            if scale
            else "Only orders in sales and customers in crm. "
        )
        + "Read README.md and metrics.sql. "
        "Document useful order and customer guidance, including order_revenue and its booked filter. "
        "Summarize with source references and note important unknowns. Do not inspect unrelated_events."
    )
    context = contexts[0]
    state = context.state
    archives = []
    for node in nodes:
        store = getattr(node, "_tool_result_store", None)
        if store is not None:
            for path in store.root.glob("*.json"):
                metadata = json.loads(path.read_text())
                raw = path.with_suffix(".raw")
                archives.append(
                    {
                        "session": node.session_id,
                        "sections": metadata["sections"],
                        "bytes": raw.stat().st_size,
                        "tokens": node.model.request_capacity().count({"input": raw.read_text()}),
                    }
                )
    report = {
        "model": model,
        "context_length": 262144,
        "max_turns": config.agentic_nodes["chat"]["max_turns"],
        "dataset": f"constructed empty SQLite; {catalog_size} tables, {len(core_tables)} selected",
        "fixture_seconds": round(started - fixture_started, 2),
        "database_bytes": database.stat().st_size,
        "wide_table_columns": 506 if scale else None,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "status": state["status"],
        "selected": list(state["selected_tables"]),
        "sessions": [n.session_id for n in nodes],
        "outputs": list(state["outputs"]),
        "stop_reason": state.get("stop_reason"),
        "progress": context.progress(),
        "catalog_io": catalog_io,
        "model_calls": model_calls,
        "archives": archives,
        "workspace": str(workspace),
        "node_results": [
            {
                "success": getattr(n.result, "success", None),
                "error": str(getattr(n.result, "error", ""))[:1000],
                "response": str(getattr(n.result, "response", ""))[:1000],
            }
            for n in nodes
        ],
        "actions": [
            {"type": a.action_type, "status": str(a.status), "output": str(a.output)[:1200]}
            for n in nodes
            for a in n.actions
            if "error" in str(a.status).lower() or "fail" in str(a.status).lower()
        ],
    }
    report_path = Path(os.environ.get("DATUS_LIVE_INIT_REPORT", str(tmp_path / "report.json")))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    for name in ("DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        if os.environ.get(name):
            rendered = rendered.replace(os.environ[name], "[REDACTED]")
    report_path.write_text(rendered)
    assert state["status"] == "completed_scope", f"Incomplete init: {report}; console: {cli.console.file.getvalue()}"
    assert set(state["selected_tables"]) == set(core_tables)
    if scale:
        assert all(state["tables"][table].get("metadata_status") == "checked" for table in core_tables), (
            "The requested per-table structure checks must actually run"
        )
    assert len(nodes) == 1 and chat.current_node is nodes[0]
    if scale:
        # Knowledge organization is model-owned; assess published content even
        # when related domains share a document. Pending questions are not facts.
        knowledge_files = [
            path for path in (workspace / "knowledge").rglob("*.md") if path.name != "pending-questions.md"
        ]
        assert knowledge_files, "Publish useful knowledge, not only pending questions"
        published = "\n".join(path.read_text() for path in [workspace / "AGENTS.md", *knowledge_files])
        sales = crm = published
    else:
        sales = (workspace / "knowledge/sales.md").read_text()
        crm = (workspace / "knowledge/crm.md").read_text()
    # The knowledge may express the same definition as SQL or concise prose.
    metric_pattern = r"\bsum(?:\s*\(\s*(?:[a-z_]+\.)?amount\s*\)|\s+of\s+(?:[a-z_]+\.)?amount\b)"
    assert re.search(metric_pattern, sales.replace("`", ""), re.I), "The revenue formula must be documented"
    assert "booked" in sales.lower() and "order_revenue" in sales
    assert "README.md" in sales and "metrics.sql" in sales
    assert "customers" in crm.lower() and "README.md" in crm
    if scale:
        assert any(a["sections"].get("items") == catalog_size for a in archives), "Preserve the full table catalog"
        assert any(a["tokens"] > 262144 for a in archives), "The raw catalog must exceed the configured window"
        checked_calls = [c for c in model_calls if "capacity_input_tokens" in c]
        assert checked_calls, "Capture actual request capacity evidence"
        assert all(c["capacity_input_tokens"] <= c["capacity_input_limit"] for c in checked_calls)
        assert not any(c.get("error_type") for c in model_calls), "No provider or local model-call errors"
    assert "Keep this user-owned instruction verbatim." in (workspace / "AGENTS.md").read_text()
    assert len((workspace / "AGENTS.md").read_text().splitlines()) <= 200
    transcript = runner.run(nodes[0]._session.get_items())
    tool_names = {item.get("name") for item in transcript if item.get("type") == "function_call"}
    assert not {"execute_sql", "bash", "task", "confirm_plan"} & tool_names
    assert not (workspace / ".datus/init").exists()
    if not scale:
        before = len(nodes)
        rule = "Booked orders require settled payment before invoicing."
        (workspace / "invoicing.md").write_text(rule + "\n")
        InitCommands(cli).cmd_init(
            "Only orders in sales. Read existing knowledge/sales.md for context. "
            "Add the invoicing rule from invoicing.md, summarize it clearly and cite the source. "
            "Preserve existing purpose, metrics and SQL filters. "
            "Do not inspect customers or unrelated_events."
        )
        second = contexts[-1]
        report["second_scope"] = {
            "status": second.state["status"],
            "tables": list(second.state["selected_tables"]),
            "new_sessions": len(nodes) - before,
            "outputs": list(second.state["outputs"]),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        assert second is not context and second.state["status"] == "completed_scope", report["second_scope"]
        assert set(second.state["selected_tables"]) == {"orders"}
        assert len(nodes) == before and chat.current_node is nodes[0]
        text = (workspace / "knowledge/sales.md").read_text()
        assert "settled" in text.lower() and "invoic" in text.lower() and "invoicing.md" in text
        assert re.search(metric_pattern, text.replace("`", ""), re.I) and "booked" in text.lower()
        assert (workspace / "knowledge/crm.md").read_text() == crm
        assert "knowledge/crm" in (workspace / "AGENTS.md").read_text()
        assert not (workspace / ".datus/init").exists()
    assert chat.current_node is nodes[0]
    assert getattr(chat.current_node, "_init_context", None) is None
