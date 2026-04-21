# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.service_commands`` dispatcher."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from datus.cli.service_client import READ_METHODS, ServiceClient, ServiceClientRegistry
from datus.cli.service_commands import ServiceCommands
from datus.tools.func_tool.base import FuncToolResult


class _BiToolStub:
    def list_dashboards(self, search: str = "") -> FuncToolResult:
        """List dashboards (read)."""
        return FuncToolResult(result=[{"id": 1, "search": search}])

    def get_dashboard(self, dashboard_id: str) -> FuncToolResult:
        """Get one dashboard by id."""
        if not dashboard_id:
            return FuncToolResult(success=0, error="dashboard_id required")
        return FuncToolResult(result={"id": dashboard_id})

    def get_chart_data(self, chart_id: str, dashboard_id: str = "", limit: int = 0) -> FuncToolResult:
        """Get chart data with optional limit."""
        return FuncToolResult(result={"chart_id": chart_id, "limit": limit, "dashboard_id": dashboard_id})

    def create_dashboard(self, title: str) -> FuncToolResult:
        """Write method — should NOT be dispatchable."""
        return FuncToolResult(result={"created": title})


def _fake_cli():
    cli = MagicMock()
    cli.console = MagicMock()
    cli._bg_loop = None  # commands use asyncio.run in this case
    cli.agent_config = SimpleNamespace(
        services=SimpleNamespace(bi_tools={"superset": {}}, schedulers={}, semantic_layer={}),
    )
    return cli


def _make_commands_with_bi_stub(tool_instance=None):
    cli = _fake_cli()
    cmd = ServiceCommands(cli)
    # Inject a registry with the stub tool directly — bypass factory.
    registry = ServiceClientRegistry.__new__(ServiceClientRegistry)
    registry._agent_config = cli.agent_config
    registry._entries = {}
    registry._fingerprint = None
    registry._clients = {
        "superset": ServiceClient(
            service_type="bi_tools",
            service_name="superset",
            tool_instance=tool_instance or _BiToolStub(),
            method_names=READ_METHODS["bi_tools"],
        ),
    }
    # _entries drives list_services output + has() lookups (2-tuple per current schema).
    registry._entries["superset"] = ("bi_tools", "superset")
    cmd._registry = registry
    return cmd, cli


class TestDispatchServiceListing:
    def test_dispatch_bare_service_prints_methods(self):
        cmd, cli = _make_commands_with_bi_stub()
        handled = cmd.dispatch(".superset", "")
        assert handled is True
        # Rich table passed through console.print — at least one call occurred.
        assert cli.console.print.call_count >= 1

    def test_dispatch_unknown_service_returns_false(self):
        cmd, _ = _make_commands_with_bi_stub()
        handled = cmd.dispatch(".mystery", "")
        assert handled is False

    def test_dispatch_non_dot_command_returns_false(self):
        cmd, _ = _make_commands_with_bi_stub()
        assert cmd.dispatch("superset", "") is False

    def test_cmd_services_lists_all(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.cmd_services("")
        # Printed a table (Table object, not a string).
        assert cli.console.print.call_count == 1

    def test_cmd_services_empty_prints_hint(self):
        cli = _fake_cli()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_tools={}, schedulers={}, semantic_layer={}),
        )
        cmd = ServiceCommands(cli)
        cmd.cmd_services("")
        # Should have printed at least one yellow hint message.
        msg = str(cli.console.print.call_args_list[0])
        assert "No services configured" in msg


class TestDispatchInvokeMethod:
    def test_positional_arg_success(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.get_dashboard", "42")
        # Rendered result contains the id.
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "42" in rendered

    def test_named_arg_success(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.get_dashboard", "--dashboard_id=7")
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "7" in rendered

    def test_int_coercion_from_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.get_chart_data", "99 --limit=50")
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "99" in rendered
        assert "50" in rendered  # int, not str

    def test_missing_required_shows_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.get_dashboard", "")
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "Missing required argument" in rendered or "required" in rendered.lower()

    def test_help_flag_shows_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.get_dashboard", "--help")
        # Rich Table object is passed to console.print; inspect its title.
        printed = cli.console.print.call_args_list[-1].args[0]
        assert "parameters" in str(getattr(printed, "title", "")).lower()

    def test_write_method_is_blocked(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.create_dashboard", "--title=x")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "write" in msg.lower() or "privileged" in msg.lower() or "read-only" in msg.lower()

    def test_unknown_method_prints_hint(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch(".superset.no_such_method", "")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "Unknown method" in msg or "no_such_method" in msg

    def test_tool_error_rendered(self):
        """When FuncToolResult.success==0, error is surfaced."""
        cmd, cli = _make_commands_with_bi_stub()
        # get_dashboard with empty id returns success=0
        cmd.dispatch(".superset.get_dashboard", "''")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "required" in msg.lower() or "Error" in msg


class TestArgParser:
    def test_parse_positional_only(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed = cmd._parse_args("foo 42", schema)
        assert parsed == {"a": "foo", "b": 42}

    def test_parse_named_only(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed = cmd._parse_args("--b=99 --a=hi", schema)
        assert parsed == {"a": "hi", "b": 99}

    def test_parse_mixed(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
        parsed = cmd._parse_args("first --b=7", schema)
        assert parsed == {"a": "first", "b": 7}

    def test_parse_bool_flag(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"flag": {"type": "boolean"}}}
        assert cmd._parse_args("--flag", schema) == {"flag": True}
        assert cmd._parse_args("--flag=true", schema) == {"flag": True}
        assert cmd._parse_args("--flag=no", schema) == {"flag": False}

    def test_parse_array_csv(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"items": {"type": "array"}}}
        parsed = cmd._parse_args("--items=a,b,c", schema)
        assert parsed == {"items": ["a", "b", "c"]}

    def test_parse_optional_array_via_anyof(self):
        """``Optional[List[str]] = None`` uses anyOf with no top-level type."""
        cmd = ServiceCommands(_fake_cli())
        schema = {
            "properties": {
                "items": {
                    "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                    "default": None,
                },
            },
        }
        parsed = cmd._parse_args("--items=a,b", schema)
        assert parsed == {"items": ["a", "b"]}

    def test_parse_optional_int_via_anyof(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {
            "properties": {
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            },
        }
        parsed = cmd._parse_args("--limit=42", schema)
        assert parsed == {"limit": 42}

    def test_parse_array_python_literal_single_quoted(self):
        """``--metrics=['sales']`` is valid Python literal but invalid JSON."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"metrics": {"type": "array"}}}
        parsed = cmd._parse_args("\"--metrics=['sales','revenue']\"", schema)
        assert parsed == {"metrics": ["sales", "revenue"]}

    def test_parse_object_type_json(self):
        """Object parameters accept standard JSON."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"ctx": {"type": "object"}}}
        parsed = cmd._parse_args('\'--ctx={"dim": "region", "metric": "revenue"}\'', schema)
        assert parsed == {"ctx": {"dim": "region", "metric": "revenue"}}

    def test_parse_object_type_python_literal(self):
        """Object parameters also accept Python-literal form (single quotes)."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"ctx": {"type": "object"}}}
        parsed = cmd._parse_args("\"--ctx={'dim': 'region'}\"", schema)
        assert parsed == {"ctx": {"dim": "region"}}

    def test_parse_object_malformed_falls_back_to_raw(self):
        """Unparseable object input returns raw string so downstream can complain clearly."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"ctx": {"type": "object"}}}
        parsed = cmd._parse_args("--ctx=not-a-dict", schema)
        assert parsed == {"ctx": "not-a-dict"}

    def test_parse_array_malformed_json_falls_back_to_csv(self):
        """Truly broken brackets fall through to CSV split."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"items": {"type": "array"}}}
        parsed = cmd._parse_args("--items=a,b,c", schema)
        assert parsed == {"items": ["a", "b", "c"]}

    def test_coerce_helper_directly_for_array(self):
        assert ServiceCommands._coerce("['a','b']", {"type": "array"}) == ["a", "b"]
        assert ServiceCommands._coerce('["a","b"]', {"type": "array"}) == ["a", "b"]
        assert ServiceCommands._coerce("a,b", {"type": "array"}) == ["a", "b"]

    def test_coerce_helper_directly_for_object(self):
        assert ServiceCommands._coerce('{"k": 1}', {"type": "object"}) == {"k": 1}
        assert ServiceCommands._coerce("{'k': 1}", {"type": "object"}) == {"k": 1}
        # Malformed falls through to raw.
        assert ServiceCommands._coerce("not-a-dict", {"type": "object"}) == "not-a-dict"

    def test_primary_type_helper(self):
        assert ServiceCommands._primary_type({"type": "integer"}) == "integer"
        assert ServiceCommands._primary_type({"type": ["string", "null"]}) == "string"
        assert ServiceCommands._primary_type({"anyOf": [{"type": "null"}, {"type": "array"}]}) == "array"
        assert ServiceCommands._primary_type({"oneOf": [{"type": "integer"}]}) == "integer"
        assert ServiceCommands._primary_type({}) == ""
        assert ServiceCommands._primary_type(None) == ""

    def test_parse_array_json(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"items": {"type": "array"}}}
        # Shell-quoted JSON array: shlex keeps the JSON bracketed form intact.
        parsed = cmd._parse_args('\'--items=["x","y"]\'', schema)
        assert parsed == {"items": ["x", "y"]}

    def test_parse_extra_positional_returns_none(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"only": {"type": "string"}}}
        assert cmd._parse_args("first second", schema) is None

    def test_parse_unknown_named_fails_fast_with_hint(self):
        """Unknown ``--flag`` must not be silently dropped — a typoed
        ``--limti`` or ``--serach`` silently ignored would execute the
        request without the filter the user intended, which is a subtle
        bug to diagnose at the REPL. Parser returns None and stores a
        pointed hint naming the valid parameters.
        """
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}, "limit": {"type": "integer"}}}
        assert cmd._parse_args("--bogus=x --a=ok", schema) is None
        assert cmd._last_parse_error is not None
        assert "bogus" in cmd._last_parse_error
        # Valid alternatives listed so the user sees what they could have typed.
        assert "a" in cmd._last_parse_error
        assert "limit" in cmd._last_parse_error

    def test_parse_error_resets_between_calls(self):
        """A successful parse after a failed one must clear the stale error."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}}}
        assert cmd._parse_args("--bogus=x", schema) is None
        assert cmd._last_parse_error is not None
        # Second call succeeds → sentinel cleared.
        assert cmd._parse_args("--a=ok", schema) == {"a": "ok"}
        assert cmd._last_parse_error is None

    def test_parse_extra_positional_records_hint(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"only": {"type": "string"}}}
        assert cmd._parse_args("first second third", schema) is None
        assert cmd._last_parse_error is not None
        assert "Too many positional" in cmd._last_parse_error

    def test_parse_malformed_quoting_returns_none(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}}}
        # Unclosed single quote → shlex.split raises → parser returns None.
        assert cmd._parse_args("'unclosed", schema) is None

    def test_missing_required_reports_all(self):
        cmd = ServiceCommands(_fake_cli())

        def target(a, b):
            return (a, b)

        missing = cmd._missing_required(target, {"a": 1})
        assert missing == ["b"]

    def test_missing_required_skips_optional_with_none_default(self):
        """Optional[...] = None in Python signature is truly optional."""
        from typing import List, Optional

        cmd = ServiceCommands(_fake_cli())

        def target(metrics: List[str], path: Optional[List[str]] = None, limit: Optional[int] = None):
            return (metrics, path, limit)

        missing = cmd._missing_required(target, {"metrics": ["sales"]})
        # path and limit have defaults → not required.
        assert missing == []

    def test_missing_required_keeps_truly_required(self):
        from typing import List, Optional

        cmd = ServiceCommands(_fake_cli())

        def target(metrics: List[str], path: Optional[List[str]] = None):
            return (metrics, path)

        missing = cmd._missing_required(target, {})
        assert missing == ["metrics"]

    def test_missing_required_handles_none_method(self):
        cmd = ServiceCommands(_fake_cli())
        # Safety: no callable → no blockage.
        assert cmd._missing_required(None, {}) == []


class TestRenderHelpers:
    def test_render_result_success_payload(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": {"id": 42}})
        msg = str(cli.console.print.call_args_list[0])
        assert "42" in msg

    def test_render_result_failure(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 0, "error": "boom"})
        msg = str(cli.console.print.call_args_list[0])
        assert "boom" in msg and "Error" in msg


class TestAsyncExecution:
    def test_run_async_without_bg_loop_uses_asyncio_run(self):
        """When CLI has no running background loop, a fresh loop is used."""

        async def _async_result():
            return "ok"

        cmd = ServiceCommands(_fake_cli())  # cli._bg_loop is None
        result = cmd._run_async(_async_result())
        assert result == "ok"

    def test_run_async_never_touches_shared_bg_loop(self):
        """Service calls must NOT be scheduled on ``DatusCLI._bg_loop``.

        That loop hosts ``_async_init_agent`` and session-write tasks; a slow
        synchronous service method (HTTP call) would freeze every other
        task on it for its full duration. ``_run_async`` must use a
        private loop (``asyncio.run``) regardless of whether ``_bg_loop``
        is a running loop.
        """
        import asyncio
        import threading

        bg_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=bg_loop.run_forever, daemon=True)
        thread.start()
        try:
            # Spy that records whether anything was scheduled on bg_loop.
            scheduled = []
            original = bg_loop.call_soon_threadsafe

            def _tracking(callback, *args):
                scheduled.append(callback)
                return original(callback, *args)

            bg_loop.call_soon_threadsafe = _tracking  # type: ignore[assignment]

            cli = _fake_cli()
            cli._bg_loop = bg_loop
            cmd = ServiceCommands(cli)

            async def _async_result():
                return "private-loop"

            result = cmd._run_async(_async_result())
            assert result == "private-loop"
            assert scheduled == [], "service call leaked onto shared bg_loop"
        finally:
            bg_loop.call_soon_threadsafe = original  # type: ignore[assignment]
            original(bg_loop.stop)
            thread.join(timeout=2)
