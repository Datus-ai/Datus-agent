# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for ``datus.cli.service_commands`` dispatcher."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
        services=SimpleNamespace(bi_platforms={"superset": {}}, schedulers={}, semantic_layer={}),
    )
    return cli


def _printed_text(cli) -> str:
    """Join all text that was sent to ``cli.console.print``.

    Rich Table / Panel arguments are rendered through a disposable Rich
    ``Console`` so assertions can check the actual on-screen text rather
    than object reprs. Plain strings pass through unchanged.
    """
    import io

    from rich.console import Console

    buf = io.StringIO()
    probe = Console(file=buf, no_color=True, width=400)
    for call in cli.console.print.call_args_list:
        for arg in call.args:
            if isinstance(arg, str):
                buf.write(arg + "\n")
            else:
                probe.print(arg)
    return buf.getvalue()


def _make_commands_with_bi_stub(tool_instance=None):
    cli = _fake_cli()
    cmd = ServiceCommands(cli)
    # Inject a registry with the stub tool directly — bypass factory.
    registry = ServiceClientRegistry.__new__(ServiceClientRegistry)
    registry._agent_config = cli.agent_config
    registry._entries = {}
    registry._fingerprint = None
    # The probe path would check for a real datus_bi_core registration; since
    # we're injecting a stub tool, mark the adapter as available so preflight
    # in ``_invoke`` doesn't short-circuit.
    registry._adapter_available = {"superset": True}
    registry._clients = {
        "superset": ServiceClient(
            service_type="bi_platforms",
            service_name="superset",
            tool_instance=tool_instance or _BiToolStub(),
            method_names=READ_METHODS["bi_platforms"],
        ),
    }
    # _entries drives list_services output + has() lookups (2-tuple per current schema).
    registry._entries["superset"] = ("bi_platforms", "superset")
    cmd._registry = registry
    return cmd, cli


class TestDispatchServiceListing:
    def test_dispatch_bare_service_prints_methods(self):
        cmd, cli = _make_commands_with_bi_stub()
        handled = cmd.dispatch("/superset", "")
        assert handled is True
        # Rich table passed through console.print — at least one call occurred.
        assert cli.console.print.call_count >= 1

    def test_dispatch_unknown_service_returns_false(self):
        cmd, _ = _make_commands_with_bi_stub()
        handled = cmd.dispatch("/mystery", "")
        assert handled is False

    def test_dispatch_non_slash_command_returns_false(self):
        cmd, _ = _make_commands_with_bi_stub()
        # Only slash-prefixed input is recognised; bare or dot-prefixed
        # tokens are ignored so the caller can fall through to SQL / chat
        # without the dispatcher swallowing the input.
        assert cmd.dispatch("superset", "") is False
        assert cmd.dispatch(".superset", "") is False

    def test_cmd_services_lists_all(self):
        """The ``list`` sub-token preserves the read-only listing path."""
        cmd, cli = _make_commands_with_bi_stub()
        cmd.cmd_services("list")
        # Printed a table (Table object, not a string).
        assert cli.console.print.call_count == 1

    def test_cmd_services_empty_prints_hint(self):
        """``list`` on an empty config still emits the "no services" hint."""
        cli = _fake_cli()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={}, schedulers={}, semantic_layer={}),
        )
        cmd = ServiceCommands(cli)
        cmd.cmd_services("list")
        # Should have printed at least one yellow hint message.
        msg = str(cli.console.print.call_args_list[0])
        assert "No services configured" in msg


class TestDispatchInvokeMethod:
    def test_positional_arg_success(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.get_dashboard", "42")
        # Single-dict payload renders as a K/V Table; id appears as a Value cell.
        assert "42" in _printed_text(cli)

    def test_named_arg_success(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.get_dashboard", "--dashboard_id=7")
        assert "7" in _printed_text(cli)

    def test_int_coercion_from_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.get_chart_data", "99 --limit=50")
        rendered = _printed_text(cli)
        assert "99" in rendered
        assert "50" in rendered  # int, not str

    def test_missing_required_shows_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.get_dashboard", "")
        rendered = _printed_text(cli)
        assert "Missing required argument(s): dashboard_id" in rendered

    def test_help_flag_shows_schema(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.get_dashboard", "--help")
        # Rich Table object is passed to console.print; inspect its title.
        printed = cli.console.print.call_args_list[-1].args[0]
        assert "parameters" in str(getattr(printed, "title", "")).lower()

    def test_write_method_is_blocked(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.create_dashboard", "--title=x")
        msg = _printed_text(cli)
        assert "Method 'create_dashboard' is a write or privileged operation." in msg
        assert "The CLI only exposes read-only service methods. Use agent mode to invoke writes." in msg

    def test_unknown_method_prints_hint(self):
        cmd, cli = _make_commands_with_bi_stub()
        cmd.dispatch("/superset.no_such_method", "")
        msg = _printed_text(cli)
        assert "Unknown method 'no_such_method' on service 'superset'." in msg

    def test_tool_error_rendered(self):
        """When FuncToolResult.success==0, error is surfaced."""
        cmd, cli = _make_commands_with_bi_stub()
        # get_dashboard with empty id returns success=0
        cmd.dispatch("/superset.get_dashboard", "''")
        msg = _printed_text(cli)
        assert "dashboard_id required" in msg


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
        assert cmd._last_parse_error == "Unknown parameter '--bogus'. Valid parameters: a, limit."

    def test_parse_error_resets_between_calls(self):
        """A successful parse after a failed one must clear the stale error."""
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"a": {"type": "string"}}}
        assert cmd._parse_args("--bogus=x", schema) is None
        assert cmd._last_parse_error == "Unknown parameter '--bogus'. Valid parameters: a."
        # Second call succeeds → sentinel cleared.
        assert cmd._parse_args("--a=ok", schema) == {"a": "ok"}
        assert cmd._last_parse_error is None

    def test_parse_extra_positional_records_hint(self):
        cmd = ServiceCommands(_fake_cli())
        schema = {"properties": {"only": {"type": "string"}}}
        assert cmd._parse_args("first second third", schema) is None
        assert cmd._last_parse_error == "Too many positional arguments. Method accepts 1 (got extra: 'second')."

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


class TestPreflightMissingAdapter:
    """Preflight check in ``_invoke`` / ``dispatch`` for missing adapter packages.

    When the adapter package behind a configured service isn't installed, the
    generic ``_build_adapter`` path raises a cryptic ``DatusException`` deep
    inside the call chain. Preflight surfaces a pointed, actionable message
    before any invocation attempt.
    """

    def _make_registry_with_missing_adapter(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        registry = ServiceClientRegistry.__new__(ServiceClientRegistry)
        registry._agent_config = cli.agent_config
        registry._entries = {"superset": ("bi_platforms", "superset")}
        registry._fingerprint = None
        # Probe fails → adapter reported as missing.
        registry._adapter_available = {"superset": False}
        registry._clients = {
            "superset": ServiceClient(
                service_type="bi_platforms",
                service_name="superset",
                tool_instance=_BiToolStub(),
                method_names=READ_METHODS["bi_platforms"],
            ),
        }
        cmd._registry = registry
        return cmd, cli

    def test_invoke_skips_with_install_hint(self):
        cmd, cli = self._make_registry_with_missing_adapter()
        cmd.dispatch("/superset.list_dashboards", "")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "not installed" in msg
        assert "datus-bi-" in msg

    def test_bare_service_also_skips(self):
        cmd, cli = self._make_registry_with_missing_adapter()
        cmd.dispatch("/superset", "")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "not installed" in msg
        # The normal "read methods" table is NOT shown when the adapter is missing.
        assert "read methods" not in msg

    def test_scheduler_missing_mentions_scheduler_package(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        registry = ServiceClientRegistry.__new__(ServiceClientRegistry)
        registry._agent_config = cli.agent_config
        registry._entries = {"airflow": ("schedulers", "airflow")}
        registry._fingerprint = None
        registry._adapter_available = {"airflow": False}
        registry._clients = {
            "airflow": ServiceClient(
                service_type="schedulers",
                service_name="airflow",
                tool_instance=_BiToolStub(),  # stub; never reached
                method_names=READ_METHODS["schedulers"],
            ),
        }
        cmd._registry = registry
        cmd.dispatch("/airflow.list_scheduler_jobs", "")
        msg = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "datus-scheduler" in msg


class TestRenderHelpers:
    def test_render_result_success_payload(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": {"id": 42}})
        assert "42" in _printed_text(cli)

    def test_render_result_failure(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 0, "error": "boom"})
        msg = str(cli.console.print.call_args_list[0])
        assert "boom" in msg and "Error" in msg

    def test_render_list_of_dict_as_table(self):
        """Row-shaped payloads should render via Rich Table, not JSON."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": [
                    {"id": 1, "name": "Finance"},
                    {"id": 2, "name": "Sales"},
                ],
            }
        )
        # Single print call, argument is a Rich Table.
        assert cli.console.print.call_count == 1
        assert isinstance(cli.console.print.call_args_list[0].args[0], Table)

    def test_render_list_takes_union_of_keys(self):
        """Sparse rows still share a consistent column set."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": [
                    {"id": 1, "name": "a"},
                    {"id": 2, "name": "b", "extra": "x"},
                ],
            }
        )
        arg = cli.console.print.call_args_list[0].args[0]
        assert isinstance(arg, Table)
        column_labels = [str(c.header) for c in arg.columns]
        assert column_labels == ["id", "name", "extra"]

    def test_render_single_dict_as_kv_table(self):
        """Single-object payloads render as a Field/Value table so large
        nested blobs (``extra.raw`` on BI get_* responses) don't turn
        the output into a wall of JSON."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": {"id": 42, "name": "solo"}})
        arg = cli.console.print.call_args_list[0].args[0]
        assert isinstance(arg, Table)
        headers = [str(c.header) for c in arg.columns]
        assert headers == ["Field", "Value"]
        # Each row is (key, value) — pull cells out of Rich's internals.
        field_cells = list(arg.columns[0].cells)
        value_cells = list(arg.columns[1].cells)
        assert field_cells == ["id", "name"]
        assert value_cells == ["42", "solo"]

    def test_render_single_dict_truncates_wide_values(self):
        """Long nested values are middle-truncated so the table fits."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        wide = {"k": "x" * 500}
        cmd._render_result({"success": 1, "result": {"extra": wide}})
        arg = cli.console.print.call_args_list[0].args[0]
        assert isinstance(arg, Table)
        value_cell = list(arg.columns[1].cells)[0]
        # Truncated form contains the marker plus the original head/tail.
        assert "..." in value_cell
        assert len(value_cell) < 500

    def test_render_nested_values_serialised_inline(self):
        """Cells for dict / list values show compact JSON, not a repr."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": [{"id": 1, "chart_ids": [], "extra": {"k": "v"}}],
            }
        )
        arg = cli.console.print.call_args_list[0].args[0]
        assert isinstance(arg, Table)
        # Pull cell contents out of Rich internals — the column whose
        # header is "extra" should contain the JSON-stringified dict.
        headers = [str(c.header) for c in arg.columns]
        extra_idx = headers.index("extra")
        cells = list(arg.columns[extra_idx].cells)
        assert cells == ['{"k": "v"}']

    def test_render_list_envelope_as_row_table(self):
        """``FuncToolListResult`` envelope (items + total + has_more + extra)
        from any list_* tool renders as a Rich table over ``items``."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": {
                    "items": [
                        {"name": "revenue", "type": "metric"},
                        {"name": "orders", "type": "count"},
                    ],
                    "total": 2,
                    "has_more": False,
                    "extra": None,
                },
            }
        )
        tables = [c.args[0] for c in cli.console.print.call_args_list if isinstance(c.args[0], Table)]
        assert len(tables) == 1
        headers = [str(c.header) for c in tables[0].columns]
        assert headers == ["name", "type"]
        name_cells = list(tables[0].columns[0].cells)
        assert name_cells == ["revenue", "orders"]

    def test_render_list_envelope_shows_pagination_hint(self):
        """When more rows exist upstream and ``extra.next_offset`` is set, a
        dim ``Showing 2 of 137. Next: /<service>.<method> --offset=2`` hint
        follows the table so the user can paste the command back in."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": {
                    "items": [{"id": 1}, {"id": 2}],
                    "total": 137,
                    "has_more": True,
                    "extra": {"next_offset": 2},
                },
            },
            service="superset",
            method="list_dashboards",
        )
        calls = cli.console.print.call_args_list
        tables = [c.args[0] for c in calls if isinstance(c.args[0], Table)]
        hints = [c.args[0] for c in calls if isinstance(c.args[0], str)]
        assert len(tables) == 1
        assert any("Showing 2 of 137" in h for h in hints)
        assert any("/superset.list_dashboards --offset=2" in h for h in hints)

    def test_render_list_envelope_without_next_offset_is_silent(self):
        """Last page: ``has_more=False`` (no next_offset) means no hint."""
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": {
                    "items": [{"id": 1}, {"id": 2}],
                    "total": 2,
                    "has_more": False,
                    "extra": None,
                },
            },
            service="superset",
            method="list_dashboards",
        )
        calls = cli.console.print.call_args_list
        tables = [c.args[0] for c in calls if isinstance(c.args[0], Table)]
        hints = [c.args[0] for c in calls if isinstance(c.args[0], str) and "Showing" in c.args[0]]
        assert len(tables) == 1
        assert hints == []

    def test_render_empty_list_envelope_prints_empty_set(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result(
            {
                "success": 1,
                "result": {"items": [], "total": 0, "has_more": False, "extra": None},
            }
        )
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "Empty set" in rendered

    def test_render_heterogeneous_list_falls_back_to_json(self):
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": [1, 2, "three"]})
        arg = cli.console.print.call_args_list[0].args[0]
        # Heterogeneous list → JSON string, not a Table.
        assert not isinstance(arg, Table)
        assert isinstance(arg, str)
        assert "1" in arg and "three" in arg


class TestQueryEnvelopeRendering:
    """``query_metrics`` returns ``{columns, data: <compressor envelope>, metadata}``.

    Without special handling the CLI renders the compressor envelope as a
    K/V cell that shows only serializer metadata (``original_rows`` etc.)
    and hides the row values. ``_render_query_envelope`` unwraps it.
    """

    _COMPRESSOR_PAYLOAD = {
        "columns": ["new_product_ac_ratio"],
        "data": {
            "original_rows": 1,
            "original_columns": ["new_product_ac_ratio"],
            "is_compressed": False,
            "compressed_data": "new_product_ac_ratio\n0.5284\n",
            "removed_columns": [],
            "compression_type": "none",
        },
        "metadata": {"request_id": "abc"},
    }

    def test_compressor_csv_rendered_as_row_table(self):
        from rich.table import Table

        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": self._COMPRESSOR_PAYLOAD})
        tables = [c.args[0] for c in cli.console.print.call_args_list if isinstance(c.args[0], Table)]
        assert len(tables) == 1
        headers = [str(c.header) for c in tables[0].columns]
        assert headers == ["new_product_ac_ratio"]
        cells = list(tables[0].columns[0].cells)
        assert cells == ["0.5284"]

    def test_metadata_rendered_when_serializable(self):
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": self._COMPRESSOR_PAYLOAD})
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "request_id" in rendered

    def test_empty_dataset_prints_empty_set(self):
        payload = {
            "columns": [],
            "data": {
                "original_rows": 0,
                "original_columns": [],
                "is_compressed": False,
                "compressed_data": "Empty dataset",
                "removed_columns": [],
                "compression_type": "none",
            },
            "metadata": {},
        }
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": payload})
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "Empty set" in rendered

    def test_truncation_hint_when_rows_exceed_shown(self):
        payload = {
            "columns": ["a"],
            "data": {
                "original_rows": 500,
                "original_columns": ["a", "b"],
                "is_compressed": True,
                "compressed_data": "a\n1\n2\n3\n",
                "removed_columns": ["b"],
                "compression_type": "drop_columns",
            },
            "metadata": {},
        }
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": payload})
        rendered = " ".join(str(c) for c in cli.console.print.call_args_list)
        assert "Showing 3 of 500" in rendered
        assert "Omitted columns: b" in rendered

    def test_non_compressor_dict_falls_through_to_kv(self):
        """A ``{columns, data, metadata}`` look-alike without the full
        compressor key set must not be mistakenly unwrapped."""
        payload = {"columns": ["x"], "data": {"some": "object"}, "metadata": {}}
        cli = _fake_cli()
        cmd = ServiceCommands(cli)
        cmd._render_result({"success": 1, "result": payload})
        from rich.table import Table

        # Falls back to K/V table rendering (no unwrapping).
        assert any(isinstance(c.args[0], Table) for c in cli.console.print.call_args_list)


class TestServiceConfigDispatch:
    """``cmd_services`` token dispatch — the TUI path is invoked by
    ``dashboard`` / ``scheduler`` / ``config`` tokens; bare and ``list``
    tokens render the read-only listing instead."""

    def test_bare_token_opens_menu_on_dashboard_tab(self):
        """Bare ``/services`` lands directly in the TUI; the listing is
        only emitted on ``/services list``."""
        cmd, _ = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_config_menu") as menu:
            cmd.cmd_services("")
        menu.assert_called_once()
        assert menu.call_args.kwargs == {"initial_tab": "dashboard"}

    def test_list_token_renders_listing(self):
        cmd, cli = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_config_menu") as menu:
            cmd.cmd_services("list")
        menu.assert_not_called()
        assert cli.console.print.call_count >= 1

    def test_dashboard_token_opens_menu_with_dashboard_tab(self):
        cmd, _ = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_config_menu") as menu:
            cmd.cmd_services("dashboard")
        menu.assert_called_once()
        assert menu.call_args.kwargs == {"initial_tab": "dashboard"}

    def test_scheduler_token_opens_menu_with_scheduler_tab(self):
        cmd, _ = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_config_menu") as menu:
            cmd.cmd_services("scheduler")
        menu.assert_called_once()
        assert menu.call_args.kwargs == {"initial_tab": "scheduler"}

    def test_unknown_token_falls_back_to_listing_with_hint(self):
        cmd, cli = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_config_menu") as menu:
            cmd.cmd_services("foobar")
        menu.assert_not_called()
        rendered = _printed_text(cli)
        assert (
            "Use `/services dashboard`, `/services scheduler`, or `/services semantic` to open the configuration TUI."
            in rendered
        )


class TestApplySelectionPersistence:
    """``_apply_selection`` orchestrates install → probe → persist for save,
    and the right ``configuration_manager`` calls for delete /
    set_default / set_project_default."""

    def _commands_with_writable_cli(self):
        cli = _fake_cli()
        # Set ``services`` shape so the listing path doesn't crash.
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"old": {}}, schedulers={}, semantic_layer={}),
            set_active_dashboard=MagicMock(),
            set_active_scheduler=MagicMock(),
            active_dashboard=MagicMock(return_value=None),
            active_scheduler=MagicMock(return_value=None),
        )
        return ServiceCommands(cli), cli

    def test_save_runs_install_probe_and_persists(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, cli = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(
            action="save",
            section="bi_platforms",
            name="superset",
            payload={"type": "superset", "api_base_url": "http://x", "username": "u"},
        )
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {}, "schedulers": {}, "semantic_layer": {}})
        with (
            patch("datus.cli.service_adapter_installer.ensure_adapter") as ensure,
            patch("datus.cli.service_adapter_installer.hot_reload_adapter") as hot,
            patch("datus.cli.service_commands.ServiceCommands._probe", return_value=(True, "5 dashboards")),
            patch("datus.cli.service_commands.ServiceCommands._reload_agent_config"),
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
        ):
            ensure.return_value = SimpleNamespace(
                ok=True, package="datus-bi-superset", import_name="datus_bi_superset", stdout="", stderr=""
            )
            status = cmd._apply_selection(sel)
        ensure.assert_called_once_with("bi_platforms", "superset")
        hot.assert_called_once_with("bi_platforms", "superset")
        mgr.update_item.assert_called_once()
        kwargs = mgr.update_item.call_args
        assert kwargs.args[0] == "services"
        assert "Saved" in (status or "")
        assert "Connected" in (status or "")

    def test_save_persists_even_when_probe_fails(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(
            action="save",
            section="bi_platforms",
            name="superset",
            payload={"type": "superset", "api_base_url": "http://broken"},
        )
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {}})
        with (
            patch("datus.cli.service_adapter_installer.ensure_adapter") as ensure,
            patch("datus.cli.service_adapter_installer.hot_reload_adapter"),
            patch(
                "datus.cli.service_commands.ServiceCommands._probe",
                return_value=(False, "Connection refused"),
            ),
            patch("datus.cli.service_commands.ServiceCommands._reload_agent_config"),
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
        ):
            ensure.return_value = SimpleNamespace(
                ok=True, package="datus-bi-superset", import_name="datus_bi_superset", stdout="", stderr=""
            )
            status = cmd._apply_selection(sel)
        # YAML still written — probe failure is informational, not fatal.
        mgr.update_item.assert_called_once()
        assert "Probe failed" in (status or "")

    def test_save_skipped_on_install_failure(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(
            action="save",
            section="bi_platforms",
            name="ghost",
            payload={"type": "ghost", "api_base_url": "http://x"},
        )
        mgr = MagicMock()
        with (
            patch("datus.cli.service_adapter_installer.ensure_adapter") as ensure,
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
        ):
            ensure.return_value = SimpleNamespace(
                ok=False,
                package="datus-bi-ghost",
                import_name="datus_bi_ghost",
                stdout="",
                stderr="not found",
                error="exit 1",
            )
            status = cmd._apply_selection(sel)
        mgr.update_item.assert_not_called()
        assert status == "Saving `ghost` skipped — adapter install failed."

    def test_delete_persists_with_delete_old_key_true(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(action="delete", section="bi_platforms", name="old")
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {"old": {}}, "schedulers": {}})
        with (
            patch("datus.cli.service_commands.ServiceCommands._reload_agent_config"),
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
        ):
            cmd._apply_selection(sel)
        # delete_old_key=True so the YAML key is dropped, not just blanked.
        mgr.update_item.assert_called_once()
        assert mgr.update_item.call_args.kwargs.get("delete_old_key") is True

    def test_set_global_default_clears_others(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(action="set_default", section="schedulers", name="airflow_dev")
        mgr = MagicMock()
        mgr.get = MagicMock(
            return_value={
                "schedulers": {
                    "airflow_prod": {"type": "airflow", "default": True},
                    "airflow_dev": {"type": "airflow"},
                },
            }
        )
        with (
            patch("datus.cli.service_commands.ServiceCommands._reload_agent_config"),
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
        ):
            cmd._apply_selection(sel)
        mgr.update_item.assert_called_once()
        # The committed payload should mark the new entry default and strip the old one.
        committed = mgr.update_item.call_args.args[1]
        schedulers = committed["schedulers"]
        assert schedulers["airflow_dev"]["default"] is True
        assert "default" not in schedulers["airflow_prod"]

    def test_set_project_default_calls_setter(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, cli = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(action="set_project_default", section="bi_platforms", name="superset")
        cmd._apply_selection(sel)
        cli.agent_config.set_active_dashboard.assert_called_once_with("superset")

    def test_set_project_default_clears_when_name_empty(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, cli = self._commands_with_writable_cli()
        sel = ServiceConfigSelection(action="set_project_default", section="schedulers", name="")
        cmd._apply_selection(sel)
        cli.agent_config.set_active_scheduler.assert_called_once_with(None)


class TestRunConfigMenu:
    """``_run_config_menu`` loops the TUI; ``_run_app`` wraps stdin via
    ``tui_app.suspend_input`` when the outer REPL is in TUI mode.
    Direct unit coverage of these helpers because the integration path
    (real prompt_toolkit Application) is impossible to exercise headless."""

    def test_loop_exits_when_app_returns_none(self):
        cmd, _ = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_run_app", return_value=None) as run_app:
            cmd._run_config_menu(initial_tab="dashboard")
        assert run_app.call_count == 1

    def test_loop_iterates_then_exits(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        scripted = [
            ServiceConfigSelection(action="test", section="bi_platforms", name="superset"),
            None,  # second iteration cancels
        ]
        with (
            patch.object(ServiceCommands, "_run_app", side_effect=scripted) as run_app,
            patch.object(ServiceCommands, "_apply_selection", return_value="status") as apply_sel,
            patch.object(ServiceCommands, "_refresh_after_change") as refresh,
        ):
            cmd._run_config_menu(initial_tab="dashboard")
        assert run_app.call_count == 2
        apply_sel.assert_called_once()
        refresh.assert_called_once()

    def test_seed_tab_follows_section_of_last_selection(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        sequences = [
            ServiceConfigSelection(action="test", section="schedulers", name="airflow"),
            None,
        ]
        captured_tabs = []

        def fake_run_app(self_inner, app):
            # Inspect the tab the App was constructed with.
            captured_tabs.append(app._tab.value)
            return sequences.pop(0)

        with (
            patch.object(ServiceCommands, "_run_app", new=fake_run_app),
            patch.object(ServiceCommands, "_apply_selection", return_value=None),
        ):
            cmd._run_config_menu(initial_tab="dashboard")
        # First app starts on dashboard (initial_tab); second app re-seeded
        # to scheduler because the prior selection.section == "schedulers".
        assert captured_tabs == ["dashboard", "scheduler"]

    def test_run_app_uses_run_wizard_when_tui_active(self):
        """When the parent TUI has an active loop, the wizard is
        embedded via ``DatusApp.run_wizard`` (mounts it in the bottom
        slot). ``fake_app.run()`` is bypassed entirely."""
        cmd, cli = _make_commands_with_bi_stub()
        cli.tui_app = MagicMock()
        cli.tui_app._loop = MagicMock()  # truthy → embedded path
        cli.tui_app.run_wizard = MagicMock(return_value="embedded-result")

        fake_app = MagicMock()
        fake_app.build_embedded_panel = MagicMock()
        fake_app.run = MagicMock(return_value="should-not-be-called")
        result = cmd._run_app(fake_app)
        assert result == "embedded-result"
        cli.tui_app.run_wizard.assert_called_once_with(fake_app.build_embedded_panel)
        fake_app.run.assert_not_called()

    def test_run_app_runs_directly_without_tui(self):
        cmd, cli = _make_commands_with_bi_stub()
        cli.tui_app = None  # outer REPL not in TUI mode
        fake_app = MagicMock()
        fake_app.run = MagicMock(return_value="ran-direct")
        assert cmd._run_app(fake_app) == "ran-direct"

    def test_refresh_after_change_drops_registry_cache(self):
        cmd, _ = _make_commands_with_bi_stub()
        assert isinstance(cmd._registry, ServiceClientRegistry)  # primed by _make_commands_with_bi_stub
        cmd._refresh_after_change()
        assert cmd._registry is None


class TestApplySelectionMissingType:
    """Save with no ``type`` payload short-circuits without YAML write."""

    def test_save_with_missing_type_returns_none(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cli = _fake_cli()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={}, schedulers={}, semantic_layer={}),
        )
        cmd = ServiceCommands(cli)
        sel = ServiceConfigSelection(action="save", section="bi_platforms", name="x", payload={})
        with patch("datus.configuration.agent_config_loader.configuration_manager") as mgr_factory:
            result = cmd._apply_selection(sel)
        assert result is None
        mgr_factory.assert_not_called()

    def test_unknown_action_returns_none(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        sel = ServiceConfigSelection(action="garbled", section="bi_platforms", name="x")
        assert cmd._apply_selection(sel) is None


class TestDeleteEdgeCases:
    """``_do_delete`` covers absent entries, persistence failure, and
    automatic clearing of a project pin that pointed at the deleted entry."""

    def _fresh_cmd(self):
        cli = _fake_cli()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={"old": {}}, schedulers={}, semantic_layer={}),
            active_dashboard=MagicMock(return_value="old"),
            active_scheduler=MagicMock(return_value=None),
            set_active_dashboard=MagicMock(),
            set_active_scheduler=MagicMock(),
        )
        return ServiceCommands(cli), cli

    def test_delete_missing_entry_warns(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, cli = self._fresh_cmd()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {}})
        with patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr):
            status = cmd._apply_selection(ServiceConfigSelection(action="delete", section="bi_platforms", name="ghost"))
        mgr.update_item.assert_not_called()
        assert "not found" in (status or "")

    def test_delete_persistence_exception_surfaces(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = self._fresh_cmd()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {"old": {}}})
        mgr.update_item = MagicMock(side_effect=RuntimeError("disk full"))
        with patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr):
            status = cmd._apply_selection(ServiceConfigSelection(action="delete", section="bi_platforms", name="old"))
        assert "failed" in (status or "").lower()

    def test_delete_clears_project_default_when_pointing_at_removed(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, cli = self._fresh_cmd()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"bi_platforms": {"old": {}}})
        with (
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
            patch.object(ServiceCommands, "_reload_agent_config"),
        ):
            cmd._apply_selection(ServiceConfigSelection(action="delete", section="bi_platforms", name="old"))
        cli.agent_config.set_active_dashboard.assert_called_once_with(None)


class TestTestAndDefaultEdgeCases:
    def test_test_action_reports_ok(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        with patch.object(ServiceCommands, "_probe", return_value=(True, "5 dashboards")):
            status = cmd._apply_selection(
                ServiceConfigSelection(action="test", section="bi_platforms", name="superset")
            )
        assert "Probe ok" in (status or "")

    def test_set_global_default_missing_entry_warns(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"schedulers": {}})
        with patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr):
            status = cmd._apply_selection(
                ServiceConfigSelection(action="set_default", section="schedulers", name="ghost")
            )
        mgr.update_item.assert_not_called()
        assert "not found" in (status or "")

    def test_set_global_default_persistence_exception(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={"schedulers": {"airflow": {"type": "airflow"}}})
        mgr.update_item = MagicMock(side_effect=RuntimeError("disk full"))
        with patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr):
            status = cmd._apply_selection(
                ServiceConfigSelection(action="set_default", section="schedulers", name="airflow")
            )
        assert "failed" in (status or "").lower()

    def test_set_global_default_skips_non_dict_entries(self):
        """Defensive: a non-dict entry under ``schedulers`` (legacy YAML
        glitch) is left alone instead of crashing the loop."""
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(
            return_value={
                "schedulers": {
                    "airflow": {"type": "airflow"},
                    "stale_token": "not-a-dict",  # malformed entry
                }
            }
        )
        with (
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
            patch.object(ServiceCommands, "_reload_agent_config"),
        ):
            cmd._apply_selection(ServiceConfigSelection(action="set_default", section="schedulers", name="airflow"))
        mgr.update_item.assert_called_once()
        committed = mgr.update_item.call_args.args[1]["schedulers"]
        # Non-dict entry preserved verbatim; default flag flipped on the airflow entry.
        assert committed["stale_token"] == "not-a-dict"
        assert committed["airflow"]["default"] is True

    def test_set_global_default_works_for_bi_platforms(self):
        """``set_default`` is now generic across BI / Scheduler / Semantic.
        For bi_platforms, the YAML ``default`` flag flips on the chosen
        entry and clears from the others, mirroring the scheduler test
        above."""
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(
            return_value={
                "bi_platforms": {
                    "superset": {"type": "superset", "default": True},
                    "grafana": {"type": "grafana"},
                }
            }
        )
        with (
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
            patch.object(ServiceCommands, "_reload_agent_config"),
        ):
            cmd._apply_selection(ServiceConfigSelection(action="set_default", section="bi_platforms", name="grafana"))
        mgr.update_item.assert_called_once()
        committed = mgr.update_item.call_args.args[1]["bi_platforms"]
        assert committed["grafana"]["default"] is True
        assert "default" not in committed["superset"]

    def test_set_global_default_works_for_semantic_layer(self):
        """Same generic pathway for semantic_layer."""
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(
            return_value={
                "semantic_layer": {
                    "metricflow": {"type": "metricflow", "default": True},
                    "dbt": {"type": "dbt"},
                }
            }
        )
        with (
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
            patch.object(ServiceCommands, "_reload_agent_config"),
        ):
            cmd._apply_selection(ServiceConfigSelection(action="set_default", section="semantic_layer", name="dbt"))
        mgr.update_item.assert_called_once()
        committed = mgr.update_item.call_args.args[1]["semantic_layer"]
        assert committed["dbt"]["default"] is True
        assert "default" not in committed["metricflow"]

    def test_set_global_default_unknown_section_returns_none(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        with patch("datus.configuration.agent_config_loader.configuration_manager") as mgr_factory:
            result = cmd._apply_selection(ServiceConfigSelection(action="set_default", section="datasources", name="x"))
        assert result is None
        mgr_factory.assert_not_called()

    def test_set_project_default_for_semantic_layer(self):
        """semantic_layer is now wired into ``_do_set_project_default``;
        the section's ``set_active_semantic`` setter must be invoked with
        the chosen name."""
        from datus.cli.service_config_app import ServiceConfigSelection

        cli = _fake_cli()
        setter = MagicMock()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={}, schedulers={}, semantic_layer={}),
            set_active_semantic=setter,
        )
        cmd = ServiceCommands(cli)
        result = cmd._apply_selection(
            ServiceConfigSelection(action="set_project_default", section="semantic_layer", name="metricflow")
        )
        setter.assert_called_once_with("metricflow")
        assert result and "metricflow" in result

    def test_set_project_default_unknown_section_returns_none(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cmd, _ = _make_commands_with_bi_stub()
        result = cmd._apply_selection(
            ServiceConfigSelection(action="set_project_default", section="datasources", name="x")
        )
        assert result is None

    def test_set_project_default_setter_missing(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cli = _fake_cli()
        # Strip the setters so the helper sees ``None``.
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={}, schedulers={}, semantic_layer={}),
        )
        cmd = ServiceCommands(cli)
        result = cmd._apply_selection(
            ServiceConfigSelection(action="set_project_default", section="bi_platforms", name="x")
        )
        # Setter unavailable → command logs an error and returns None.
        assert result is None

    def test_set_project_default_setter_raises(self):
        from datus.cli.service_config_app import ServiceConfigSelection

        cli = _fake_cli()
        cli.agent_config = SimpleNamespace(
            services=SimpleNamespace(bi_platforms={}, schedulers={}, semantic_layer={}),
            set_active_dashboard=MagicMock(side_effect=RuntimeError("disk full")),
            set_active_scheduler=MagicMock(),
        )
        cmd = ServiceCommands(cli)
        result = cmd._apply_selection(
            ServiceConfigSelection(action="set_project_default", section="bi_platforms", name="x")
        )
        assert "failed" in (result or "").lower()


class TestMergeServiceEntry:
    def test_strips_blank_strings_and_empty_dicts(self):
        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={})
        with (
            patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr),
            patch.object(ServiceCommands, "_reload_agent_config"),
        ):
            ok = cmd._merge_service_entry(
                "bi_platforms",
                "superset",
                {"type": "superset", "api_base_url": "http://x", "username": "", "extra": {}},
            )
        assert ok is True
        committed = mgr.update_item.call_args.args[1]["bi_platforms"]["superset"]
        assert "username" not in committed  # empty string stripped
        assert "extra" not in committed  # empty dict stripped
        assert committed["api_base_url"] == "http://x"

    def test_persistence_exception_returns_false(self):
        cmd, _ = _make_commands_with_bi_stub()
        mgr = MagicMock()
        mgr.get = MagicMock(return_value={})
        mgr.update_item = MagicMock(side_effect=RuntimeError("disk full"))
        with patch("datus.configuration.agent_config_loader.configuration_manager", return_value=mgr):
            assert cmd._merge_service_entry("bi_platforms", "x", {"type": "superset"}) is False


class TestProbeImplementation:
    """``_probe`` runs a real (mocked) ``BIFuncTool`` / ``SchedulerTools``
    call so connection errors surface as ``(False, message)``."""

    def test_probe_bi_success_counts_envelope(self):
        cmd, _ = _make_commands_with_bi_stub()
        fake_tool = MagicMock()
        fake_tool.list_dashboards = MagicMock(return_value={"success": 1, "result": {"items": [{"id": 1}, {"id": 2}]}})
        with patch("datus.tools.func_tool.bi_tools.BIFuncTool", return_value=fake_tool):
            ok, msg = cmd._probe("bi_platforms", "superset")
        assert ok is True
        assert "2 dashboards" in msg

    def test_probe_scheduler_success(self):
        cmd, _ = _make_commands_with_bi_stub()
        fake_tool = MagicMock()
        fake_tool.list_scheduler_jobs = MagicMock(return_value=[{"name": "a"}, {"name": "b"}])
        with patch("datus.tools.func_tool.scheduler_tools.SchedulerTools", return_value=fake_tool):
            ok, msg = cmd._probe("schedulers", "airflow")
        assert ok is True
        assert "2 scheduler jobs" in msg

    def test_probe_exception_is_caught(self):
        cmd, _ = _make_commands_with_bi_stub()
        fake_tool = MagicMock()
        fake_tool.list_dashboards = MagicMock(side_effect=RuntimeError("connection refused"))
        with patch("datus.tools.func_tool.bi_tools.BIFuncTool", return_value=fake_tool):
            ok, msg = cmd._probe("bi_platforms", "superset")
        assert ok is False
        assert "connection refused" in msg

    def test_probe_unsupported_section(self):
        cmd, _ = _make_commands_with_bi_stub()
        ok, msg = cmd._probe("unknown_section", "x")
        assert ok is False
        assert "Unsupported section" in msg

    def test_probe_semantic_returns_true_when_metadata_registered(self):
        cmd, _ = _make_commands_with_bi_stub()
        fake_registry = SimpleNamespace(get_metadata=lambda name: {"name": name})
        with patch.dict(
            "sys.modules",
            {"datus.tools.semantic_tools.registry": SimpleNamespace(semantic_adapter_registry=fake_registry)},
        ):
            ok, msg = cmd._probe("semantic_layer", "metricflow")
        assert ok is True
        assert "metricflow" in msg
        assert "registered" in msg

    def test_probe_semantic_returns_false_when_metadata_missing(self):
        cmd, _ = _make_commands_with_bi_stub()
        fake_registry = SimpleNamespace(get_metadata=lambda name: None)
        with patch.dict(
            "sys.modules",
            {"datus.tools.semantic_tools.registry": SimpleNamespace(semantic_adapter_registry=fake_registry)},
        ):
            ok, msg = cmd._probe("semantic_layer", "metricflow")
        assert ok is False
        assert "not registered" in msg


class TestProbeTimeout:
    """Verify that ``_probe`` enforces ``_PROBE_TIMEOUT_SECS`` for the
    network-bound branches (bi_platforms, schedulers) so a hung adapter
    cannot freeze the REPL.
    """

    def test_probe_bi_times_out_and_returns_error(self):
        import threading as _threading

        cmd, _ = _make_commands_with_bi_stub()
        cmd._PROBE_TIMEOUT_SECS = 0.05
        release = _threading.Event()
        try:
            fake_tool = MagicMock()
            fake_tool.list_dashboards = MagicMock(side_effect=lambda: release.wait(2.0) or {"items": []})
            with patch("datus.tools.func_tool.bi_tools.BIFuncTool", return_value=fake_tool):
                ok, msg = cmd._probe("bi_platforms", "superset")
            assert ok is False
            assert "timeout" in msg.lower()
            assert "unreachable" in msg.lower()
        finally:
            release.set()

    def test_probe_scheduler_times_out_and_returns_error(self):
        import threading as _threading

        cmd, _ = _make_commands_with_bi_stub()
        cmd._PROBE_TIMEOUT_SECS = 0.05
        release = _threading.Event()
        try:
            fake_tool = MagicMock()
            fake_tool.list_scheduler_jobs = MagicMock(side_effect=lambda: release.wait(2.0) or [])
            with patch("datus.tools.func_tool.scheduler_tools.SchedulerTools", return_value=fake_tool):
                ok, msg = cmd._probe("schedulers", "airflow")
            assert ok is False
            assert "timeout" in msg.lower()
        finally:
            release.set()

    def test_probe_bi_returns_quickly_when_adapter_responds(self):
        """Sanity check: a fast adapter response under the timeout still
        returns the success envelope, so the timeout path doesn't
        accidentally short-circuit the success path.
        """
        cmd, _ = _make_commands_with_bi_stub()
        cmd._PROBE_TIMEOUT_SECS = 2.0
        fake_tool = MagicMock()
        fake_tool.list_dashboards = MagicMock(return_value={"result": {"items": [{"id": 1}]}})
        with patch("datus.tools.func_tool.bi_tools.BIFuncTool", return_value=fake_tool):
            ok, msg = cmd._probe("bi_platforms", "superset")
        assert ok is True
        assert "1 dashboards" in msg

    def test_probe_semantic_layer_skips_threading_path(self):
        """Semantic-layer probe must run on the calling thread (no
        timeout) because it's a pure in-memory registry lookup.
        """
        cmd, _ = _make_commands_with_bi_stub()
        cmd._PROBE_TIMEOUT_SECS = 0.0  # would trip immediately if threaded
        fake_registry = SimpleNamespace(get_metadata=lambda name: {"name": name})
        with patch.dict(
            "sys.modules",
            {"datus.tools.semantic_tools.registry": SimpleNamespace(semantic_adapter_registry=fake_registry)},
        ):
            ok, msg = cmd._probe("semantic_layer", "metricflow")
        assert ok is True
        assert "registered" in msg


class TestCountEnvelope:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"items": [1, 2, 3]}, 3),
            ({"result": {"items": [1, 2]}}, 2),
            ({"result": [1, 2, 3, 4]}, 4),
            ([1, 2], 2),
            ("scalar", 0),
            ({}, 0),
        ],
    )
    def test_count(self, payload, expected):
        assert ServiceCommands._count_envelope(payload) == expected


class TestReloadAgentConfig:
    def test_swaps_cli_agent_config_on_success(self):
        cmd, cli = _make_commands_with_bi_stub()
        fresh = SimpleNamespace(name="fresh")
        with patch(
            "datus.configuration.agent_config_loader.load_agent_config",
            return_value=fresh,
        ):
            cmd._reload_agent_config()
        assert cli.agent_config is fresh
        assert cmd._registry is None

    def test_load_failure_keeps_existing_agent_config(self):
        cmd, cli = _make_commands_with_bi_stub()
        original = cli.agent_config
        with patch(
            "datus.configuration.agent_config_loader.load_agent_config",
            side_effect=RuntimeError("boom"),
        ):
            cmd._reload_agent_config()
        assert cli.agent_config is original


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
            bg_loop.close()
