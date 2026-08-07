# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for the tool argument middleware.

Covers the transformer contract end to end: rewrite propagation into the
wrapped tool, fail-closed denial (exception and non-dict return), sync/async
transformers, chaining order, malformed-argument passthrough, and the
node-level ``apply_tool_transformers`` matching/skipping behavior.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest
from agents import FunctionTool

from datus.tools.middleware.tool_middleware import (
    apply_tool_transformers,
    tool_is_transformed,
    wrap_tool_with_transformers,
)
from datus.tools.registry.tool_registry import ToolRegistry


def _make_tool(name="execute_sql", record=None):
    """Build a FunctionTool that records the args JSON it was invoked with."""

    async def invoke(tool_ctx, args_str):
        if record is not None:
            record.append(args_str)
        return {"success": 1, "result": f"ran:{args_str}", "error": None}

    return FunctionTool(
        name=name,
        description="test tool",
        params_json_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )


class TestWrapToolWithTransformers:
    @pytest.mark.asyncio
    async def test_rewrite_reaches_original_tool(self):
        record = []
        tool = _make_tool(record=record)

        def add_scope(tool_name, args, context):
            assert tool_name == "execute_sql"
            args["sql"] = args["sql"] + " WHERE tenant_id = 't1'"
            return args

        wrapped = wrap_tool_with_transformers(tool, [add_scope])
        result = await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT * FROM orders"}))

        assert result["success"] == 1
        assert json.loads(record[0]) == {"sql": "SELECT * FROM orders WHERE tenant_id = 't1'"}

    def test_wrapped_tool_preserves_metadata(self):
        tool = _make_tool()
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: a])
        assert wrapped.name == tool.name
        assert wrapped.description == tool.description
        assert wrapped.params_json_schema == tool.params_json_schema
        assert wrapped.strict_json_schema == tool.strict_json_schema

    def test_wrapped_tool_preserves_disabled_state(self):
        # Rebuilding the FunctionTool must not silently re-enable a tool that
        # was intentionally disabled/gated just because it matched a pattern.
        tool = _make_tool()
        tool.is_enabled = False
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: a])
        assert wrapped.is_enabled is False

    def test_wrapped_tool_carries_all_declared_fields(self):
        # The clone must forward every FunctionTool dataclass field except
        # ``on_invoke_tool`` so approval/timeout/guardrail settings survive
        # across openai-agents SDK versions that add new fields.
        tool = _make_tool()
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: a])
        # ``on_invoke_tool`` is the single field the wrapper intentionally replaces.
        assert wrapped.on_invoke_tool is not tool.on_invoke_tool
        carried = [f.name for f in dataclasses.fields(FunctionTool) if f.name != "on_invoke_tool"]
        assert {name: getattr(wrapped, name) for name in carried} == {name: getattr(tool, name) for name in carried}
        assert tool_is_transformed(wrapped)
        assert not tool_is_transformed(tool)

    @pytest.mark.asyncio
    async def test_transformer_exception_denies_fail_closed(self):
        record = []
        tool = _make_tool(record=record)

        def deny(tool_name, args, context):
            raise ValueError("tenant scope missing")

        wrapped = wrap_tool_with_transformers(tool, [deny])
        result = await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT 1"}))

        assert result["success"] == 0
        assert "tenant scope missing" in result["error"]
        assert result["result"] is None
        assert record == []  # original tool never executed

    @pytest.mark.asyncio
    async def test_non_dict_return_denies_fail_closed(self):
        record = []
        tool = _make_tool(record=record)
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: "not a dict"])
        result = await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT 1"}))

        assert result["success"] == 0
        assert "str" in result["error"]
        assert record == []

    @pytest.mark.asyncio
    async def test_async_transformer_supported(self):
        record = []
        tool = _make_tool(record=record)

        async def async_rewrite(tool_name, args, context):
            args["sql"] = "SELECT 2"
            return args

        wrapped = wrap_tool_with_transformers(tool, [async_rewrite])
        result = await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT 1"}))

        assert result["success"] == 1
        assert json.loads(record[0]) == {"sql": "SELECT 2"}

    @pytest.mark.asyncio
    async def test_transformers_chain_in_order(self):
        record = []
        tool = _make_tool(record=record)

        def first(tool_name, args, context):
            args["sql"] = args["sql"] + "|first"
            return args

        def second(tool_name, args, context):
            args["sql"] = args["sql"] + "|second"
            return args

        wrapped = wrap_tool_with_transformers(tool, [first, second])
        await wrapped.on_invoke_tool(None, json.dumps({"sql": "base"}))

        assert json.loads(record[0]) == {"sql": "base|first|second"}

    @pytest.mark.asyncio
    async def test_chain_stops_at_first_denial(self):
        record = []
        calls = []
        tool = _make_tool(record=record)

        def deny(tool_name, args, context):
            calls.append("deny")
            raise PermissionError("blocked")

        def never(tool_name, args, context):
            calls.append("never")
            return args

        wrapped = wrap_tool_with_transformers(tool, [deny, never])
        result = await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT 1"}))

        assert result["success"] == 0
        assert calls == ["deny"]
        assert record == []

    @pytest.mark.asyncio
    async def test_malformed_json_passes_through_untouched(self):
        record = []
        tool = _make_tool(record=record)

        def never(tool_name, args, context):
            raise AssertionError("transformer must not run on malformed args")

        wrapped = wrap_tool_with_transformers(tool, [never])
        result = await wrapped.on_invoke_tool(None, "{not json")

        assert result["success"] == 1
        assert record == ["{not json"]

    @pytest.mark.asyncio
    async def test_non_object_json_passes_through_untouched(self):
        record = []
        tool = _make_tool(record=record)
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: a])
        await wrapped.on_invoke_tool(None, json.dumps([1, 2]))
        assert record == ["[1, 2]"]

    @pytest.mark.asyncio
    async def test_context_provider_called_per_invocation(self):
        tool = _make_tool()
        principal_holder = {"tenant": "t1"}
        seen = []

        def transformer(tool_name, args, context):
            seen.append(context["principal"])
            return args

        wrapped = wrap_tool_with_transformers(tool, [transformer], lambda: {"principal": dict(principal_holder)})
        await wrapped.on_invoke_tool(None, "{}")
        principal_holder["tenant"] = "t2"
        await wrapped.on_invoke_tool(None, "{}")

        assert seen == [{"tenant": "t1"}, {"tenant": "t2"}]

    @pytest.mark.asyncio
    async def test_context_provider_failure_yields_empty_context(self):
        tool = _make_tool()
        seen = []

        def transformer(tool_name, args, context):
            seen.append(context)
            return args

        def broken_provider():
            raise RuntimeError("no context")

        wrapped = wrap_tool_with_transformers(tool, [transformer], broken_provider)
        result = await wrapped.on_invoke_tool(None, "{}")

        assert result["success"] == 1
        assert seen == [{}]

    @pytest.mark.asyncio
    async def test_non_ascii_args_survive_roundtrip(self):
        record = []
        tool = _make_tool(record=record)
        wrapped = wrap_tool_with_transformers(tool, [lambda n, a, c: a])
        await wrapped.on_invoke_tool(None, json.dumps({"sql": "SELECT '租户'"}, ensure_ascii=False))
        assert json.loads(record[0]) == {"sql": "SELECT '租户'"}


def _add_scope(tool_name, args, context):
    args["sql"] = args["sql"] + " WHERE tenant_id = 't1'"
    return args


def _make_write_back_tool(record=None, raises=False, name="execute_sql"):
    """A tool that canonicalises its arguments onto the context, as datus tools do."""

    async def invoke(tool_ctx, args_str):
        if record is not None:
            record.append(args_str)
        tool_ctx.tool_arguments = args_str
        if isinstance(tool_ctx.tool_call, dict):
            tool_ctx.tool_call["arguments"] = args_str
        else:
            tool_ctx.tool_call.arguments = args_str
        if raises:
            raise RuntimeError("boom")
        return {"success": 1, "result": None, "error": None}

    return FunctionTool(
        name=name,
        description="test tool",
        params_json_schema={"type": "object", "properties": {"sql": {"type": "string"}}},
        on_invoke_tool=invoke,
        strict_json_schema=False,
    )


def _make_ctx(args_str, as_dict=False):
    tool_call = {"arguments": args_str} if as_dict else SimpleNamespace(arguments=args_str)
    return SimpleNamespace(tool_arguments=args_str, tool_call=tool_call)


class TestOriginalArgumentsSurviveRewrite:
    """Datus tools write the arguments they received back onto the SDK tool call,
    which is what the session persists and the next turn replays. A rewrite left
    there shows the model a call it never made.
    """

    @pytest.mark.asyncio
    async def test_tool_call_keeps_what_the_model_sent(self):
        sent = json.dumps({"sql": "SELECT * FROM orders"})
        ctx = _make_ctx(sent)
        record = []
        wrapped = wrap_tool_with_transformers(_make_write_back_tool(record=record), [_add_scope])

        await wrapped.on_invoke_tool(ctx, sent)

        assert json.loads(record[0])["sql"] == "SELECT * FROM orders WHERE tenant_id = 't1'"
        assert ctx.tool_arguments == sent
        assert ctx.tool_call.arguments == sent

    @pytest.mark.asyncio
    async def test_injected_argument_is_kept_out_too(self):
        """Metric policies add a ``where`` the model never wrote; same rule applies."""
        sent = json.dumps({"metrics": ["revenue"]})
        ctx = _make_ctx(sent)
        record = []

        def add_where(tool_name, args, context):
            args["where"] = "orders.store_id IN ('S001')"
            return args

        wrapped = wrap_tool_with_transformers(_make_write_back_tool(record=record, name="query_metrics"), [add_where])

        await wrapped.on_invoke_tool(ctx, sent)

        assert json.loads(record[0])["where"] == "orders.store_id IN ('S001')"
        assert "where" not in json.loads(ctx.tool_call.arguments)

    @pytest.mark.asyncio
    async def test_dict_shaped_tool_call_is_restored(self):
        sent = json.dumps({"sql": "SELECT * FROM orders"})
        ctx = _make_ctx(sent, as_dict=True)
        wrapped = wrap_tool_with_transformers(_make_write_back_tool(), [_add_scope])

        await wrapped.on_invoke_tool(ctx, sent)

        assert ctx.tool_call["arguments"] == sent

    @pytest.mark.asyncio
    async def test_restored_even_when_the_tool_raises(self):
        sent = json.dumps({"sql": "SELECT * FROM orders"})
        ctx = _make_ctx(sent)
        wrapped = wrap_tool_with_transformers(_make_write_back_tool(raises=True), [_add_scope])

        with pytest.raises(RuntimeError):
            await wrapped.on_invoke_tool(ctx, sent)

        assert ctx.tool_call.arguments == sent

    @pytest.mark.asyncio
    async def test_denied_call_leaves_the_tool_call_untouched(self):
        sent = json.dumps({"sql": "SELECT * FROM orders"})
        ctx = _make_ctx(sent)

        def deny(tool_name, args, context):
            raise PermissionError("no principal")

        wrapped = wrap_tool_with_transformers(_make_write_back_tool(), [deny])
        result = await wrapped.on_invoke_tool(ctx, sent)

        assert result["success"] == 0
        assert ctx.tool_call.arguments == sent

    @pytest.mark.asyncio
    async def test_context_without_a_tool_call_is_tolerated(self):
        sent = json.dumps({"sql": "SELECT * FROM orders"})
        record = []
        wrapped = wrap_tool_with_transformers(_make_tool(record=record), [_add_scope])

        result = await wrapped.on_invoke_tool(None, sent)

        assert result["success"] == 1
        assert json.loads(record[0])["sql"].endswith("WHERE tenant_id = 't1'")

    @pytest.mark.asyncio
    async def test_unsettable_context_does_not_fail_the_call(self):
        class _Frozen:
            __slots__ = ()

        record = []
        wrapped = wrap_tool_with_transformers(_make_tool(record=record), [_add_scope])

        result = await wrapped.on_invoke_tool(_Frozen(), json.dumps({"sql": "SELECT 1"}))

        assert result["success"] == 1


def _make_node(tools, registry_map=None, proxied=None):
    return SimpleNamespace(
        tools=tools,
        tool_registry=ToolRegistry(registry_map or {}),
        proxied_tool_names=proxied or set(),
        get_node_name=lambda: "chat",
        db_func_tool=SimpleNamespace(principal={"tenant": {"id": "t1"}}),
        agent_config=SimpleNamespace(project_root="/proj"),
    )


class TestApplyToolTransformers:
    @pytest.mark.asyncio
    async def test_wraps_matching_tool_by_name(self):
        record = []
        node = _make_node([_make_tool("execute_sql", record), _make_tool("read_file")])

        def rewrite(tool_name, args, context):
            args["sql"] = "REWRITTEN"
            return args

        wrapped_count = apply_tool_transformers(node, {"execute_sql": [rewrite]})

        assert wrapped_count == 1
        await node.tools[0].on_invoke_tool(None, json.dumps({"sql": "x"}))
        assert json.loads(record[0]) == {"sql": "REWRITTEN"}

    def test_category_wildcard_uses_registry(self):
        node = _make_node(
            [_make_tool("execute_sql"), _make_tool("read_file")],
            registry_map={"execute_sql": "db_tools", "read_file": "filesystem_tools"},
        )
        wrapped_count = apply_tool_transformers(node, {"db_tools.*": [lambda n, a, c: a]})
        assert wrapped_count == 1

    def test_skips_proxied_tools(self):
        node = _make_node([_make_tool("execute_sql")], proxied={"execute_sql"})
        wrapped_count = apply_tool_transformers(node, {"execute_sql": [lambda n, a, c: a]})
        assert wrapped_count == 0

    def test_empty_mapping_is_noop(self):
        tool = _make_tool("execute_sql")
        node = _make_node([tool])
        assert apply_tool_transformers(node, {}) == 0
        assert node.tools[0] is tool

    def test_non_function_tools_preserved(self):
        sentinel = object()
        node = _make_node([sentinel, _make_tool("execute_sql")])
        wrapped_count = apply_tool_transformers(node, {"execute_sql": [lambda n, a, c: a]})
        assert wrapped_count == 1
        assert node.tools[0] is sentinel

    @pytest.mark.asyncio
    async def test_reapply_skips_already_wrapped_tool(self):
        # A second pass over the same node (e.g. after a tool-list rebuild reset
        # the node's applied flag) must not re-wrap an already-wrapped tool, or
        # the transformers would run twice per call.
        record = []
        calls = []
        node = _make_node([_make_tool("execute_sql", record)])

        def rewrite(tool_name, args, context):
            calls.append(1)
            args["sql"] = args.get("sql", "") + "|x"
            return args

        assert apply_tool_transformers(node, {"execute_sql": [rewrite]}) == 1
        wrapped = node.tools[0]
        # Re-run: the tool is already transformed, so nothing new is wrapped.
        assert apply_tool_transformers(node, {"execute_sql": [rewrite]}) == 0
        assert node.tools[0] is wrapped

        await node.tools[0].on_invoke_tool(None, json.dumps({"sql": "base"}))
        # Transformer ran exactly once — no double-wrapping.
        assert calls == [1]
        assert json.loads(record[0]) == {"sql": "base|x"}

    @pytest.mark.asyncio
    async def test_context_carries_node_fields(self):
        seen = {}

        def transformer(tool_name, args, context):
            seen.update(context)
            return args

        node = _make_node([_make_tool("execute_sql")])
        apply_tool_transformers(node, {"execute_sql": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen["node_name"] == "chat"
        assert seen["principal"] == {"tenant": {"id": "t1"}}
        assert seen["project_root"] == "/proj"
        assert seen["agent_config"] is node.agent_config
        assert seen["metric_datasets"] is None

    @pytest.mark.asyncio
    async def test_context_carries_metric_datasets_from_semantic_tools(self):
        seen = {}

        def transformer(tool_name, args, context):
            seen.update(context)
            return args

        node = _make_node([_make_tool("query_metrics")])
        node.semantic_tools = SimpleNamespace(metric_datasets=lambda: {"revenue": ["orders"]})
        apply_tool_transformers(node, {"query_metrics": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen["metric_datasets"] == {"revenue": ["orders"]}

    @pytest.mark.asyncio
    async def test_metric_datasets_read_fresh_per_call(self):
        seen = []
        catalog = {"revenue": ["orders"]}

        def transformer(tool_name, args, context):
            seen.append(context["metric_datasets"])
            return args

        node = _make_node([_make_tool("query_metrics")])
        node.semantic_tools = SimpleNamespace(metric_datasets=lambda: dict(catalog))
        apply_tool_transformers(node, {"query_metrics": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")
        catalog["signups"] = ["users"]
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen == [{"revenue": ["orders"]}, {"revenue": ["orders"], "signups": ["users"]}]

    @pytest.mark.asyncio
    async def test_metric_datasets_of_unexpected_type_is_dropped(self):
        seen = {}

        def transformer(tool_name, args, context):
            seen.update(context)
            return args

        node = _make_node([_make_tool("query_metrics")])
        node.semantic_tools = SimpleNamespace(metric_datasets=lambda: "not a mapping")
        apply_tool_transformers(node, {"query_metrics": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen["metric_datasets"] is None

    @pytest.mark.asyncio
    async def test_unreadable_metric_catalog_yields_none(self):
        seen = {}

        def transformer(tool_name, args, context):
            seen.update(context)
            return args

        def boom():
            raise RuntimeError("catalog unavailable")

        node = _make_node([_make_tool("query_metrics")])
        node.semantic_tools = SimpleNamespace(metric_datasets=boom)
        apply_tool_transformers(node, {"query_metrics": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen["metric_datasets"] is None

    @pytest.mark.asyncio
    async def test_principal_read_fresh_per_call(self):
        seen = []

        def transformer(tool_name, args, context):
            seen.append(context["principal"])
            return args

        node = _make_node([_make_tool("execute_sql")])
        apply_tool_transformers(node, {"execute_sql": [transformer]})
        await node.tools[0].on_invoke_tool(None, "{}")
        node.db_func_tool.principal = {"tenant": {"id": "t2"}}
        await node.tools[0].on_invoke_tool(None, "{}")

        assert seen == [{"tenant": {"id": "t1"}}, {"tenant": {"id": "t2"}}]

    @pytest.mark.asyncio
    async def test_multiple_patterns_accumulate_on_one_tool(self):
        record = []
        node = _make_node([_make_tool("execute_sql", record)], registry_map={"execute_sql": "db_tools"})

        def first(tool_name, args, context):
            args["sql"] = args["sql"] + "|byname"
            return args

        def second(tool_name, args, context):
            args["sql"] = args["sql"] + "|bycat"
            return args

        wrapped_count = apply_tool_transformers(node, {"execute_sql": [first], "db_tools.*": [second]})
        assert wrapped_count == 1
        await node.tools[0].on_invoke_tool(None, json.dumps({"sql": "base"}))
        assert json.loads(record[0])["sql"].startswith("base|")
        assert set(json.loads(record[0])["sql"].split("|")[1:]) == {"byname", "bycat"}
