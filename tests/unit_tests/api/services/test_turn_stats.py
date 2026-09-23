"""Tests for datus.api.services.turn_stats — per-turn subagent / tool tallies."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from datus.api.services.turn_stats import TurnStatsCollector, resolve_subagent
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

T0 = datetime(2026, 9, 23, 10, 0, 0)


def _start(call_id, tool, *, depth=0, parent=None, arguments=None, at=0):
    return ActionHistory(
        action_id=call_id,
        role=ActionRole.TOOL,
        action_type=tool,
        messages="",
        input={"function_name": tool, "arguments": arguments or {}},
        output={},
        status=ActionStatus.PROCESSING,
        start_time=T0 + timedelta(seconds=at),
        depth=depth,
        parent_action_id=parent,
    )


def _complete(call_id, tool, *, ok=True, depth=0, parent=None, start=0, end=1):
    return ActionHistory(
        action_id="complete_" + call_id,
        role=ActionRole.TOOL,
        action_type=tool,
        messages="",
        input={},
        output={"success": ok},
        status=ActionStatus.SUCCESS if ok else ActionStatus.FAILED,
        start_time=T0 + timedelta(seconds=start),
        end_time=T0 + timedelta(seconds=end),
        depth=depth,
        parent_action_id=parent,
    )


def _resolver(name):
    return {"my_sales_bot": ("gen_sql", "custom")}.get(name, (name, "builtin"))


def _by_tool(tools):
    return {(t.name, t.caller): t for t in tools}


class TestTurnStatsCollector:
    def test_pairs_tool_calls_and_splits_success_from_failure(self):
        collector = TurnStatsCollector(_resolver)
        for action in (
            _start("c1", "execute_sql"),
            _complete("c1", "execute_sql", ok=True, end=2),
            _start("c2", "execute_sql", at=3),
            _complete("c2", "execute_sql", ok=False, start=3, end=4),
            _start("c3", "list_tables", at=5),
            _complete("c3", "list_tables", start=5, end=5),
        ):
            collector.observe(action)

        subagents, tools = collector.finalize("completed", 6000)

        assert subagents == []
        tools = _by_tool(tools)
        sql = tools[("execute_sql", "main")]
        assert (sql.calls, sql.success, sql.failed, sql.interrupted) == (2, 1, 1, 0)
        assert sql.duration_ms == 3000
        assert tools[("list_tables", "main")].success == 1

    def test_ignores_non_tool_actions_and_duplicate_starts(self):
        collector = TurnStatsCollector(_resolver)
        collector.observe(
            ActionHistory(
                action_id="r1",
                role=ActionRole.ASSISTANT,
                action_type="response",
                messages="hi",
                input={},
                output={},
                status=ActionStatus.PROCESSING,
            )
        )
        collector.observe(_start("c1", "read_file"))
        collector.observe(_start("c1", "read_file"))
        collector.observe(_complete("c1", "read_file"))

        _, tools = collector.finalize("completed", 10)

        assert len(tools) == 1
        assert tools[0].calls == 1

    def test_dispatched_subagent_reports_root_type_and_nested_tools(self):
        collector = TurnStatsCollector(_resolver)
        for action in (
            _start("t1", "task", arguments=json.dumps({"type": "my_sales_bot", "prompt": "q"})),
            _start("n1", "describe_table", depth=1, parent="t1", at=1),
            _complete("n1", "describe_table", depth=1, parent="t1", start=1, end=2),
            _start("n2", "execute_sql", depth=1, parent="t1", at=2),
            _complete("n2", "execute_sql", ok=False, depth=1, parent="t1", start=2, end=3),
            _complete("t1", "task", start=0, end=10),
        ):
            collector.observe(action)

        subagents, tools = collector.finalize("completed", 11000)

        assert len(subagents) == 1
        sub = subagents[0]
        assert (sub.name, sub.kind, sub.entry) == ("gen_sql", "custom", "dispatch")
        assert (sub.calls, sub.success, sub.failed, sub.interrupted) == (1, 1, 0, 0)
        assert sub.tool_calls == 2
        assert sub.duration_ms == 10000
        tools = _by_tool(tools)
        assert tools[("task", "main")].success == 1
        assert tools[("execute_sql", "subagent")].failed == 1
        assert tools[("describe_table", "subagent")].success == 1

    def test_failed_task_counts_as_failed_subagent(self):
        collector = TurnStatsCollector(_resolver)
        collector.observe(_start("t1", "task", arguments={"type": "gen_report"}))
        collector.observe(_complete("t1", "task", ok=False))

        subagents, _ = collector.finalize("completed", 1000)

        assert (subagents[0].name, subagents[0].kind) == ("gen_report", "builtin")
        assert subagents[0].failed == 1

    def test_unfinished_calls_are_interrupted_not_dropped(self):
        collector = TurnStatsCollector(_resolver)
        for action in (
            _start("t1", "task", arguments={"type": "ask_metrics"}),
            _start("n1", "query_metrics", depth=1, parent="t1"),
            _start("c1", "write_file"),
        ):
            collector.observe(action)

        subagents, tools = collector.finalize("cancelled", 5000)

        assert (subagents[0].calls, subagents[0].interrupted, subagents[0].tool_calls) == (1, 1, 1)
        tools = _by_tool(tools)
        for key in (("task", "main"), ("query_metrics", "subagent"), ("write_file", "main")):
            assert (tools[key].calls, tools[key].interrupted) == (1, 1)

    def test_task_without_type_argument_is_counted_as_tool_only(self):
        collector = TurnStatsCollector(_resolver)
        collector.observe(_start("t1", "task", arguments="not json"))
        collector.observe(_complete("t1", "task"))

        subagents, tools = collector.finalize("completed", 10)

        assert subagents == []
        assert tools[0].name == "task"

    def test_direct_entry_records_one_invocation_per_turn_status(self):
        for status, field in (("completed", "success"), ("error", "failed"), ("cancelled", "interrupted")):
            collector = TurnStatsCollector(_resolver)
            collector.observe(_start("c1", "query_metrics"))
            collector.observe(_complete("c1", "query_metrics"))

            subagents, _ = collector.finalize(status, 4200, direct=("ask_metrics", "custom"))

            sub = subagents[0]
            assert (sub.name, sub.kind, sub.entry, sub.calls) == ("ask_metrics", "custom", "direct", 1)
            assert getattr(sub, field) == 1
            assert sub.tool_calls == 1
            assert sub.duration_ms == 4200


class TestResolveSubagent:
    def _config(self):
        return SimpleNamespace(
            agentic_nodes={
                "gen_sql": {"id": None},
                "sales_bot": {"id": "sa-1", "node_class": "ask_metrics"},
                "legacy_bot": {"id": "sa-2", "type": "gen_sql"},
                "semantic_bot": {"id": "sa-3", "node_class": "gen_metrics"},
                "bare_bot": {"id": "sa-4"},
            }
        )

    def test_builtins_report_under_their_own_name(self):
        config = self._config()
        assert resolve_subagent(config, "gen_sql") == ("gen_sql", "builtin")
        assert resolve_subagent(config, "gen_visual_report") == ("gen_visual_report", "builtin")
        assert resolve_subagent(config, "explore") == ("explore", "builtin")

    def test_retired_builtin_maps_to_semantic_modeling(self):
        assert resolve_subagent(self._config(), "gen_metrics") == ("semantic_modeling", "builtin")

    def test_custom_by_name_or_id_reports_root_type(self):
        config = self._config()
        assert resolve_subagent(config, "sales_bot") == ("ask_metrics", "custom")
        assert resolve_subagent(config, "sa-1") == ("ask_metrics", "custom")
        assert resolve_subagent(config, "sa-2") == ("gen_sql", "custom")
        assert resolve_subagent(config, "semantic_bot") == ("semantic_modeling", "custom")

    def test_unknown_or_classless_custom_falls_back_to_gen_sql(self):
        config = self._config()
        assert resolve_subagent(config, "bare_bot") == ("gen_sql", "custom")
        assert resolve_subagent(config, "ghost") == ("gen_sql", "custom")
        assert resolve_subagent(SimpleNamespace(agentic_nodes=None), "ghost") == ("gen_sql", "custom")
