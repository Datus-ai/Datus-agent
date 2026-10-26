# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Cross-node regressions for Web file mutation version relationships."""

from pathlib import Path

import pytest
import yaml

from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.semantic_tools.metric_identity import ensure_metric_uid
from tests.unit_tests.tools.func_tool.test_metric_filesystem_tools import (
    _osi_metric,
)
from tests.unit_tests.tools.func_tool.test_metric_filesystem_tools import (
    identity_tool as original_identity_tool,
)


@pytest.fixture(params=["sql_summary", "semantic"])
def authoring_file(tmp_path, request):
    if request.param == "sql_summary":
        node = SqlSummaryAgenticNode.__new__(SqlSummaryAgenticNode)
        node.knowledge_base_dir = str(tmp_path / "subject")
        node._mutation_snapshots = {}
        node._mutated_files = set()
        node._web_mutation_revisions = {}
        target = tmp_path / "subject/sql_summaries/query.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("sql: SELECT 1\ndescription: original\n")
        tool = FilesystemFuncTool(
            root_path=str(tmp_path),
            mutation_guard=node._sql_summary_mutation_guard,
            mutation_callback=node._record_sql_summary_mutation,
            client_mutation_guard=node._validate_sql_summary_mutation_path,
            client_mutation_callback=node._record_sql_summary_client_mutation,
        )
        node.filesystem_func_tool = tool
        return (
            tool,
            target,
            node._rollback_sql_summary_mutations,
            [("SELECT 1", "SELECT 2"), ("description: original", "description: generated")],
        )
    metric = _osi_metric("revenue", "SUM(orders.amount)")
    ensure_metric_uid(metric)
    # Bypass only the fixture decorator; do not install its mocked validator.
    tool, target, state, _ = original_identity_tool.__wrapped__(tmp_path, None)([metric])
    document = yaml.safe_load(target.read_text())
    document["semantic_model"][0]["datasets"][0]["fields"] = [
        {"name": name, "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": name}]}}
        for name in ["amount", "total", "final_total"]
    ]
    target.write_text(yaml.safe_dump(document, sort_keys=False))
    state.record_selected_write()
    assert tool._validate_osi_document(target.read_text()) is None
    return (
        tool,
        target,
        tool.rollback_failed_authoring,
        [("Definition for revenue", "New description"), ("SUM(orders.amount)", "SUM(orders.total)")],
    )


def prepare(tool, target, replacement):
    old, new = replacement
    return tool._prepare_client_mutation("edit_file", {"path": str(target), "old_string": old, "new_string": new})


def execute(target: Path, mutation) -> None:
    args = mutation.arguments
    content = target.read_text()
    assert content.count(args["old_string"]) == 1
    target.write_text(content.replace(args["old_string"], args["new_string"], 1))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("delayed_results", [False, True])
def test_prepared_edits_preserve_owned_rollback_across_execution_and_report_orders(
    authoring_file, reverse, delayed_results
):
    tool, target, rollback, replacements = authoring_file
    original = target.read_bytes()
    mutations = [prepare(tool, target, replacement) for replacement in replacements]
    execution_order = list(reversed(mutations)) if reverse else mutations
    results = []
    for mutation in execution_order:
        execute(target, mutation)
        if not delayed_results:
            results.append(mutation.complete())
    if delayed_results:
        # Both WebSocket writes have completed; acknowledge them in reverse
        # order to model independently posted tool_result requests.
        results = [mutation.complete() for mutation in reversed(execution_order)]
    assert results == [None] * len(mutations)
    assert all(new in target.read_text() for _, new in replacements)
    rollback()
    assert target.read_bytes() == original


def test_external_save_between_successful_operations_never_reuses_the_old_snapshot(authoring_file):
    tool, target, rollback, replacements = authoring_file
    first = prepare(tool, target, replacements[0])
    execute(target, first)
    assert first.complete() is None
    user_content = target.read_text() + "# Saved independently in the IDE\n"
    target.write_text(user_content)
    second = prepare(tool, target, replacements[1])
    execute(target, second)
    assert second.complete() is None
    rollback()
    assert "# Saved independently in the IDE\n" in target.read_text()


def test_canceling_an_unused_prepared_edit_keeps_the_successful_edit_rollback_owned(authoring_file):
    tool, target, rollback, replacements = authoring_file
    original = target.read_bytes()
    first, canceled = [prepare(tool, target, replacement) for replacement in replacements]
    execute(target, first)
    assert first.complete() is None
    canceled.cancel()
    rollback()
    assert target.read_bytes() == original


def test_external_save_during_a_prepared_batch_is_preserved(authoring_file):
    tool, target, rollback, replacements = authoring_file
    first, second = [prepare(tool, target, replacement) for replacement in replacements]
    execute(target, first)
    assert first.complete() is None
    target.write_text(target.read_text() + "# User save between prepared cards\n")
    execute(target, second)
    assert second.complete() is None
    rollback()
    assert "# User save between prepared cards\n" in target.read_text()


def test_write_delete_and_recreate_keep_distinct_version_batches():
    from datus.tools.func_tool.client_file_mutations import ClientMutationHistory

    original = (b"original", 0o600)
    history = ClientMutationHistory(original)
    write = history.prepare(original, "write_file", {"content": "created"})
    delete = history.prepare(original, "delete_file", {})
    assert history.complete(write, b"created") == "owned"
    assert history.complete(delete, None) == "owned"
    assert delete.before == original
    recreated = history.prepare(None, "write_file", {"content": "recreated"})
    assert history.complete(recreated, b"recreated") == "owned"
    assert recreated.before is None


def test_unacknowledged_changes_are_not_owned_until_their_result_arrives():
    from datus.tools.func_tool.client_file_mutations import ClientMutationHistory

    original = (b"first=old;second=old", 0o600)
    history = ClientMutationHistory(original)
    first = history.prepare(original, "edit_file", {"old_string": "first=old", "new_string": "first=new"})
    second = history.prepare(original, "edit_file", {"old_string": "second=old", "new_string": "second=new"})
    final = b"first=new;second=new"
    assert history.complete(second, final) == "pending"
    assert history.complete(first, final) == "owned"
    assert first.before == original


def test_canceled_unknown_revision_is_not_adopted_as_owned():
    from datus.tools.func_tool.client_file_mutations import ClientMutationHistory

    original = (b"original", 0o600)
    history = ClientMutationHistory(original)
    pending = history.prepare(original, "write_file", {"content": "proposed"})
    assert history.classify(b"unacknowledged or user content") == "pending"
    history.cancel(pending)
    assert history.classify(b"user content") == "external"


def test_a_larger_prepared_batch_keeps_its_successful_cumulative_edits_owned():
    from datus.tools.func_tool.client_file_mutations import ClientMutationHistory

    original = (";".join(f"slot{i:02d}=old" for i in range(12)).encode(), 0o600)
    history = ClientMutationHistory(original)
    records = [
        history.prepare(original, "edit_file", {"old_string": f"slot{i:02d}=old", "new_string": f"slot{i:02d}=new"})
        for i in range(12)
    ]
    current = original[0]
    for record in records:
        current = current.replace(record.arguments["old_string"].encode(), record.arguments["new_string"].encode(), 1)
        assert history.complete(record, current) == "owned"
    assert current.count(b"=new") == 12
    assert records[-1].before == original


@pytest.mark.parametrize("authoring_file", ["semantic"], indirect=True)
def test_native_semantic_authoring_between_web_batches_remains_owned(authoring_file):
    tool, target, rollback, _ = authoring_file
    original = target.read_bytes()
    first = prepare(tool, target, ("SUM(orders.amount)", "SUM(orders.total)"))
    execute(target, first)
    assert first.complete() is None
    metric = yaml.safe_load(target.read_text())["semantic_model"][0]["metrics"][0]
    metric["description"] = "Native guarded authoring"
    assert tool.upsert_osi_metrics(str(target), [metric]).success == 1
    second = prepare(tool, target, ("SUM(orders.total)", "SUM(orders.final_total)"))
    execute(target, second)
    assert second.complete() is None
    rollback()
    assert target.read_bytes() == original


@pytest.mark.parametrize("authoring_file", ["semantic"], indirect=True)
def test_new_request_does_not_inherit_previous_external_ownership(authoring_file):
    from tests.unit_tests.tools.func_tool.test_metric_filesystem_tools import _bound_state

    tool, target, rollback, replacements = authoring_file
    first = prepare(tool, target, replacements[0])
    execute(target, first)
    assert first.complete() is None
    target.write_text(target.read_text() + "# User save\n")
    second = prepare(tool, target, replacements[1])
    execute(target, second)
    assert second.complete() is None
    rollback()
    user_version = target.read_bytes()
    state = tool.osi_target_state
    state.reset()
    tool._reset_client_mutations()
    state.select(_bound_state(target).selected, mode="bound")
    third = prepare(tool, target, ("SUM(orders.total)", "SUM(orders.final_total)"))
    execute(target, third)
    assert third.complete() is None
    assert rollback() is True
    assert target.read_bytes() == user_version
