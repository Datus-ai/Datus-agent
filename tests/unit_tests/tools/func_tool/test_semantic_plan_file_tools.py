# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for converting optional planning intent into a reviewable graph."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools


@pytest.mark.parametrize("structured", [False, True])
def test_writes_reviewable_plan_without_semantic_model_mutation(tmp_path, structured):
    tool = SemanticPlanFileTools(tmp_path)
    plan = {
        "summary": "orders -> revenue",
        "datasets": [
            {
                "name": "orders",
                "action": "create",
                "source_kind": "physical",
                "source": "catalog.sales.orders",
                "reason": "order amount",
            },
            {
                "name": "customers",
                "action": "reuse",
                "source_kind": "physical",
                "source": "catalog.sales.customers",
            },
            {
                "name": "refund_rows",
                "action": "create",
                "source_kind": "query",
                "source_tables": ["catalog.sales.refunds", "catalog.sales.orders"],
                "rowset_semantics": "one row per valid refund",
                "native_gap": "requires row-level deduplication",
            },
        ],
        "field_decisions": [{"dataset": "orders", "field": "order_month", "role": "time", "group_by": True}],
        "relationships": [
            {
                "name": "orders_customer",
                "action": "create",
                "from_dataset": "orders",
                "to_dataset": "customers",
                "from_fields": ["customer_id"],
                "to_fields": ["customer_id"],
                "reason": "declared key",
            }
        ],
        "metrics": [
            {"name": "order_amount", "action": "create", "kind": "atomic", "dataset": "orders"},
            {"name": "refund_amount", "action": "create", "kind": "atomic", "dataset": "refund_rows"},
            {
                "name": "revenue",
                "action": "create",
                "kind": "compose",
                "role": "business_output",
                "depends_on": ["order_amount", "refund_amount"],
            },
        ],
    }

    result = tool.write_semantic_model_plan_file("sales_model", plan if structured else json.dumps(plan))

    assert result.success == 1
    assert re.fullmatch(
        r"\.datus/semantic-model-plans/[0-9a-f]{32}/semantic-model-plan\.json",
        result.result["path"],
    )
    saved = tmp_path / result.result["path"]
    graph = json.loads(saved.read_text())
    assert graph["version"] == 1
    assert graph["model"] == "sales_model"
    assert graph["plan"] == {"id": saved.parent.name, "revision": 1, "summary": plan["summary"]}
    assert [layer["kind"] for layer in graph["layers"]] == [
        "physical_table",
        "dataset",
        "metric_atomic",
        "metric_derived",
    ]
    assert [node["id"] for node in graph["nodes"]] == [
        "table:catalog.sales.orders",
        "table:catalog.sales.customers",
        "table:catalog.sales.refunds",
        "dataset:orders",
        "dataset:customers",
        "dataset:refund_rows",
        "metric:order_amount",
        "metric:refund_amount",
        "metric:revenue",
    ]
    assert graph["nodes"][3]["detail"]["fields"] == [{"field": "order_month", "role": "time", "group_by": True}]
    assert graph["nodes"][5]["detail"]["native_gap"] == "requires row-level deduplication"
    assert [node["detail"]["metric_kind"] for node in graph["nodes"][6:]] == ["atomic", "atomic", "compose"]
    assert [(edge["kind"], edge["source"], edge["target"]) for edge in graph["edges"]] == [
        ("reads_table", "table:catalog.sales.orders", "dataset:orders"),
        ("reads_table", "table:catalog.sales.customers", "dataset:customers"),
        ("reads_table", "table:catalog.sales.refunds", "dataset:refund_rows"),
        ("reads_table", "table:catalog.sales.orders", "dataset:refund_rows"),
        ("aggregates", "dataset:orders", "metric:order_amount"),
        ("aggregates", "dataset:refund_rows", "metric:refund_amount"),
        ("compose_member", "metric:order_amount", "metric:revenue"),
        ("compose_member", "metric:refund_amount", "metric:revenue"),
        ("join", "dataset:orders", "dataset:customers"),
    ]
    assert result.result["counts"] == {"datasets": 3, "relationships": 1, "metrics": 3}
    assert not (tmp_path / "subject").exists()


def test_rejects_unsafe_name_and_invalid_json_without_overwriting(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    assert tool.write_semantic_model_plan_file("../subject", "{}").success == 0
    assert tool.write_semantic_model_plan_file("sales_model", "not json").success == 0
    assert tool.write_semantic_model_plan_file("sales_model", "[]").success == 0
    assert json.loads(saved.read_text())["plan"]["summary"] == "first"


def test_projection_is_deterministic_for_the_same_plan(tmp_path):
    plan = json.dumps(
        {
            "summary": "monthly revenue",
            "datasets": [{"name": "orders", "action": "create", "source_kind": "physical", "source": "sales.orders"}],
            "metrics": [
                {"name": "revenue", "action": "create", "kind": "atomic", "dataset": "orders"},
                {"name": "rolling_revenue", "action": "create", "kind": "window", "depends_on": ["revenue"]},
            ],
        }
    )
    graphs = []
    for directory in (tmp_path / "first", tmp_path / "second"):
        result = SemanticPlanFileTools(directory).write_semantic_model_plan_file("sales_model", plan)
        assert result.success == 1
        graphs.append(json.loads((directory / result.result["path"]).read_text()))

    assert graphs[0]["layers"] == graphs[1]["layers"]
    assert graphs[0]["nodes"] == graphs[1]["nodes"]
    assert graphs[0]["edges"] == graphs[1]["edges"]
    assert graphs[0]["nodes"][-1]["detail"]["metric_kind"] == "window"
    assert graphs[0]["edges"][-1]["kind"] == "window_base"


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_rejects_non_finite_numbers_without_overwriting(tmp_path, number):
    tool = SemanticPlanFileTools(tmp_path)
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    result = tool.write_semantic_model_plan_file("sales_model", f'{{"score":{number}}}')

    assert result.success == 0
    assert "valid JSON" in result.error
    assert json.loads(saved.read_text())["plan"]["summary"] == "first"


def test_revisions_reuse_plan_id_but_different_models_do_not(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    first = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    revision = tool.write_semantic_model_plan_file("sales_model", '{"summary":"revised"}')
    other = tool.write_semantic_model_plan_file("inventory_model", '{"summary":"other"}')

    assert revision.result["path"] == first.result["path"]
    assert other.result["path"] != first.result["path"]
    assert json.loads((tmp_path / first.result["path"]).read_text())["plan"] == {
        "id": (tmp_path / first.result["path"]).parent.name,
        "revision": 2,
        "summary": "revised",
    }
    assert json.loads((tmp_path / other.result["path"]).read_text())["plan"]["summary"] == "other"


@pytest.mark.parametrize(
    "invalid_plan",
    [
        {"summary": "x", "datasets": [{"name": "orders"}, {"name": "orders"}]},
        {"summary": "x", "datasets": [{"name": "orders"}], "field_decisions": [{"dataset": "missing", "field": "id"}]},
        {
            "summary": "x",
            "datasets": [{"name": "orders"}],
            "field_decisions": [{"dataset": "orders", "field": "id"}, {"dataset": "orders", "field": "id"}],
        },
        {"summary": "x", "metrics": [{"name": "revenue", "depends_on": ["missing"]}]},
        {"summary": "x", "relationships": [{"name": "join", "from_dataset": "left", "to_dataset": "right"}]},
        {"summary": "x", "datasets": "orders"},
        {"summary": "x", "unsupported_section": []},
        {"summary": "x", "datasets": [{"name": "rows", "source_kind": "query", "source": "SELECT 1"}]},
        {"summary": "x", "datasets": [{"name": "rows", "action": "create", "source_kind": "query"}]},
        {"summary": "x", "datasets": [{"name": "orders", "action": "create", "source_kind": "physical"}]},
    ],
)
def test_rejects_graph_references_and_shapes_without_overwriting(tmp_path, invalid_plan):
    tool = SemanticPlanFileTools(tmp_path)
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    result = tool.write_semantic_model_plan_file("sales_model", json.dumps(invalid_plan))

    assert result.success == 0
    assert json.loads(saved.read_text())["plan"]["summary"] == "first"
    assert tool.write_semantic_model_plan_file("sales_model", '{"summary":"second"}').result["revision"] == 2


def test_concurrent_first_writes_share_plan_id(tmp_path):
    tool = SemanticPlanFileTools(tmp_path)
    start = Barrier(2)

    def write(summary):
        start.wait(timeout=2)
        return tool.write_semantic_model_plan_file("sales_model", {"summary": summary})

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(write, ("first", "second"))

    assert first.success == second.success == 1
    assert first.result["path"] == second.result["path"]
    assert {first.result["revision"], second.result["revision"]} == {1, 2}
    revision = tool.write_semantic_model_plan_file("sales_model", {"summary": "revised"})
    assert revision.result["path"] == first.result["path"]
    assert revision.result["revision"] == 3


def test_rejects_symlinked_plan_directory_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    plan_parent = workspace / ".datus"
    plan_parent.mkdir(parents=True)
    outside.mkdir()
    (plan_parent / "semantic-model-plans").symlink_to(outside, target_is_directory=True)

    result = SemanticPlanFileTools(workspace).write_semantic_model_plan_file("sales_model", "{}")

    assert result.success == 0
    assert not (outside / "sales_model").exists()


def test_direct_cli_write_and_later_instance_share_plan_identity(tmp_path):
    initial = SemanticPlanFileTools(tmp_path).write_semantic_model_plan_file("sales", {"summary": "first"})
    revised = SemanticPlanFileTools(tmp_path).write_semantic_model_plan_file("sales", {"summary": "second"})
    assert revised.result["path"] == initial.result["path"]
    assert revised.result["revision"] == 2
    assert revised.result["review_status"] == "applied"


def _commit_plan_in_worker(root, prepared, started, results):
    started.set()
    result = SemanticPlanFileTools(root).commit(prepared)
    results.put((result.success, result.error))


def test_commit_waits_for_another_process_and_rechecks_baseline(tmp_path):
    import multiprocessing

    from datus.tools.func_tool.semantic_plan_file_tools import fcntl

    if fcntl is None:
        pytest.skip("Cross-process advisory locks require POSIX")

    writer = SemanticPlanFileTools(tmp_path)
    saved = writer.write_semantic_model_plan_file("sales", {"summary": "accepted"})
    prepared = writer.prepare("sales", {"summary": "proposal"})
    target = tmp_path / saved.result["path"]
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    results = context.Queue()
    worker = context.Process(target=_commit_plan_in_worker, args=(tmp_path, prepared, started, results))
    with target.with_name(f".{target.name}.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        worker.start()
        try:
            assert started.wait(10)
            worker.join(0.2)
            assert worker.is_alive()
            assert target.read_text() == prepared.old_text
            target.write_text("concurrent worker content")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    try:
        success, error = results.get(timeout=10)
        worker.join(10)
        assert worker.exitcode == 0
        assert success == 0
        assert "changed during review" in error
        assert target.read_text() == "concurrent worker content"
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(10)
        results.close()


def test_first_proposals_from_separate_sessions_share_one_accepted_baseline(tmp_path):
    first_writer = SemanticPlanFileTools(tmp_path)
    second_writer = SemanticPlanFileTools(tmp_path)
    first = first_writer.prepare("sales", {"summary": "one"})
    second = second_writer.prepare("sales", {"summary": "two"})
    assert first.old_text is second.old_text is None
    assert first.path == second.path
    assert first_writer.commit(first).success == 1
    rejected = second_writer.commit(second)
    assert rejected.success == 0
    assert "changed during review" in rejected.error
    assert [p.read_text() for p in tmp_path.rglob("semantic-model-plan.json")] == [first.new_text]
    feedback = second_writer.prepare("sales", {"summary": "feedback"})
    assert feedback.path == first.path
    assert feedback.old_text == first.new_text
    assert feedback.revision == 2
    assert second_writer.commit(feedback).success == 1


def test_existing_random_plan_identity_is_preserved(tmp_path):
    from uuid import uuid4

    writer = SemanticPlanFileTools(tmp_path)
    prepared = writer.prepare("sales", {"summary": "existing"})
    legacy_id = uuid4().hex
    legacy_path = tmp_path / ".datus/semantic-model-plans" / legacy_id / "semantic-model-plan.json"
    graph = json.loads(prepared.new_text)
    graph["plan"]["id"] = legacy_id
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps(graph))
    updated = SemanticPlanFileTools(tmp_path).prepare("sales", {"summary": "feedback"})
    assert updated.path == legacy_path.relative_to(tmp_path).as_posix()
    assert updated.old_text == legacy_path.read_text()
    assert updated.revision == 2
    assert writer.commit(updated).success == 1


def test_commit_rejects_another_plan_identity_accepted_for_the_same_model(tmp_path):
    from dataclasses import replace
    from uuid import uuid4

    writer = SemanticPlanFileTools(tmp_path)
    pending = writer.prepare("sales", {"summary": "pending before identity convergence"})
    legacy_id = uuid4().hex
    graph = json.loads(pending.new_text)
    graph["plan"]["id"] = legacy_id
    old_pending = replace(
        pending,
        path=f".datus/semantic-model-plans/{legacy_id}/semantic-model-plan.json",
        new_text=json.dumps(graph),
    )
    assert writer.commit(pending).success == 1
    assert writer.commit(old_pending).success == 0
    assert [p.read_text() for p in tmp_path.rglob("semantic-model-plan.json")] == [pending.new_text]
