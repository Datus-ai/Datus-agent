# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for converting optional planning intent into a reviewable graph."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from datus.tools.func_tool.semantic_plan_file_tools import SemanticPlanFileTools
from tests.unit_tests.semantic_plan_support import plan_writer, select_plan_model

MODELS = Path(__file__).resolve().parents[3] / "data/semantic_models"

SALES_YAML = """\
version: 0.2.0.dev0
name: sales
datasets:
  - name: orders
    source: main.orders
    fields:
      - name: amount
        expression:
          dialects: [{dialect: ANSI_SQL, expression: amount}]
      - name: currency
        expression:
          dialects: [{dialect: ANSI_SQL, expression: currency}]
metrics:
  - name: revenue
    description: Existing revenue calculation
    expression:
      dialects: [{dialect: ANSI_SQL, expression: SUM(orders.amount)}]
  - name: order_count
    expression:
      dialects: [{dialect: ANSI_SQL, expression: COUNT(orders.amount)}]
"""


@pytest.fixture
def existing_sales(tmp_path):
    path = tmp_path / "subject/semantic_models/sales.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(SALES_YAML)
    return path


def _graph_nodes(graph):
    return {node["id"]: node for node in graph["nodes"]}


@pytest.mark.parametrize("structured", [False, True])
def test_writes_reviewable_plan_without_semantic_model_mutation(tmp_path, structured):
    tool = plan_writer(tmp_path, "sales_model")
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
                "action": "create",
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
    assert graph["plan"] == {
        "id": saved.parent.name,
        "revision": 1,
        "summary": plan["summary"],
        "model_file": "subject/semantic_models/sales_model.yaml",
    }
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
    tool = plan_writer(tmp_path, "sales_model")
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
        result = plan_writer(directory, "sales_model").write_semantic_model_plan_file("sales_model", plan)
        assert result.success == 1
        graphs.append(json.loads((directory / result.result["path"]).read_text()))

    assert graphs[0]["layers"] == graphs[1]["layers"]
    assert graphs[0]["nodes"] == graphs[1]["nodes"]
    assert graphs[0]["edges"] == graphs[1]["edges"]
    assert graphs[0]["nodes"][-1]["detail"]["metric_kind"] == "window"
    assert graphs[0]["edges"][-1]["kind"] == "window_base"


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_rejects_non_finite_numbers_without_overwriting(tmp_path, number):
    tool = plan_writer(tmp_path, "sales_model")
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    result = tool.write_semantic_model_plan_file("sales_model", f'{{"score":{number}}}')

    assert result.success == 0
    assert "valid JSON" in result.error
    assert json.loads(saved.read_text())["plan"]["summary"] == "first"


def test_revisions_reuse_plan_id_but_different_models_do_not(tmp_path):
    tool = plan_writer(tmp_path, "sales_model")
    first = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    revision = tool.write_semantic_model_plan_file("sales_model", '{"summary":"revised"}')
    other = plan_writer(tmp_path, "inventory_model").write_semantic_model_plan_file(
        "inventory_model", '{"summary":"other"}'
    )

    assert revision.result["path"] == first.result["path"]
    assert other.result["path"] != first.result["path"]
    assert json.loads((tmp_path / first.result["path"]).read_text())["plan"] == {
        "id": (tmp_path / first.result["path"]).parent.name,
        "revision": 2,
        "summary": "revised",
        "model_file": "subject/semantic_models/sales_model.yaml",
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
    tool = plan_writer(tmp_path, "sales_model")
    good = tool.write_semantic_model_plan_file("sales_model", '{"summary":"first"}')
    saved = tmp_path / good.result["path"]

    result = tool.write_semantic_model_plan_file("sales_model", json.dumps(invalid_plan))

    assert result.success == 0
    assert json.loads(saved.read_text())["plan"]["summary"] == "first"
    assert tool.write_semantic_model_plan_file("sales_model", '{"summary":"second"}').result["revision"] == 2


def test_concurrent_first_writes_share_plan_id(tmp_path):
    tool = plan_writer(tmp_path, "sales_model")
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

    result = plan_writer(workspace, "sales_model").write_semantic_model_plan_file("sales_model", "{}")

    assert result.success == 0
    assert not (outside / "sales_model").exists()


def test_direct_cli_write_and_later_instance_share_plan_identity(tmp_path):
    initial = plan_writer(tmp_path).write_semantic_model_plan_file("sales", {"summary": "first"})
    revised = plan_writer(tmp_path).write_semantic_model_plan_file("sales", {"summary": "second"})
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

    writer = plan_writer(tmp_path)
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


def test_concurrent_proposals_in_the_same_chat_reject_an_outdated_plan(tmp_path):
    first_writer = plan_writer(tmp_path)
    second_writer = plan_writer(tmp_path)
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


def test_legacy_plan_does_not_become_the_chat_plan(tmp_path):
    from uuid import uuid4

    legacy = SemanticPlanFileTools(tmp_path).prepare("sales", {"summary": "existing"})
    legacy_id = uuid4().hex
    legacy_path = tmp_path / ".datus/semantic-model-plans" / legacy_id / "semantic-model-plan.json"
    graph = json.loads(legacy.new_text)
    graph["plan"]["id"] = legacy_id
    before = json.dumps(graph)
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(before)

    writer = plan_writer(tmp_path)
    updated = writer.prepare("sales", {"summary": "feedback"})

    assert tmp_path / updated.path != legacy_path
    assert updated.old_text is None
    assert updated.revision == 1
    assert writer.commit(updated).success == 1
    assert legacy_path.read_text() == before


def test_different_chats_and_model_files_have_independent_plans(tmp_path):
    first = plan_writer(tmp_path).write_semantic_model_plan_file("sales", {"summary": "first"})
    other_chat = plan_writer(tmp_path, chat_id="chat-2").write_semantic_model_plan_file(
        "sales", {"summary": "another chat"}
    )
    other_file = plan_writer(
        tmp_path, model_path=tmp_path / "subject" / "semantic_models" / "archived-sales.yaml"
    ).write_semantic_model_plan_file("sales", {"summary": "another file"})
    assert first.success == other_chat.success == other_file.success == 1
    assert len({result.result["path"] for result in (first, other_chat, other_file)}) == 3
    assert all(result.result["revision"] == 1 for result in (first, other_chat, other_file))


def test_existing_yaml_seeds_first_plan_and_revision_drops_only_withdrawn_proposal(tmp_path, existing_sales):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    c = {"name": "rolling", "kind": "window", "action": "create", "depends_on": ["revenue"]}
    first = writer.prepare("sales", {"metrics": [c, {"name": "other_proposal", "kind": "atomic", "action": "create"}]})
    revised = writer.prepare(
        "sales",
        {"metrics": [c, {"name": "revenue", "action": "reuse", "definition": "A proposal paraphrase"}]},
    )
    assert first.path == revised.path
    assert revised.revision == 2
    assert first.baseline_text == revised.baseline_text
    old_graph = json.loads(revised.preview()["oldText"])
    new_graph = json.loads(revised.preview()["newText"])
    old_nodes = _graph_nodes(old_graph)
    new_nodes = _graph_nodes(new_graph)
    assert {"metric:revenue", "metric:order_count", "dataset:orders", "table:main.orders"} <= set(old_nodes)
    assert set(new_nodes) - set(old_nodes) == {"metric:rolling"}
    assert {key: new_nodes[key] for key in old_nodes} == old_nodes
    assert "metric:other_proposal" in _graph_nodes(json.loads(first.new_text))
    assert old_graph["edges"] == [edge for edge in new_graph["edges"] if edge["target"] != "metric:rolling"]
    assert [edge for edge in new_graph["edges"] if edge["target"] == "metric:rolling"] == [
        {
            "id": "window_base:metric:revenue->metric:rolling",
            "kind": "window_base",
            "source": "metric:revenue",
            "target": "metric:rolling",
        }
    ]
    assert writer.commit(revised).success == 1
    assert existing_sales.read_text() == SALES_YAML


def test_explicit_delete_and_stale_model_review(tmp_path, existing_sales):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    delete = writer.prepare("sales", {"metrics": [{"name": "order_count", "action": "delete"}]})
    assert "metric:order_count" in _graph_nodes(json.loads(delete.baseline_text))
    assert "metric:order_count" not in _graph_nodes(json.loads(delete.new_text))
    existing_sales.write_text(SALES_YAML + "# concurrent edit\n")
    result = writer.commit(delete)
    assert result.success == 0
    assert "semantic model changed during review" in result.error
    assert not (tmp_path / delete.path).exists()


def test_commit_holds_shared_model_lock_until_publication(tmp_path, existing_sales, monkeypatch):
    from datus.storage.semantic_model.artifact_file import atomic_write_text, model_lock_held
    from datus.tools.func_tool import semantic_plan_file_tools

    writer = plan_writer(tmp_path, model_path=existing_sales)
    prepared = writer.prepare("sales", {"summary": "Publish the reviewed plan"})
    locks_during_publication = []

    def observe_publication(target, content):
        locks_during_publication.append(model_lock_held(existing_sales))
        atomic_write_text(target, content)

    monkeypatch.setattr(semantic_plan_file_tools, "atomic_write_text", observe_publication)
    result = writer.commit(prepared)

    assert result.success == 1
    assert locks_during_publication == [True]
    assert (tmp_path / prepared.path).read_text() == prepared.new_text
    assert existing_sales.read_text() == SALES_YAML


@pytest.mark.parametrize(
    "datasets",
    [
        [],
        [{"name": "orders", "action": "reuse"}],
        [{"name": "orders", "action": "update"}],
        [{"name": "orders", "action": "update", "source_kind": "physical", "source": "main.orders"}],
    ],
)
def test_existing_dataset_field_update_preserves_its_sources_and_metrics(tmp_path, existing_sales, datasets):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    revision = writer.prepare(
        "sales",
        {
            "datasets": datasets,
            "field_decisions": [
                {"dataset": "orders", "field": "amount", "role": "measure"},
                {"dataset": "orders", "field": "order_day", "role": "dimension"},
            ],
        },
    )
    baseline = json.loads(revision.baseline_text)
    candidate = json.loads(revision.new_text)
    assert candidate["edges"] == baseline["edges"]
    old_nodes, new_nodes = _graph_nodes(baseline), _graph_nodes(candidate)
    assert [node_id for node_id in old_nodes if old_nodes[node_id] != new_nodes[node_id]] == ["dataset:orders"]
    assert new_nodes["dataset:orders"]["detail"]["source_tables"] == ["main.orders"]
    assert {field["field"]: field.get("role") for field in new_nodes["dataset:orders"]["detail"]["fields"]} == {
        "amount": "measure",
        "currency": None,
        "order_day": "dimension",
    }


def test_existing_filter_update_keeps_the_same_dependency(tmp_path):
    model = tmp_path / "attribution.yaml"
    model.write_bytes((MODELS / "attribution/model.yaml").read_bytes())
    writer = plan_writer(tmp_path, "attribution_model", model_path=model)
    prepared = writer.prepare(
        "attribution_model",
        {
            "metrics": [
                {
                    "name": "x_above",
                    "action": "update",
                    "kind": "filter",
                    "definition": "Changed description",
                    "depends_on": ["x_total"],
                }
            ],
        },
    )
    baseline = json.loads(prepared.baseline_text)
    proposal = json.loads(prepared.new_text)
    edge_id = "derive_base:metric:x_total->metric:x_above"
    old_edge = next(edge for edge in baseline["edges"] if edge["id"] == edge_id)
    assert [edge for edge in proposal["edges"] if edge["target"] == "metric:x_above"] == [old_edge]
    assert _graph_nodes(proposal)["metric:x_above"]["detail"]["definition"] == "Changed description"


@pytest.mark.parametrize(("key", "name"), [("datasets", "orders"), ("metrics", "revenue")])
def test_rejects_unknown_actions_for_existing_objects(tmp_path, existing_sales, key, name):
    writer = plan_writer(tmp_path, model_path=existing_sales)

    result = writer.prepare("sales", {key: [{"name": name, "action": "upadte"}]})

    assert result.success == 0
    assert f"{name} has an unknown action" in result.error


@pytest.mark.parametrize("move_existing_reader", [False, True])
def test_table_readers_follow_dataset_source_changes(tmp_path, existing_sales, move_existing_reader):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    datasets = [{"name": "orders_daily", "action": "create", "source_kind": "physical", "source": "main.orders"}]
    if move_existing_reader:
        datasets.append(
            {"name": "orders", "action": "update", "source_kind": "query", "source_tables": ["main.raw_orders"]}
        )

    graph = json.loads(writer.prepare("sales", {"datasets": datasets}).new_text)

    nodes = _graph_nodes(graph)
    expected_readers = {"orders_daily"} if move_existing_reader else {"orders_daily", "orders"}
    assert set(nodes["table:main.orders"]["detail"]["datasets"]) == expected_readers
    assert {edge["target"] for edge in graph["edges"] if edge["source"] == "table:main.orders"} == {
        f"dataset:{name}" for name in expected_readers
    }
    expected_source = (
        (None, "query", ["main.raw_orders"]) if move_existing_reader else ("main.orders", "table", ["main.orders"])
    )
    detail = nodes["dataset:orders"]["detail"]
    assert (detail["source"], detail["source_kind"], detail["source_tables"]) == expected_source


def test_existing_join_exposes_review_fields_and_requires_explicit_update(tmp_path):
    model = tmp_path / "aviation.yaml"
    model.write_bytes((MODELS / "aviation/aviation.yaml").read_bytes())
    writer = plan_writer(tmp_path, "aviation", model_path=model)
    prepared = writer.prepare("aviation", {})
    old = json.loads(prepared.baseline_text)
    candidate = json.loads(prepared.new_text)
    edge_id = "join:flights_operated_by"
    edge = next(edge for edge in candidate["edges"] if edge["id"] == edge_id)
    assert edge in old["edges"]
    assert edge["detail"] == {
        "from_fields": ["aircraft_id"],
        "to_fields": ["aircraft_id"],
        "cardinality": "many_to_one",
        "join_kind": "left",
    }
    same_endpoints = {
        "name": "flights_operated_by",
        "action": "update",
        "from_dataset": "flights",
        "to_dataset": "aircraft",
        "join_kind": "inner",
    }
    updated = writer.prepare("aviation", {"relationships": [same_endpoints]})
    updated_edge = next(edge for edge in json.loads(updated.new_text)["edges"] if edge["id"] == edge_id)
    assert updated_edge["detail"] == {**edge["detail"], "join_kind": "inner"}
    same_endpoints["from_fields"] = ["route_id"]
    updated = writer.prepare("aviation", {"relationships": [same_endpoints]})
    updated_edge = next(edge for edge in json.loads(updated.new_text)["edges"] if edge["id"] == edge_id)
    assert updated_edge["detail"] == {**edge["detail"], "join_kind": "inner", "from_fields": ["route_id"]}
    replacement = {
        "name": "flights_operated_by",
        "action": "create",
        "from_dataset": "aircraft",
        "to_dataset": "flights",
    }
    duplicate = writer.prepare("aviation", {"relationships": [replacement]})
    assert duplicate.success == 0
    assert "already exists" in duplicate.error
    replacement["action"] = "update"
    updated = writer.prepare("aviation", {"relationships": [replacement]})
    updated_edge = next(edge for edge in json.loads(updated.new_text)["edges"] if edge["id"] == edge_id)
    assert updated_edge["source"] == "dataset:aircraft"
    assert "from_fields" not in updated_edge["detail"]
    assert "to_fields" not in updated_edge["detail"]


@pytest.mark.parametrize(("name", "action"), [("flights_operated_by", "update"), ("new_join", "create")])
def test_rejects_mismatched_join_columns(tmp_path, name, action):
    model = tmp_path / "aviation.yaml"
    model.write_bytes((MODELS / "aviation/aviation.yaml").read_bytes())
    writer = plan_writer(tmp_path, "aviation", model_path=model)

    result = writer.prepare(
        "aviation",
        {
            "relationships": [
                {
                    "name": name,
                    "action": action,
                    "from_dataset": "flights",
                    "to_dataset": "aircraft",
                    "from_fields": ["aircraft_id", "flight_id"],
                }
            ]
        },
    )

    assert result.success == 0
    assert f"Relationship {name} requires matching lengths for from_fields and to_fields" in result.error


@pytest.mark.parametrize("to_dataset", ["aircraft", "carriers"])
def test_rejects_reusing_a_join_whose_existing_endpoint_is_deleted(tmp_path, to_dataset):
    model = tmp_path / "aviation.yaml"
    model.write_bytes((MODELS / "aviation/aviation.yaml").read_bytes())
    writer = plan_writer(tmp_path, "aviation", model_path=model)

    result = writer.write_semantic_model_plan_file(
        "aviation",
        {
            "datasets": [{"name": "aircraft", "action": "delete"}],
            "relationships": [
                {
                    "name": "flights_operated_by",
                    "from_dataset": "flights",
                    "to_dataset": to_dataset,
                    "action": "reuse",
                }
            ],
        },
    )

    assert result.success == 0
    assert "Relationship flights_operated_by refers to a dataset planned for deletion" in result.error
    assert not list(tmp_path.rglob("semantic-model-plan.json"))


def test_explicit_repair_uses_readable_baseline_instead_of_requiring_compilation(real_agent_config):
    from datus.tools.func_tool.osi_target_tools import OsiSemanticModelTargetTools

    root = Path(real_agent_config.project_root)
    model_dir = root / "subject" / "semantic_models" / real_agent_config.current_datasource
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = model_dir / "sales.yaml"
    broken = SALES_YAML.replace("    source: main.orders\n", "").replace(
        "\nmetrics:",
        """
  - name: refunds
    source: main.refunds
  - name: from_select
    source: /* fixture */ SELECT 1 AS select_marker
    fields:
      - name: select_marker
        expression:
          dialects: [{dialect: ANSI_SQL, expression: select_marker}]
  - name: from_cte
    source: (WITH sample AS (SELECT 2 AS cte_marker) SELECT cte_marker FROM sample)
  - name: from_extension
    source: VALUES (303)
    custom_extensions:
      - vendor_name: DATUS
        data: '{"source_type":"query"}'
metrics:""",
    )
    existing.write_text(broken)
    targets = OsiSemanticModelTargetTools(real_agent_config)
    assert targets.list_existing_osi_semantic_models().result["status"] == "repairable"
    assert targets.plan_osi_semantic_model_target(semantic_model_name="sales").result["repair_required"]
    writer = SemanticPlanFileTools(root, target_state=targets.target_state, chat_identity=lambda: ("owner", "repair"))
    prepared = writer.prepare(
        "sales",
        {"datasets": [{"name": "orders", "action": "update", "source_kind": "physical", "source": "main.orders"}]},
    )
    old = json.loads(prepared.baseline_text)
    new = json.loads(prepared.new_text)
    assert "dataset:orders" in _graph_nodes(old)
    assert _graph_nodes(old)["dataset:refunds"]["detail"]["source"] == "main.refunds"
    assert _graph_nodes(new)["table:main.refunds"]["detail"]["datasets"] == ["refunds"]
    assert any(
        edge["kind"] == "reads_table" and edge["source"] == "table:main.refunds" and edge["target"] == "dataset:refunds"
        for edge in old["edges"]
    )
    assert _graph_nodes(new)["dataset:orders"]["detail"]["source"] == "main.orders"
    assert _graph_nodes(new)["metric:revenue"] == _graph_nodes(old)["metric:revenue"]
    assert "without compiled lineage" in new["plan"]["summary"]
    old_nodes = _graph_nodes(old)
    query_names = {"from_select", "from_cte", "from_extension"}
    assert {node["name"] for node in old_nodes.values() if node["detail"].get("source_kind") == "query"} == query_names
    assert old_nodes["dataset:from_select"]["detail"]["fields"] == [{"field": "select_marker"}]
    assert {node["name"] for node in old_nodes.values() if node["kind"] == "physical_table"} == {"main.refunds"}
    assert {node["name"] for node in _graph_nodes(new).values() if node["kind"] == "physical_table"} == {
        "main.orders",
        "main.refunds",
    }
    assert writer.commit(prepared).success == 1
    saved = (root / prepared.path).read_text()
    assert all(
        sql not in text
        for text in (prepared.baseline_text, prepared.new_text, saved)
        for sql in ("SELECT 1 AS select_marker", "WITH sample AS", "VALUES (303)")
    )
    assert existing.read_text() == broken


def test_managed_yaml_update_invalidates_fixed_baseline_until_next_turn(tmp_path, existing_sales):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    proposal = {"metrics": [{"name": "planned", "action": "create"}]}
    first = writer.prepare("sales", proposal)

    updated = (
        SALES_YAML
        + """\
  - name: landed
    expression:
      dialects: [{dialect: ANSI_SQL, expression: MAX(orders.amount)}]
"""
    )
    existing_sales.write_text(updated)
    writer._target_state.record_metric_touch(existing_sales, updated.encode(), ["landed"])
    stale = writer.prepare("sales", proposal)

    assert stale.success == 0
    assert "start a fresh planning turn" in stale.error
    assert writer.commit(first).success == 0
    writer.reset_turn()
    refreshed = writer.prepare("sales", proposal)
    assert "metric:landed" in _graph_nodes(json.loads(refreshed.new_text))
    assert writer.commit(refreshed).success == 1


@pytest.mark.parametrize("completed", [False, True])
def test_next_turn_uses_actual_yaml_after_an_accepted_or_partially_applied_plan(tmp_path, existing_sales, completed):
    writer = plan_writer(tmp_path, model_path=existing_sales)
    accepted = writer.prepare(
        "sales",
        {"metrics": [{"name": "landed", "action": "create"}, {"name": "unlanded", "action": "create"}]},
    )
    assert writer.commit(accepted).success == 1
    if completed:
        existing_sales.write_text(
            SALES_YAML
            + """\
  - name: landed
    expression:
      dialects: [{dialect: ANSI_SQL, expression: MAX(orders.amount)}]
"""
        )
        select_plan_model(writer._target_state, existing_sales, "sales")
    writer.record_outcome(completed=completed)
    recorded = json.loads((tmp_path / accepted.path).read_text())
    assert "metric:unlanded" in _graph_nodes(recorded)
    assert recorded["plan"]["execution"] == ("completed" if completed else "not_applied")
    if completed:
        assert "metric:landed" in _graph_nodes(recorded["applied"])
        assert "metric:unlanded" not in _graph_nodes(recorded["applied"])
    else:
        assert "applied" not in recorded
    writer.reset_turn()
    later = writer.prepare("sales", {"metrics": [{"name": "another_metric", "action": "create"}]})
    actual_baseline = _graph_nodes(json.loads(later.baseline_text))
    assert later.path == accepted.path
    assert later.revision == 2
    assert "metric:unlanded" not in actual_baseline
    assert ("metric:landed" in actual_baseline) is completed


def test_unreadable_existing_model_is_not_an_empty_baseline(tmp_path, existing_sales):
    existing_sales.write_text("[invalid yaml")
    result = plan_writer(tmp_path, model_path=existing_sales).prepare("sales", {"metrics": [{"name": "something"}]})
    assert result.success == 0
    assert "Cannot read model lineage from subject/semantic_models/sales.yaml" in result.error
    assert not list(tmp_path.rglob("semantic-model-plan.json"))


def test_scoped_plan_rechecks_file_access_when_a_review_is_committed(tmp_path):
    """A reviewed proposal cannot retain a file grant narrowed before acceptance."""
    from datus.tools.func_tool.base import FuncToolResult

    allowed = True
    writer = SemanticPlanFileTools(tmp_path, path_allowed=lambda path: allowed)
    proposal = writer.prepare("sales", {"summary": "Review sales"})
    assert not isinstance(proposal, FuncToolResult)
    allowed = False
    rejected = writer.commit(proposal)
    assert rejected.success == 0
    assert "project_files" in rejected.error
    assert not (tmp_path / proposal.path).exists()


def test_standalone_review_reuses_only_its_own_plan(tmp_path, existing_sales):
    standalone = SemanticPlanFileTools(tmp_path)
    first = standalone.write_semantic_model_plan_file("sales", {"summary": "Standalone plan"})
    original_path = tmp_path / first.result["path"]
    original = original_path.read_text()
    author = plan_writer(tmp_path, model_path=existing_sales)
    chat_plan = author.write_semantic_model_plan_file("sales", {"summary": "Chat plan"})

    assert chat_plan.result["path"] != first.result["path"]
    proposal = SemanticPlanFileTools(tmp_path).prepare("sales", {"summary": "Revised standalone plan"})
    assert proposal.path == first.result["path"]
    assert proposal.preview()["oldText"] == original
    assert "metric:revenue" not in _graph_nodes(json.loads(proposal.new_text))


def test_standalone_commit_rejects_a_second_plan_identity_for_the_same_model(tmp_path):
    from dataclasses import replace
    from uuid import uuid4

    writer = SemanticPlanFileTools(tmp_path)
    pending = writer.prepare("sales", {"summary": "Pending before identity convergence"})
    competing_id = uuid4().hex
    graph = json.loads(pending.new_text)
    graph["plan"]["id"] = competing_id
    competing = replace(
        pending,
        path=f".datus/semantic-model-plans/{competing_id}/semantic-model-plan.json",
        new_text=json.dumps(graph),
    )

    assert writer.commit(pending).success == 1
    assert writer.commit(competing).success == 0
    assert [path.read_text() for path in tmp_path.rglob("semantic-model-plan.json")] == [pending.new_text]
