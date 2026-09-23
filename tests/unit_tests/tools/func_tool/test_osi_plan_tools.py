# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import json
from pathlib import Path

import pytest

from datus.tools.func_tool.osi_plan_tools import (
    OsiSemanticModelPlan,
    OsiSemanticModelPlanState,
    OsiSemanticModelPlanTools,
)


def _target(path: Path) -> dict:
    return {
        "semantic_model_name": "commerce",
        "semantic_model_file": "subject/semantic_models/test/commerce.yml",
        "absolute_path": str(path),
        "exists": path.exists(),
    }


def _dataset_plan(*, action="create", source_kind="physical", **overrides) -> OsiSemanticModelPlan:
    dataset = {
        "name": "orders",
        "action": action,
        "source_kind": source_kind,
        **overrides,
    }
    if source_kind == "physical":
        dataset["source"] = "analytics.orders"
    elif source_kind == "query":
        dataset.setdefault("source_tables", ["analytics.orders"])
    return OsiSemanticModelPlan.model_validate(
        {
            "summary": "Create reusable order semantics",
            "datasets": [dataset],
        }
    )


def test_workflow_plan_is_auto_approved_and_persisted(tmp_path):
    target_path = tmp_path / "commerce.yml"
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")

    result = state.submit(_dataset_plan(), target=_target(target_path))

    assert result["status"] == "approved"
    assert result["confirmation_required"] is False
    artifact = Path(result["plan_file"])
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["model"] == "commerce"
    assert result["graph"] == payload
    assert [layer["kind"] for layer in payload["layers"]] == [
        "physical_table",
        "dataset",
        "metric_atomic",
        "metric_derived",
    ]
    assert [node["id"] for node in payload["nodes"]] == ["table:analytics.orders", "dataset:orders"]
    assert payload["nodes"][1]["detail"]["source_kind"] == "table"
    assert payload["edges"] == [
        {
            "id": "reads_table:table:analytics.orders->dataset:orders",
            "kind": "reads_table",
            "source": "table:analytics.orders",
            "target": "dataset:orders",
            "via": "source",
        }
    ]
    state.require_approved(target_path)


def test_interactive_plan_blocks_writes_until_explicit_approval(tmp_path):
    target_path = tmp_path / "commerce.yml"
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="interactive", authoring_scope="full")

    result = state.submit(_dataset_plan(), target=_target(target_path))

    assert result["status"] == "pending_confirmation"
    with pytest.raises(ValueError, match="Submit and approve"):
        state.require_approved(target_path)
    state.approve_after_user_confirmation(result["plan_id"])
    state.require_approved(target_path)


def test_plan_schema_stays_focused_on_authoring_decisions():
    plan = _dataset_plan()
    payload = plan.model_dump(mode="json")

    assert "outputs" not in payload
    assert "verification" not in payload
    assert "relationship_budget" not in payload


def test_empty_plan_cannot_unlock_yaml_mutation(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    plan = OsiSemanticModelPlan.model_validate({"summary": "No semantic changes"})

    with pytest.raises(ValueError, match="at least one dataset, relationship, or metric"):
        state.submit(plan, target=_target(tmp_path / "commerce.yml"))


def test_query_backed_plan_requires_rowset_semantics_and_native_gap(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")

    with pytest.raises(ValueError, match="rowset_semantics, native_gap"):
        state.submit(_dataset_plan(source_kind="query"), target=_target(tmp_path / "commerce.yml"))


def test_query_backed_plan_keeps_implementation_sql_out_of_plan(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    plan = _dataset_plan(
        source_kind="query",
        source="SELECT * FROM analytics.orders",
        rowset_semantics="One reusable row per order cohort member",
        native_gap="The row-level cohort membership is not a metric aggregation",
    )

    with pytest.raises(ValueError, match="omit implementation SQL"):
        state.submit(plan, target=_target(tmp_path / "commerce.yml"))


def test_query_backed_plan_keeps_source_table_lineage_without_sql(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    plan = _dataset_plan(
        source_kind="query",
        source_tables=["analytics.orders", "analytics.customers"],
        rowset_semantics="One reusable row per qualified order and customer mapping",
        native_gap="The reusable row-level mapping is not a metric aggregation",
    )

    result = state.submit(plan, target=_target(tmp_path / "commerce.yml"))
    graph = result["graph"]

    assert [node["id"] for node in graph["nodes"]] == [
        "table:analytics.orders",
        "table:analytics.customers",
        "dataset:orders",
    ]
    dataset = graph["nodes"][-1]
    assert dataset["detail"]["source_kind"] == "query"
    assert dataset["detail"]["source_tables"] == ["analytics.orders", "analytics.customers"]
    assert "source" not in dataset["detail"]
    assert [edge["via"] for edge in graph["edges"]] == ["source_sql", "source_sql"]


def test_relationship_plan_only_requires_intended_endpoints(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    raw_plan = _dataset_plan().model_dump(mode="json")
    raw_plan["relationships"] = [
        {
            "name": "orders_customer",
            "action": "create",
            "from_dataset": "orders",
            "to_dataset": "customers",
            "reason": "Analyze orders by customer",
        }
    ]

    with pytest.raises(ValueError, match="requires from_fields and to_fields"):
        state.submit(OsiSemanticModelPlan.model_validate(raw_plan), target=_target(tmp_path / "commerce.yml"))


def test_persisted_plan_projects_relationships_and_metric_dependencies_as_dag(tmp_path):
    target_path = tmp_path / "commerce.yml"
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    plan = OsiSemanticModelPlan.model_validate(
        {
            "summary": "Plan order metrics",
            "datasets": [
                {
                    "name": "orders",
                    "action": "create",
                    "source_kind": "physical",
                    "source": "analytics.orders",
                    "grain": "one row per order",
                },
                {
                    "name": "customers",
                    "action": "reuse",
                    "source_kind": "physical",
                    "source": "analytics.customers",
                },
            ],
            "field_decisions": [
                {
                    "dataset": "orders",
                    "field": "ordered_at",
                    "role": "time",
                    "group_by": True,
                    "reason": "Business event time",
                }
            ],
            "relationships": [
                {
                    "name": "orders_customer",
                    "action": "create",
                    "from_dataset": "orders",
                    "to_dataset": "customers",
                    "from_fields": ["customer_id"],
                    "to_fields": ["customer_id"],
                    "cardinality": "many_to_one",
                }
            ],
            "metrics": [
                {
                    "name": "revenue",
                    "action": "create",
                    "kind": "atomic",
                    "role": "business_output",
                    "dataset": "orders",
                    "definition": "Sum order amount",
                },
                {
                    "name": "revenue_rank",
                    "action": "create",
                    "kind": "window",
                    "role": "helper",
                    "definition": "Rank revenue within the requested partition",
                    "depends_on": ["revenue"],
                    "additivity": "non_additive",
                },
            ],
        }
    )

    result = state.submit(plan, target=_target(target_path))
    payload = json.loads(Path(result["plan_file"]).read_text(encoding="utf-8"))

    assert [node["id"] for node in payload["nodes"]] == [
        "table:analytics.orders",
        "table:analytics.customers",
        "dataset:orders",
        "dataset:customers",
        "metric:revenue",
        "metric:revenue_rank",
    ]
    orders = next(node for node in payload["nodes"] if node["id"] == "dataset:orders")
    assert orders["detail"]["fields"] == [
        {
            "field": "ordered_at",
            "role": "time",
            "group_by": True,
            "reason": "Business event time",
        }
    ]
    assert [(edge["kind"], edge["source"], edge["target"]) for edge in payload["edges"]] == [
        ("reads_table", "table:analytics.orders", "dataset:orders"),
        ("reads_table", "table:analytics.customers", "dataset:customers"),
        ("join", "dataset:orders", "dataset:customers"),
        ("aggregates", "dataset:orders", "metric:revenue"),
        ("window_base", "metric:revenue", "metric:revenue_rank"),
    ]


def test_plan_rejects_atomic_metric_dependencies(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    raw_plan = _dataset_plan().model_dump(mode="json")
    raw_plan["metrics"] = [
        {
            "name": "revenue",
            "action": "create",
            "kind": "atomic",
            "role": "business_output",
            "dataset": "orders",
            "definition": "Sum order amount",
        },
        {
            "name": "revenue_score",
            "action": "create",
            "kind": "atomic",
            "role": "business_output",
            "dataset": "orders",
            "definition": "Score derived from revenue",
            "depends_on": ["revenue"],
        },
    ]

    with pytest.raises(ValueError, match="Atomic metric 'revenue_score' cannot depend"):
        state.submit(OsiSemanticModelPlan.model_validate(raw_plan), target=_target(tmp_path / "commerce.yml"))


def test_plan_rejects_duplicate_node_names(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    raw_plan = _dataset_plan().model_dump(mode="json")
    raw_plan["datasets"].append(dict(raw_plan["datasets"][0]))

    with pytest.raises(ValueError, match="duplicate dataset names: orders"):
        state.submit(OsiSemanticModelPlan.model_validate(raw_plan), target=_target(tmp_path / "commerce.yml"))


@pytest.mark.parametrize(
    ("plan_update", "error"),
    [
        (
            {
                "field_decisions": [
                    {
                        "dataset": "customers",
                        "field": "region",
                        "role": "dimension",
                        "reason": "Requested grouping",
                    }
                ]
            },
            "Field decision customers.region references an unplanned dataset",
        ),
        (
            {
                "relationships": [
                    {
                        "name": "orders_customer",
                        "action": "create",
                        "from_dataset": "orders",
                        "to_dataset": "customers",
                        "from_fields": ["customer_id"],
                        "to_fields": ["customer_id"],
                    }
                ]
            },
            "references unplanned datasets: customers",
        ),
        (
            {
                "metrics": [
                    {
                        "name": "customer_count",
                        "action": "create",
                        "kind": "atomic",
                        "role": "business_output",
                        "dataset": "customers",
                        "definition": "Count customers",
                    }
                ]
            },
            "references unplanned dataset 'customers'",
        ),
        (
            {
                "metrics": [
                    {
                        "name": "revenue_score",
                        "action": "create",
                        "kind": "compose",
                        "role": "business_output",
                        "definition": "Score derived from revenue",
                        "depends_on": ["revenue"],
                    }
                ]
            },
            "references unplanned metrics: revenue",
        ),
    ],
)
def test_plan_rejects_dangling_graph_references(tmp_path, plan_update, error):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    raw_plan = _dataset_plan().model_dump(mode="json")
    raw_plan.update(plan_update)

    with pytest.raises(ValueError, match=error):
        state.submit(OsiSemanticModelPlan.model_validate(raw_plan), target=_target(tmp_path / "commerce.yml"))


def test_replacing_plan_does_not_require_a_revision_contract(tmp_path):
    target_path = tmp_path / "commerce.yml"
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    first = state.submit(_dataset_plan(), target=_target(target_path))

    second = state.submit(
        _dataset_plan(reason="Use the live order table as the reusable base"),
        target=_target(target_path),
    )

    assert second["plan_id"] == first["plan_id"]
    assert second["revision"] == 2


def test_tool_returns_structured_error_for_invalid_query_plan(tmp_path):
    target_path = tmp_path / "commerce.yml"
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="full")
    target_state = type("TargetState", (), {"selected": _target(target_path)})()
    tools = OsiSemanticModelPlanTools(plan_state=state, target_state=target_state)

    result = tools.submit_osi_semantic_model_plan(_dataset_plan(source_kind="query"))

    assert result.success == 0
    assert result.result["code"] == "semantic_model_plan_invalid"


def test_datasets_only_plan_rejects_metric_mutations(tmp_path):
    state = OsiSemanticModelPlanState(project_root=tmp_path, execution_mode="workflow", authoring_scope="datasets")
    raw_plan = _dataset_plan().model_dump(mode="json")
    raw_plan["metrics"] = [
        {
            "name": "order_count",
            "action": "create",
            "kind": "atomic",
            "role": "business_output",
            "dataset": "orders",
            "definition": "Count orders",
        }
    ]

    with pytest.raises(ValueError, match="cannot plan metric mutations"):
        state.submit(OsiSemanticModelPlan.model_validate(raw_plan), target=_target(tmp_path / "commerce.yml"))
