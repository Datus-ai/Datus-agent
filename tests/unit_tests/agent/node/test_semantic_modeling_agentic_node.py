# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the unified semantic modeling subagent."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from datus.agent.node import semantic_authoring
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput, SourceQueryEvidence


@pytest.fixture(autouse=True)
def _stub_osi_schema_validation(monkeypatch):
    monkeypatch.setattr(semantic_authoring, "validate_osi_core_document", lambda document: None)
    monkeypatch.setattr(
        semantic_authoring,
        "validate_osi_authoring_document",
        lambda document, *, semantic_adapter: None,
    )
    monkeypatch.setattr(
        semantic_authoring,
        "render_required_authoring_skill",
        lambda _name, content, *, include_osi_core=False: (
            content
            + (
                "\n## Active OSI Core authoring specification\n# Apache Ossie - Core Metadata Spec"
                if include_osi_core
                else ""
            )
            + "\n## Active DATUS extension authoring specification"
        ),
    )


def _set_adapter(agent_config, adapter: str) -> None:
    agent_config.resolve_semantic_adapter = MagicMock(return_value=adapter)


def test_unified_dosi_node_composes_existing_authoring_surfaces(real_agent_config, mock_llm_create):
    from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    assert not issubclass(SemanticModelingAgenticNode, GenMetricsAgenticNode)
    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Create the order dataset and its revenue metric")
    tool_names = {tool.name for tool in node.tools}

    assert node.get_node_name() == "semantic_modeling"
    assert {
        "list_existing_osi_semantic_models",
        "plan_osi_semantic_model_target",
        "bind_osi_semantic_model_target",
        "read_file",
        "edit_file",
        "upsert_osi_datasets",
        "delete_osi_datasets",
        "upsert_osi_metrics",
        "delete_osi_metrics",
        "glob",
        "grep",
        "validate_semantic",
    }.issubset(tool_names)
    assert {"write_file", "delete_file", "bash", "task"}.isdisjoint(tool_names)
    assert node._get_required_skills() == ["dosi-semantic-authoring"]
    assert node.semantic_discovery_tools.compact_source_inspection is True

    prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))
    assert "Select one target" in prompt
    assert "compact dataset, table, relationship, and metric coverage" in prompt
    assert "use the returned authoring outline" in prompt
    assert "plan that same model name so it can be repaired in place" in prompt
    assert "Treat SQL as evidence rather than a required persisted result shape" in prompt
    assert "extract reusable fields, relationships, and native business metrics" in prompt
    assert "durable reusable cohort/result set or asks for faithful one-query reproduction" in prompt
    assert '<required_skill name="dosi-semantic-authoring">' in prompt
    assert "## Active OSI Core authoring specification" in prompt
    assert "# Apache Ossie - Core Metadata Spec" in prompt
    assert "## Active DATUS extension authoring specification" in prompt


def test_datasets_only_scope_hides_metric_mutations_and_updates_prompt(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(
        agent_config=real_agent_config,
        execution_mode="workflow",
        authoring_scope="datasets",
    )
    node.input = SemanticNodeInput(user_message="Create reusable order datasets", authoring_scope="datasets")

    tool_names = {tool.name for tool in node.tools}
    assert {"upsert_osi_datasets", "delete_osi_datasets", "edit_file"}.issubset(tool_names)
    assert {"upsert_osi_metrics", "delete_osi_metrics"}.isdisjoint(tool_names)

    prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))
    assert "This run is datasets-only" in prompt
    assert "Do not author metrics in this datasets-only run" in prompt
    assert "Keep all existing metric definitions unchanged" in prompt


def test_datasets_only_scope_rolls_back_metric_changes_made_through_edit_file(
    real_agent_config,
    mock_llm_create,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

    _set_adapter(real_agent_config, "dosi")
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "orders.yml"
    original = (
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        "  - name: orders_model\n"
        "    datasets: []\n"
        "    relationships: []\n"
        "    metrics:\n"
        "      - name: order_count\n"
        "        expression:\n"
        "          dialects: [{dialect: SQLITE, expression: 'COUNT(*)'}]\n"
    )
    target.write_text(original, encoding="utf-8")
    node = SemanticModelingAgenticNode(
        agent_config=real_agent_config,
        execution_mode="workflow",
        authoring_scope="datasets",
    )
    node.input = SemanticNodeInput(user_message="Update order relationships", authoring_scope="datasets")
    node.osi_target_state.select(
        {
            "semantic_model_name": "orders_model",
            "semantic_model_file": f"subject/semantic_models/{real_agent_config.current_datasource}/orders.yml",
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="planned",
    )

    edit_result = node.filesystem_func_tool.edit_file(
        str(target),
        "name: order_count",
        "name: changed_order_count",
    )
    assert edit_result.success == 1

    ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
    ctx.response_content = {"status": "generated", "output": "Updated relationships"}
    with pytest.raises(RuntimeError, match="Datasets-only semantic_modeling cannot"):
        node._build_success_result(ctx)

    assert target.read_text(encoding="utf-8") == original


def test_unified_node_exposes_structured_request_sql_as_discovery_evidence(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(
        user_message="Create reusable order metrics",
        source_queries=[
            SourceQueryEvidence(
                source_sql_name="orders_revenue",
                question="What is order revenue?",
                sql="SELECT SUM(amount) AS revenue FROM orders",
            )
        ],
    )

    assert node._semantic_discovery_source_sql() == [
        {
            "name": "orders_revenue",
            "question": "What is order revenue?",
            "sql": "SELECT SUM(amount) AS revenue FROM orders",
        }
    ]


def test_interactive_prompt_confirms_ambiguous_semantics_and_target(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
    node.input = SemanticNodeInput(user_message="Create a revenue metric from SQL")

    prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))

    assert "multiple materially different interpretations remain plausible" in prompt
    assert "confirm the target once before selecting it" in prompt
    assert "other plausible existing models, and creating a new semantic model" in prompt
    assert "all currently known semantic ambiguities and the target choice into one `ask_user` call" in prompt


def test_dosi_target_selection_requires_existing_model_check(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    rejected = node.plan_osi_semantic_model_target(semantic_model_name="orders")
    assert rejected.success == 0
    assert rejected.result["code"] == "existing_semantic_models_check_required"

    inventory = node.list_existing_osi_semantic_models()
    assert inventory.success == 1
    planned = node.plan_osi_semantic_model_target(semantic_model_name="orders")
    assert planned.success == 1
    assert node.osi_target_state.planned["semantic_model_name"] == "orders"


@pytest.mark.parametrize("adapter", ["metricflow", "osi"])
def test_semantic_modeling_rejects_non_dosi_adapters(real_agent_config, mock_llm_create, adapter):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, adapter)
    from datus.utils.exceptions import DatusException

    with pytest.raises(DatusException, match="query-only.*migrate it to Dosi.*semantic_modeling"):
        SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")


def test_unified_dosi_selects_existing_model_once_for_dataset_and_metric_changes(
    real_agent_config,
    mock_llm_create,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "commerce.yml"
    target.write_text(
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        "  - name: commerce\n"
        "    datasets:\n"
        "      - name: orders\n"
        "        source: analytics.orders\n"
        "        fields: []\n"
        "    relationships: []\n"
        "    metrics: []\n",
        encoding="utf-8",
    )
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    assert node.list_existing_osi_semantic_models().success == 1
    bound = node.bind_osi_semantic_model_target(
        semantic_model_file=str(target),
        semantic_model_name="commerce",
    )
    assert bound.success == 1
    assert node.semantic_tools._selected_semantic_model_path() == str(target.resolve())

    dataset = {
        "name": "orders",
        "source": "analytics.orders",
        "fields": [
            {
                "name": "amount",
                "expression": {"dialects": [{"dialect": "STARROCKS", "expression": "amount"}]},
            }
        ],
    }
    updated = node.filesystem_func_tool.upsert_osi_datasets(str(target), json.dumps([dataset]))
    assert updated.success == 1
    assert node.osi_target_state.target_mutated is True

    assert node.osi_target_state.bound["absolute_path"] == str(target.resolve())
    assert node.semantic_tools._selected_semantic_model_path() == str(target.resolve())

    metric = {
        "name": "order_count",
        "expression": {"dialects": [{"dialect": "STARROCKS", "expression": "COUNT(*)"}]},
    }
    metric_result = node.filesystem_func_tool.upsert_osi_metrics(str(target), json.dumps([metric]))
    assert metric_result.success == 1
    assert node.osi_target_state.touched_metric_names == ["order_count"]


def test_unified_dosi_plans_new_model_once_for_dataset_and_metric_changes(
    real_agent_config,
    mock_llm_create,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    assert node.list_existing_osi_semantic_models().success == 1
    planned = node.plan_osi_semantic_model_target(
        semantic_model_name="commerce",
        fact_tables=["analytics.orders"],
    )
    assert planned.success == 1
    target = node.osi_target_state.selected_path
    assert target is not None

    dataset_result = node.filesystem_func_tool.upsert_osi_datasets(
        target,
        json.dumps(
            [
                {
                    "name": "orders",
                    "source": "analytics.orders",
                    "fields": [
                        {
                            "name": "amount",
                            "expression": {"dialects": [{"dialect": "STARROCKS", "expression": "amount"}]},
                        }
                    ],
                }
            ]
        ),
    )
    metric_result = node.filesystem_func_tool.upsert_osi_metrics(
        target,
        json.dumps(
            [
                {
                    "name": "revenue",
                    "expression": {"dialects": [{"dialect": "STARROCKS", "expression": "SUM(orders.amount)"}]},
                }
            ]
        ),
    )

    assert dataset_result.success == 1
    assert metric_result.success == 1
    assert node.osi_target_state.planned["semantic_model_name"] == "commerce"
    assert node.osi_target_state.touched_metric_names == ["revenue"]


def test_unified_dosi_keeps_the_first_successfully_selected_target(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    assert node.list_existing_osi_semantic_models().success == 1
    first = node.plan_osi_semantic_model_target(semantic_model_name="commerce")
    assert first.success == 1

    second = node.plan_osi_semantic_model_target(semantic_model_name="support")
    assert second.success == 0
    assert second.result["code"] == "semantic_model_target_already_selected"
    assert node.osi_target_state.selected["semantic_model_name"] == "commerce"


def test_factory_creates_unified_semantic_modeling_node(real_agent_config, mock_llm_create):
    from datus.agent.node.node_factory import create_interactive_node, create_node_input
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    node = create_interactive_node("semantic_modeling", real_agent_config, execution_mode="workflow")
    assert isinstance(node, SemanticModelingAgenticNode)
    node_input = create_node_input("Update orders", node)
    assert isinstance(node_input, SemanticNodeInput)


def test_generated_result_uses_selected_artifact_finalizer(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Update the orders dataset")
    node._finalize_selected_osi_artifact = MagicMock(return_value="subject/semantic_models/warehouse/orders.yml")
    ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
    ctx.response_content = {
        "status": "generated",
        "output": "Updated orders",
    }

    result = node._build_success_result(ctx)

    node._finalize_selected_osi_artifact.assert_called_once_with()
    assert result.status == "generated"
    assert result.semantic_models == ["subject/semantic_models/warehouse/orders.yml"]


def test_unified_result_can_skip_when_no_semantic_change_is_needed(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

    _set_adapter(real_agent_config, "dosi")
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Explain the existing metric")
    ctx = StreamRunContext(user_input=node.input, action_history_manager=ActionHistoryManager())
    ctx.response_content = {
        "status": "skipped",
        "skip_reason": "no_semantic_change",
        "output": "No semantic change is required.",
    }

    result = node._build_success_result(ctx)

    assert result.status == "skipped"
    assert result.skip_reason == "no_semantic_change"
    assert result.response == "No semantic change is required."


@pytest.mark.parametrize(
    ("metrics_yaml", "expected_scope"),
    [
        ("    metrics: []\n", "semantic_model"),
        (
            "    metrics:\n"
            "      - name: order_count\n"
            "        expression:\n"
            "          dialects: [{dialect: STARROCKS, expression: 'COUNT(*)'}]\n",
            "all",
        ),
    ],
)
def test_host_finalizer_validates_and_reconciles_complete_selected_yaml(
    real_agent_config,
    mock_llm_create,
    metrics_yaml,
    expected_scope,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.tools.func_tool.base import FuncToolResult

    _set_adapter(real_agent_config, "dosi")
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "orders.yml"
    target.write_text(
        "version: 0.2.0.dev0\n"
        "semantic_model:\n"
        "  - name: orders_model\n"
        "    datasets:\n"
        "      - name: orders\n"
        "        source: orders\n"
        "        fields: []\n"
        "    relationships: []\n" + metrics_yaml,
        encoding="utf-8",
    )
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    public_path = f"subject/semantic_models/{real_agent_config.current_datasource}/orders.yml"
    node.osi_target_state.select(
        {
            "semantic_model_name": "orders_model",
            "semantic_model_file": public_path,
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="planned",
    )
    node.osi_target_state.artifact_snapshot_path = str(target.resolve())
    node.osi_target_state.artifact_snapshot_content = target.read_bytes()
    node.semantic_tools.validate_semantic = MagicMock(
        return_value=FuncToolResult(result={"valid": True, "scope": expected_scope})
    )

    with patch.object(
        node.generation_tools,
        "sync_osi_to_db",
        return_value={"success": True},
    ) as reconcile:
        result = node._finalize_selected_osi_artifact()

    assert result == public_path
    node.semantic_tools.validate_semantic.assert_called_once_with(
        scope=expected_scope,
        semantic_model_name="orders_model",
    )
    reconcile.assert_called_once_with(
        str(target.resolve()),
        include_semantic_objects=True,
        include_metrics=True,
    )
    assert node.osi_target_state.artifact_snapshot_path == ""


def test_host_finalizer_reports_unavailable_semantic_validation(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    _set_adapter(real_agent_config, "dosi")
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "orders.yml"
    target.write_text(
        "version: 0.2.0.dev0\nsemantic_model:\n  - name: orders_model\n    datasets: []\n    metrics: []\n",
        encoding="utf-8",
    )
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.osi_target_state.select(
        {
            "semantic_model_name": "orders_model",
            "semantic_model_file": f"subject/semantic_models/{real_agent_config.current_datasource}/orders.yml",
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="planned",
    )
    node.semantic_tools = None

    with pytest.raises(RuntimeError, match="validate_semantic is unavailable"):
        node._finalize_selected_osi_artifact()
