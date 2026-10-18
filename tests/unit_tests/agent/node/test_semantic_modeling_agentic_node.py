# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the unified semantic modeling subagent."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from datus.agent.node import semantic_authoring
from datus.schemas.action_history import ActionHistoryManager
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from tests.unit_tests.mock_llm_model import build_simple_response


@pytest.fixture(autouse=True)
def _accept_any_authoring_document(monkeypatch):
    """These drive the node, not the validator, which is covered on its own."""
    monkeypatch.setattr(semantic_authoring, "validate_osi_authoring_document", lambda document, **kwargs: None)
    monkeypatch.setattr(
        semantic_authoring,
        "render_required_authoring_skill",
        lambda _name, content, *, include_osi_core=False, reference_dir=None: (
            content
            + ("\n## OSI Core reference shapes" if include_osi_core else "")
            + "\n## Active DATUS capability directory"
        ),
    )
    monkeypatch.setattr(semantic_authoring, "prepare_authoring_references", lambda cache_dir: cache_dir)


def _stream_call_count(mock_llm_create) -> int:
    return sum(1 for call in mock_llm_create.call_history if call.get("method") == "generate_with_tools_stream")


def _record_current_compiled_evidence(node, target):
    names = node._semantic_validation_metric_names()
    catalog = {name: {"name": name, "kind": "aggregate", "datasets": ["orders"], "measures": [name]} for name in names}
    assert (
        node.generation_evidence.record_compiled_validation(
            {
                "contract_digest": "sha256:" + hashlib.sha256(b"contract").hexdigest(),
                "artifact_sha256": {str(target.resolve()): "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()},
                "compiled_metrics": list(catalog.values()),
                "compiled_metric_digests": {
                    name: "sha256:" + hashlib.sha256(name.encode()).hexdigest() for name in names
                },
            },
            metric_names=names,
        )
        is True
    )
    return catalog


def test_loaded_database_references_are_readable_without_loading_them_into_the_prompt(
    real_agent_config, mock_llm_create, tmp_path, tmp_path_factory, monkeypatch
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.tools.skill_tools.skill_config import SkillConfig
    from datus.tools.skill_tools.skill_manager import SkillManager

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    node.node_config["workspace_root"] = str(workspace)
    filesystem = node._make_filesystem_tool(strict=True)
    name = node._get_database_skill_candidates()[0]
    resource_root = tmp_path_factory.mktemp("adapter_bundle")
    bundle = resource_root / name
    (bundle / "references").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Query rules\n---\n"
        "# Query rules\nFor imports, read [Loading](references/data-loading.md).\n"
    )
    reference = bundle / "references/data-loading.md"
    reference.write_text("# Loading details\nINSERT INTO target SELECT * FROM files;\n")
    outside = resource_root / "outside.md"
    outside.write_text("Outside the loaded bundle")
    (bundle / "references/escape.md").symlink_to(outside)
    node.skill_manager = SkillManager(config=SkillConfig(directories=[str(bundle.parent)]))
    monkeypatch.setattr(node, "_get_required_skills", lambda: [])

    assert not filesystem.read_file(str(reference)).success
    prompt = node._inject_required_skills("Build metrics from supplied ETL")
    assert f"<skill_location>{bundle}</skill_location>" in prompt
    assert "Loading details" not in prompt
    assert filesystem.read_file(str(reference)).result == reference.read_text()
    assert filesystem._classify(str(reference)).read_only
    assert not filesystem.write_file(str(reference), "Changed").success
    assert not filesystem.read_file(str(bundle / "references/escape.md")).success
    sql = bundle / "references/example.sql"
    sql.write_text("SELECT 1 AS n")
    assert filesystem.read_file(str(sql)).result == sql.read_text()
    policy = node._make_filesystem_policy()
    assert bundle in policy.allowlist_provider().read


def test_unified_dosi_node_composes_existing_authoring_surfaces(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(
        user_message=(
            "Create the order dataset and its revenue metric using `sql/order ETL.sql`.\n"
            "SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region;"
        )
    )
    assert node.input.user_message in node._build_enhanced_message(node.input)
    tool_names = {tool.name for tool in node.tools}

    assert node.get_node_name() == "semantic_modeling"
    assert {
        "list_existing_osi_semantic_models",
        "plan_osi_semantic_model_target",
        "bind_osi_semantic_model_target",
        "list_tables",
        "describe_table",
        "execute_sql",
        "verify_semantic_outputs",
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
    assert {
        "write_file",
        "delete_file",
        "load_file_as_table",
        "import_database_file",
        "bash",
        "task",
        "write_semantic_model_plan_file",
    }.isdisjoint(tool_names)
    assert node.db_func_tool.read_only is True
    assert node._get_required_skills() == ["dosi-semantic-authoring"]

    with patch("datus.storage.metric.store.MetricStorage.get_subject_tree_flat") as subject_tree_lookup:
        prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))
    subject_tree_lookup.assert_not_called()
    assert "Select one target" in prompt
    assert "compact dataset, table, relationship, and metric coverage" in prompt
    assert "use the returned authoring outline" in prompt
    assert "plan that same model name so it can be repaired in place" in prompt
    assert "Treat SQL as evidence rather than a required persisted result shape" in prompt
    assert "extract reusable fields, relationships, and native business metrics" in prompt
    assert '<required_skill name="dosi-semantic-authoring">' in prompt
    node._populate_tool_registry()
    assert node.tool_registry.get("verify_semantic_outputs") == "semantic_tools"
    assert "## OSI Core reference shapes" in prompt
    assert "## Active DATUS capability directory" in prompt
    assert "ensure_metric_uids" not in tool_names


def test_plan_file_uses_project_root_for_vscode_source(real_agent_config, mock_llm_create, tmp_path, monkeypatch):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    daemon_cwd = tmp_path / "daemon"
    daemon_cwd.mkdir()
    monkeypatch.chdir(daemon_cwd)
    real_agent_config._client_source = "vscode"
    real_agent_config.agentic_nodes.setdefault("semantic_modeling", {})["tools"] = (
        "semantic_tools,semantic_tools.write_semantic_model_plan_file"
    )

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    assert node._resolve_workspace_root() == "."
    assert "write_semantic_model_plan_file" in {tool.name for tool in node.tools}
    node._populate_tool_registry()
    assert node.tool_registry.get("write_semantic_model_plan_file") == "semantic_tools"

    result = node.semantic_plan_file_tool.write_semantic_model_plan_file("sales_model", '{"summary":"plan"}')

    assert result.success == 1
    assert (Path(real_agent_config.project_root) / result.result["path"]).is_file()
    assert not (daemon_cwd / result.result["path"]).exists()


def test_semantic_modeling_db_tools_reject_writes_even_with_mutable_datasource(
    mutable_real_agent_config, mock_llm_create
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode(agent_config=mutable_real_agent_config, execution_mode="workflow")

    assert node.db_func_tool.read_only is True
    assert node.db_func_tool.execute_sql("SELECT 1").success == 1
    refused = node.db_func_tool.execute_sql("CREATE TABLE semantic_plan_probe (id INTEGER)")
    assert refused.success == 0
    assert "read-only" in refused.error


def test_resumed_session_rebuilds_legacy_key_policy_and_caches_current_skill(
    real_agent_config, mock_llm_create, tmp_path, monkeypatch
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.models.session_manager import SessionManager

    monkeypatch.setattr(
        semantic_authoring,
        "authoring_prompt_snapshot_meta",
        lambda *_args: {"datus_extension_version": "1.5", "datus_authoring_contract_digest": "sha256:unchanged"},
    )
    node = SemanticModelingAgenticNode(
        agent_config=real_agent_config, execution_mode="workflow", session_id="semantic_session"
    )
    node.input = SemanticNodeInput(user_message="Update the order dataset")
    node._session_manager = SessionManager(session_dir=str(tmp_path))
    old_meta = node._system_prompt_snapshot_meta(None)
    old_meta["semantic_target_scope"] = "agent_bound_v3"
    node.session_manager.save_system_prompt_snapshot(
        node.session_id, "Declare a new key only after full-table validation.", old_meta
    )

    context = node._prepare_template_context(node.input)
    with patch.object(node, "_get_system_prompt", wraps=node._get_system_prompt) as build_prompt:
        prompt = node._get_session_system_prompt(template_context=context)

        assert '<required_skill name="dosi-semantic-authoring">' in prompt
        assert "Declare keys only from source DDL" in prompt
        assert "Never query data to discover or verify keys" in prompt
        assert node.session_manager.load_system_prompt_snapshot(node.session_id)["prompt"] == prompt
        assert node._get_session_system_prompt(template_context=context) == prompt
        build_prompt.assert_called_once()


def test_datasets_only_scope_hides_metric_mutations_and_updates_prompt(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

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


def test_node_factory_applies_datasets_only_scope_before_tool_setup(real_agent_config, mock_llm_create):
    from datus.agent.node import Node
    from datus.configuration.node_type import NodeType

    input_data = SemanticNodeInput(user_message="Create reusable order datasets", authoring_scope="datasets")

    node = Node.new_instance(
        node_id="semantic_node",
        description="dataset authoring",
        node_type=NodeType.TYPE_SEMANTIC,
        input_data=input_data,
        agent_config=real_agent_config,
        node_name="semantic_modeling",
    )

    assert node.authoring_scope == "datasets"
    assert node.input is input_data
    assert {"upsert_osi_metrics", "delete_osi_metrics"}.isdisjoint(tool.name for tool in node.tools)


def test_invalid_authoring_scope_uses_structured_error(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.utils.exceptions import DatusException, ErrorCode

    with pytest.raises(DatusException) as exc_info:
        SemanticModelingAgenticNode(
            agent_config=real_agent_config,
            execution_mode="workflow",
            authoring_scope="metrics",
        )

    assert exc_info.value.code == ErrorCode.COMMON_UNSUPPORTED
    assert "authoring_scope" in str(exc_info.value)
    assert "metrics" in str(exc_info.value)


def test_datasets_only_scope_rolls_back_metric_changes_made_through_edit_file(
    real_agent_config,
    mock_llm_create,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

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


def test_interactive_prompt_confirms_ambiguous_semantics_and_target(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
    node.input = SemanticNodeInput(user_message="Create a revenue metric from SQL")

    prompt = node._get_system_prompt(template_context=node._prepare_template_context(node.input))

    assert "multiple materially different interpretations remain plausible" in prompt
    assert "confirm the target once before selecting it" in prompt
    assert "other plausible existing models, and creating a new semantic model" in prompt
    assert "all currently known semantic ambiguities and the target choice into one `ask_user` call" in prompt


def test_dosi_target_selection_requires_existing_model_check(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    rejected = node.plan_osi_semantic_model_target(semantic_model_name="orders")
    assert rejected.success == 0
    assert rejected.result["code"] == "existing_semantic_models_check_required"

    inventory = node.list_existing_osi_semantic_models()
    assert inventory.success == 1
    planned = node.plan_osi_semantic_model_target(semantic_model_name="orders")
    assert planned.success == 1
    assert node.osi_target_state.planned["semantic_model_name"] == "orders"


def test_unified_dosi_selects_existing_model_once_for_dataset_and_metric_changes(
    real_agent_config,
    mock_llm_create,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

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

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

    assert node.list_existing_osi_semantic_models().success == 1
    planned = node.plan_osi_semantic_model_target(
        semantic_model_name="commerce",
        fact_tables=["analytics.orders"],
    )
    assert planned.success == 1
    target = node.osi_target_state.selected_path
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    assert target == str((model_dir / "commerce.yml").resolve())

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

    node = create_interactive_node("semantic_modeling", real_agent_config, execution_mode="workflow")
    assert isinstance(node, SemanticModelingAgenticNode)
    node_input = create_node_input("Update orders", node)
    assert isinstance(node_input, SemanticNodeInput)


def test_generated_result_uses_selected_artifact_finalizer(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

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
    assert result.business_validation == []


@pytest.mark.asyncio
async def test_retries_unsupported_outcome_once_before_finalizing(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    mock_llm_create.reset(
        responses=[
            build_simple_response("The semantic model is complete and validated."),
            build_simple_response(json.dumps({"status": "generated", "output": "Updated orders"})),
        ]
    )
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Update the orders dataset")
    node._finalize_selected_osi_artifact = MagicMock(return_value="subject/semantic_models/warehouse/orders.yml")

    actions = [action async for action in node.execute_stream(ActionHistoryManager())]

    assert _stream_call_count(mock_llm_create) == 2
    assert "You may use the available tools" in mock_llm_create.call_history[-1]["prompt"]
    assert mock_llm_create.call_history[-1]["tools"]
    assert actions[-1].output["success"] is True
    assert actions[-1].output["status"] == "generated"
    node._finalize_selected_osi_artifact.assert_called_once_with()


@pytest.mark.parametrize(
    ("payload", "expected_status", "expected_success"),
    [
        ({"status": "generated", "output": "Updated orders"}, "generated", True),
        (
            {"status": "skipped", "skip_reason": "no_semantic_change", "output": "No change required"},
            "skipped",
            True,
        ),
        (
            {
                "status": "blocked",
                "blocker_code": "semantic_model_required",
                "output": "A semantic model is required",
            },
            "blocked",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_supported_outcome_does_not_retry(
    real_agent_config,
    mock_llm_create,
    payload,
    expected_status,
    expected_success,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    mock_llm_create.reset(responses=[build_simple_response(json.dumps(payload))])
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Update the orders dataset")
    node._finalize_selected_osi_artifact = MagicMock(return_value="subject/semantic_models/warehouse/orders.yml")

    actions = [action async for action in node.execute_stream(ActionHistoryManager())]

    assert _stream_call_count(mock_llm_create) == 1
    assert actions[-1].output["status"] == expected_status
    assert actions[-1].output["success"] is expected_success
    if expected_status == "generated":
        node._finalize_selected_osi_artifact.assert_called_once_with()
    else:
        node._finalize_selected_osi_artifact.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_outcome_still_fails_after_one_retry(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

    mock_llm_create.reset(
        responses=[
            build_simple_response("The semantic model is complete."),
            build_simple_response("The semantic model remains complete."),
        ]
    )
    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    node.input = SemanticNodeInput(user_message="Update the orders dataset")

    actions = [action async for action in node.execute_stream(ActionHistoryManager())]

    assert _stream_call_count(mock_llm_create) == 2
    assert actions[-1].output["success"] is False
    assert "must return a supported status" in actions[-1].output["error"]


def test_unified_result_can_skip_when_no_semantic_change_is_needed(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext

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
    ("metrics_yaml", "expected_scope", "touched_metrics"),
    [
        ("    metrics: []\n", "semantic_model", []),
        (
            "    metrics:\n"
            "      - name: order_count\n"
            "        expression:\n"
            "          dialects: [{dialect: STARROCKS, expression: 'COUNT(*)'}]\n",
            "all",
            ["order_count"],
        ),
    ],
)
def test_host_finalizer_validates_and_reconciles_complete_selected_yaml(
    real_agent_config,
    mock_llm_create,
    metrics_yaml,
    expected_scope,
    touched_metrics,
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.tools.func_tool.base import FuncToolResult

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
    node.osi_target_state.touched_metric_names = list(touched_metrics)
    compiled_validation_recorded = False
    expected_compiled_catalog = {}
    if touched_metrics:
        expected_compiled_catalog = {
            "order_count": {
                "name": "order_count",
                "kind": "aggregate",
                "datasets": ["orders"],
                "measures": ["order_count"],
            }
        }
        compiled_validation_recorded = node.generation_evidence.record_compiled_validation(
            {
                "contract_digest": "sha256:" + hashlib.sha256(b"contract").hexdigest(),
                "artifact_sha256": {str(target.resolve()): "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()},
                "compiled_metrics": list(expected_compiled_catalog.values()),
                "compiled_metric_digests": {"order_count": "sha256:" + hashlib.sha256(b"order_count").hexdigest()},
            },
            metric_names=touched_metrics,
        )
    assert compiled_validation_recorded is bool(touched_metrics)

    def validate_final_content(**kwargs):
        assert _record_current_compiled_evidence(node, target) == expected_compiled_catalog
        return FuncToolResult(result={"valid": True, "scope": expected_scope})

    node.semantic_tools.validate_semantic = MagicMock(side_effect=validate_final_content)

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
        include_metrics=bool(touched_metrics),
        metric_names_to_sync=set(touched_metrics),
        compiled_metric_catalog=expected_compiled_catalog,
    )
    assert node.osi_target_state.artifact_snapshot_path == ""


def test_host_finalizer_reports_unavailable_semantic_validation(real_agent_config, mock_llm_create):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode

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


@pytest.mark.parametrize("authoring_scope", ["full", "datasets"])
@pytest.mark.parametrize("legacy_has_uid", [False, True])
def test_finalizer_fills_legacy_uids_and_reconciles_existing_kb_identities(
    real_agent_config, mock_llm_create, authoring_scope, legacy_has_uid
):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.agent.node.stream_run_context import StreamRunContext
    from datus.tools.func_tool.base import FuncToolResult
    from datus.tools.semantic_tools.metric_identity import metric_uid

    node = SemanticModelingAgenticNode(
        agent_config=real_agent_config, execution_mode="workflow", authoring_scope=authoring_scope
    )
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "uid_model.yml"
    retained_uid = "tz4a98xxat96iws9zmbrgj3a"
    metrics = [
        {
            "name": name,
            "description": name,
            "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
        }
        for name in ("legacy", "retained")
    ]
    metrics[1]["custom_extensions"] = [{"vendor_name": "DATUS", "data": json.dumps({"uid": retained_uid})}]
    if legacy_has_uid:
        metrics[0]["custom_extensions"] = [{"vendor_name": "DATUS", "data": json.dumps({"uid": "a" * 24})}]
    target.write_text(
        yaml.safe_dump(
            {
                "version": "0.2.0.dev0",
                "semantic_model": [
                    {"name": "uid_model", "datasets": [{"name": "orders", "source": "orders"}], "metrics": metrics}
                ],
            },
            sort_keys=False,
        )
    )
    node.osi_target_state.select(
        {
            "semantic_model_name": "uid_model",
            "semantic_model_file": str(target),
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        mode="bound",
    )
    # Seed an existing KB identity so the unchanged YAML metric also needs sync.
    doc = node.generation_tools._load_osi_document(metric_file=str(target))
    rows = node.generation_tools._build_osi_metric_objects(
        doc=doc, metric_file=str(target), target_metric_names={"retained"}, compiled_metric_catalog=None
    )
    rows[0]["uid"] = "previous-kb-identity"
    node.generation_tools.metric_rag.upsert_batch(rows)
    node.generation_evidence.record_semantic_artifact_validation("uid_model", target, validation_scope="all")

    def validate_current(**kwargs):
        current = yaml.safe_load(target.read_bytes())["semantic_model"][0]["metrics"]
        assert all(metric_uid(metric) for metric in current)
        assert {metric["name"] for metric in current} == set(node._semantic_validation_metric_names())
        _record_current_compiled_evidence(node, target)
        return FuncToolResult(result={"valid": True, "scope": "all"})

    node.semantic_tools.validate_semantic = MagicMock(side_effect=validate_current)
    original_content = target.read_bytes()
    ctx = StreamRunContext(
        user_input=SemanticNodeInput(user_message="Update the model"), action_history_manager=ActionHistoryManager()
    )
    ctx.response_content = {"status": "generated", "output": "Ready"}
    result = node._build_success_result(ctx)
    assert result.success is True
    assert result.status == "generated"
    current = yaml.safe_load(target.read_bytes())["semantic_model"][0]["metrics"]
    identities = {metric["name"]: metric_uid(metric) for metric in current}
    assert identities["retained"] == retained_uid
    assert len(identities["legacy"]) == 24
    assert (target.read_bytes() == original_content) is legacy_has_uid
    assert node.osi_target_state.touched_metric_names == ["legacy", "retained"]
    kb_rows = node.generation_tools.metric_rag.list_artifact_rows(str(target))
    assert {row["name"]: row["uid"] for row in kb_rows} == identities
    node.semantic_tools.validate_semantic.assert_called_once()
    assert node.generation_evidence.compiled_validation_passed(str(target), list(identities))
    assert node.osi_target_state.artifact_snapshot_path == ""
    content = target.read_bytes()
    assert node._build_success_result(ctx).success is True
    assert target.read_bytes() == content
    node.semantic_tools.validate_semantic.assert_called_once()


@pytest.mark.parametrize("second_name", ["order_count", "REVENUE"])
def test_finalizer_enforces_kb_name_uniqueness_before_uid_or_kb_writes(real_agent_config, mock_llm_create, second_name):
    from datus.agent.node.semantic_modeling_agentic_node import SemanticModelingAgenticNode
    from datus.tools.semantic_tools.dosi.authoring import validate_dosi_document
    from datus.tools.semantic_tools.metric_identity import metric_uid

    node = SemanticModelingAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "uid_name_validation.yml"
    document = {
        "version": "0.2.0.dev0",
        "semantic_model": [
            {
                "name": "uid_name_validation",
                "datasets": [
                    {
                        "name": "orders",
                        "source": "orders",
                        "fields": [
                            {
                                "name": "amount",
                                "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "amount"}]},
                            }
                        ],
                    }
                ],
                "metrics": [
                    {
                        "name": name,
                        "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "SUM(orders.amount)"}]},
                    }
                    for name in ("revenue", second_name)
                ],
            }
        ],
    }
    # Native Dosi accepts both names; the host also needs the KB's uniqueness rule.
    validate_dosi_document(document)
    target.write_text(yaml.safe_dump(document, sort_keys=False))
    original = target.read_bytes()
    node.osi_target_state.select(
        {
            "semantic_model_name": "uid_name_validation",
            "semantic_model_file": str(target),
            "absolute_path": str(target.resolve()),
            "artifact_sha256": hashlib.sha256(original).hexdigest(),
        },
        mode="bound",
    )
    if second_name == "REVENUE":
        with patch.object(node.generation_tools, "sync_osi_to_db") as sync:
            with pytest.raises(ValueError, match="Metric name conflict"):
                node._finalize_selected_osi_artifact()
            sync.assert_not_called()
        assert target.read_bytes() == original
        assert not node.osi_target_state.touched_metric_names
        assert not node.osi_target_state.artifact_snapshot_path
        assert not node.generation_tools.metric_rag.list_artifact_rows(str(target))
    else:
        assert node._finalize_selected_osi_artifact() == str(target)
        metrics = yaml.safe_load(target.read_bytes())["semantic_model"][0]["metrics"]
        identities = {metric["name"]: metric_uid(metric) for metric in metrics}
        assert len(set(identities.values())) == 2
        assert all(identities.values())
        kb_rows = node.generation_tools.metric_rag.list_artifact_rows(str(target))
        assert {row["name"]: row["uid"] for row in kb_rows} == identities
