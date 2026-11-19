# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Run a dependency-free plugin against the actual host, Dosi, SQLite and KB."""

import csv
import io
from pathlib import Path

import pytest
import yaml

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.storage.metric.store import MetricRAG
from datus.tools.permission.profiles import get_profile
from datus.tools.semantic_tools.metric_identity import metric_uid
from datus.utils.exceptions import DatusException
from datus.utils.query_result_text import parse_query_result_text
from tests.unit_tests.mock_llm_model import MockLLMResponse
from tests.unit_tests.plugins.conftest import plugin_env as plugin_env

_RUNTIME = """
class Runtime:
    def __init__(self, config, host):
        self.host, self.config = host, config

    async def call(self, name, arguments):
        result = await self.host.call_tool("semantic_tools." + name, arguments)
        if not result["success"]:
            raise ValueError(result["error"])
        return result["result"]

    async def prepare_turn(self, context):
        await self.call("list_existing_osi_semantic_models", {})
        selected = await self.call("plan_osi_semantic_model_target", {"semantic_model_name": "school_model"})
        path = selected["semantic_model_file"]
        await self.call("upsert_osi_datasets", {"path": path, "datasets_json": [{
            "name": "schools", "source": "main.schools", "description": self.config.get("description", "Schools"),
            "fields": [{"name": "CDSCode", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "CDSCode"}]}}]
        }]})
        await self.call("upsert_osi_metrics", {"path": path, "metrics_json": [{
            "name": "school_count", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "COUNT(*)"}]},
            "custom_extensions": [{"vendor_name": "DATUS", "data": '{"v":"1.4","dataset":"schools","subject_path":["schools"]}'}]
        }]})
        return {"context": {"selected": path}}

    async def finalize(self, result, context):
        if self.config.get("premature_complete"):
            await self.host.semantic_authoring.complete()
        await self.host.semantic_authoring.prepare()
        await self.host.semantic_authoring.validate()
        snapshot = await self.host.semantic_authoring.reconcile()
        if self.config.get("mutate_after_reconcile"):
            await self.call("upsert_osi_datasets", {"path": snapshot["semantic_model_file"], "datasets_json": [{
                "name": "schools", "source": "main.schools", "description": "Late mutation", "fields": []
            }]})
        if not self.config.get("omit_complete"):
            snapshot = await self.host.semantic_authoring.complete()
        return {"semantic_models": [snapshot["semantic_model_file"]], "status": "generated", "snapshot": snapshot}


def create_runtime(config, host):
    return Runtime(config, host)
"""


def _plugin_node(config, plugin_env, *, files=None, profile="dangerous", **runtime_config):
    plugin_env(
        "semantic-private",
        """manifest_version: 1
agents:
  semantic_modeling:
    description: Author semantic models with a private runtime.
    runtime_factory: '{pkg}.runtime:create_runtime'
""",
        files={"runtime.py": _RUNTIME},
    )
    config.active_profile_name = profile
    config.permissions_config = get_profile(profile)
    config.agentic_nodes["semantic_modeling"] = {
        "node_class": "configurable",
        "source": {"plugin": "semantic-private", "agent": "semantic_modeling"},
        "runtime_config": runtime_config,
        "tools": [
            "semantic_tools.list_existing_osi_semantic_models",
            "semantic_tools.plan_osi_semantic_model_target",
            "semantic_tools.upsert_osi_datasets",
            "semantic_tools.upsert_osi_metrics",
            "semantic_tools.validate_semantic",
            "filesystem_tools.read_file",
        ],
        "scoped_context": {
            "datasource": config.current_datasource,
            "project_files": files
            if files is not None
            else [f"subject/semantic_models/{config.current_datasource}/**"],
        },
    }
    return create_interactive_node("semantic_modeling", config, execution_mode="workflow")


async def _run(node):
    node.input = create_node_input("Create a school count model", node)
    return [action async for action in node.execute_stream()]


@pytest.mark.asyncio
async def test_private_runtime_authors_and_queries_using_only_injected_host(
    real_agent_config, mock_llm_create, plugin_env
):
    description = "School metadata " * 1000
    node = _plugin_node(real_agent_config, plugin_env, description=description)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Authored school count")])
    try:
        actions = await _run(node)
        assert node.result.success is True, node.result.error
        result = actions[-1].output
        path = Path(real_agent_config.project_root) / result["semantic_models"][0]
        document = yaml.safe_load(path.read_text())
        assert result["snapshot"]["document"] == document
        assert result["snapshot"]["document"]["datasets"][0]["description"] == description
        uid = metric_uid(document["metrics"][0])
        assert len(uid) == 24
        rows = MetricRAG(node.agent_config).search_all_metrics()
        assert [(item["name"], item["uid"]) for item in rows] == [("school_count", uid)]
        queried = node.semantic_tools.query_metrics(metrics=["school_count"])
        assert queried.success == 1, queried.error
        cached = node.semantic_tools.get_cached_query_metrics_result(parse_query_result_text(queried.result).result_id)
        rows = list(csv.DictReader(io.StringIO(cached["csv"])))
        oracle = node._semantic_authoring_session.db_func_tool.connector.execute_query(
            "SELECT COUNT(*) AS school_count FROM schools", result_format="list"
        )
        assert oracle.success is True, oracle.error
        assert len(rows) == 1
        assert float(rows[0]["school_count"]) == oracle.sql_return[0]["school_count"]
        # Required reference files are readable without granting unrelated files.
        references = list((Path(node._resolve_session_data_dir()) / "semantic_authoring").rglob("osi-core.yaml"))
        assert len(references) == 1
        read = await node._host.call_tool("filesystem_tools.read_file", {"path": str(references[0])})
        assert read["success"] == 1, read
        with pytest.raises(DatusException, match="project_files"):
            await node._host.call_tool("filesystem_tools.read_file", {"path": "unselected.txt"})
        assert node.agent_config.current_datasource == real_agent_config.current_datasource
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_failed_kb_reconciliation_preserves_failure_and_rolls_back_new_artifact(
    real_agent_config, mock_llm_create, plugin_env, monkeypatch
):
    from datus.tools.func_tool.generation_tools import GenerationTools

    node = _plugin_node(real_agent_config, plugin_env)
    monkeypatch.setattr(
        GenerationTools, "sync_osi_to_db", lambda *args, **kwargs: {"success": False, "error": "KB unavailable"}
    )
    mock_llm_create.reset(responses=[MockLLMResponse(content="Authored school count")])
    try:
        await _run(node)
        assert node.result.success is False
        assert "KB unavailable" in node.result.error
        assert not list(
            real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource).rglob(
                "school_model.yml"
            )
        )
        assert MetricRAG(node.agent_config).search_all_metrics() == []
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_config,error",
    [
        ({"omit_complete": True}, "Complete semantic authoring"),
        ({"mutate_after_reconcile": True}, "reconciliation"),
    ],
)
async def test_post_reconciliation_failure_keeps_kb_matched_to_published_artifact(
    real_agent_config, mock_llm_create, plugin_env, runtime_config, error
):
    node = _plugin_node(real_agent_config, plugin_env, **runtime_config)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Authored school count")])
    try:
        await _run(node)
        assert node.result.success is False
        assert error in node.result.error
        target = Path(node._semantic_authoring_session.osi_target_state.selected_path)
        document = yaml.safe_load(target.read_text())
        assert document["datasets"][0]["description"] == "Schools"
        assert [field["name"] for field in document["datasets"][0]["fields"]] == ["CDSCode"]
        rows = MetricRAG(node.agent_config).search_all_metrics()
        assert [(item["name"], item["uid"]) for item in rows] == [("school_count", metric_uid(document["metrics"][0]))]
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,files,error",
    [
        ("normal", None, "permission"),
        ("dangerous", ["public/**"], "project_files"),
    ],
)
async def test_private_runtime_cannot_bypass_caller_permissions_or_file_scope(
    real_agent_config, mock_llm_create, plugin_env, profile, files, error
):
    node = _plugin_node(real_agent_config, plugin_env, profile=profile, files=files)
    try:
        await _run(node)
        assert node.result.success is False
        assert error in node.result.error.lower()
        assert mock_llm_create.call_history == []
        assert not list(
            real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource).rglob("*.yml")
        )
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_completion_before_reconciliation_fails_and_restores_the_target(
    real_agent_config, mock_llm_create, plugin_env
):
    node = _plugin_node(real_agent_config, plugin_env, premature_complete=True)
    mock_llm_create.reset(responses=[MockLLMResponse(content="Authored school count")])
    try:
        await _run(node)
        assert node.result.success is False
        assert "has not been reconciled" in node.result.error
        assert not list(
            real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource).rglob("*.yml")
        )
        assert MetricRAG(node.agent_config).search_all_metrics() == []
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_authoring_inventory_and_target_selection_respect_the_selected_file(
    real_agent_config, mock_llm_create, plugin_env
):
    model_dir = real_agent_config.path_manager.semantic_model_path(real_agent_config.current_datasource)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "hidden.yml").write_text("name: school_model\ndatasets: [invalid", encoding="utf-8")
    selected = f"subject/semantic_models/{real_agent_config.current_datasource}/school_model.yml"
    node = _plugin_node(real_agent_config, plugin_env, files=[selected])
    mock_llm_create.reset(responses=[MockLLMResponse(content="Authored school count")])
    try:
        await _run(node)
        assert node.result.success is True, node.result.error
        inventory = await node._host.call_tool("semantic_tools.list_existing_osi_semantic_models", {})
        assert inventory["success"] == 1, inventory
        assert inventory["result"]["files_scanned"] == 1
        assert inventory["result"]["issues"] == []
        assert [model["semantic_model_file"] for model in inventory["result"]["semantic_models"]] == [selected]
    finally:
        await node.aclose()


@pytest.mark.asyncio
async def test_pure_config_authoring_receives_host_completion_without_a_private_runtime(
    real_agent_config, mock_llm_create, plugin_env
):
    from tests.unit_tests.mock_llm_model import MockToolCall

    seed = _plugin_node(real_agent_config, plugin_env)
    await seed.aclose()
    declaration = real_agent_config.agentic_nodes["semantic_modeling"]
    declaration.pop("source")
    declaration.pop("runtime_config")
    declaration["instructions"] = "Author the requested semantic model with the selected native tools."
    real_agent_config.agentic_nodes["custom_semantic"] = real_agent_config.agentic_nodes.pop("semantic_modeling")
    node = create_interactive_node("custom_semantic", real_agent_config, execution_mode="workflow")
    target = f"subject/semantic_models/{real_agent_config.current_datasource}/school_model.yml"
    dataset = {
        "name": "schools",
        "source": "main.schools",
        "fields": [{"name": "CDSCode", "expression": {"dialects": [{"dialect": "ANSI_SQL", "expression": "CDSCode"}]}}],
    }
    mock_llm_create.reset(
        responses=[
            MockLLMResponse(
                tool_calls=[
                    MockToolCall("list_existing_osi_semantic_models", {}),
                    MockToolCall("plan_osi_semantic_model_target", {"semantic_model_name": "school_model"}),
                    MockToolCall("upsert_osi_datasets", {"path": target, "datasets_json": [dataset]}),
                ],
                content="Created a reusable dataset",
            )
        ]
    )
    try:
        actions = await _run(node)
        assert node.result.success is True, node.result.error
        assert actions[-1].output["semantic_models"] == [target]
        assert Path(real_agent_config.project_root, target).is_file()
        from datus.storage.semantic_dataset.store import SemanticDatasetRAG

        fields = SemanticDatasetRAG(node.agent_config).list_fields("school_model", "schools")
        assert [field["name"] for field in fields] == ["CDSCode"]
    finally:
        await node.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_cancelled_plugin_turn_rolls_back_authored_files_before_reconciliation(
    real_agent_config, mock_llm_create, plugin_env, monkeypatch
):
    import asyncio

    node = _plugin_node(real_agent_config, plugin_env)
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def stream(**kwargs):
        entered.set()
        await blocked.wait()
        if False:
            yield

    monkeypatch.setattr(mock_llm_create, "generate_with_tools_stream", stream)
    task = asyncio.create_task(_run(node))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        target = Path(node._semantic_authoring_session.osi_target_state.selected_path)
        assert target.is_file()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not target.exists()
        assert MetricRAG(node.agent_config).search_all_metrics() == []
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await node.aclose()
