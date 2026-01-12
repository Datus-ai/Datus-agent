# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.cli.tutorial module and related metric/reference_sql init functions."""

from argparse import Namespace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from rich.console import Console

import datus.storage.metric.metric_init as metric_init
import datus.storage.reference_sql.reference_sql_init as reference_sql_init
from datus.cli import tutorial as tutorial_module
from datus.cli.tutorial import BenchmarkTutorial, dict_to_yaml_str
from datus.schemas.action_history import ActionStatus

# =============================================================================
# Test Helpers and Stubs
# =============================================================================


class DummyAgentConfig:
    """Lightweight AgentConfig stand-in used to satisfy process_line dependencies."""

    def __init__(self):
        self.db_type = "sqlite"
        self.current_namespace = "test_namespace"
        self._db_config = SimpleNamespace(catalog="catalog", database="database", schema="schema")

    def current_db_config(self):
        return self._db_config


class DummyReferenceStorage:
    """Minimal storage stub that exposes the methods init_reference_sql expects."""

    def __init__(self, size: int = 0):
        self.size = size

    def get_reference_sql_size(self):
        return self.size

    def after_init(self):
        return None


class AsyncIteratorStub:
    """Async iterator that either yields preset actions or raises an exception immediately."""

    def __init__(self, actions=None, exc: Exception | None = None):
        self._actions = actions or []
        self._exc = exc
        self._iter = iter(self._actions)

    def __aiter__(self):
        self._iter = iter(self._actions)
        return self

    async def __anext__(self):
        if self._exc:
            raise self._exc
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def patch_node_class(monkeypatch, module, class_name: str, behavior_map: dict[str, object]):
    """Patch an Agentic node class so each node_name follows the provided behavior."""

    class _NodeStub:
        def __init__(self, *args, **kwargs):
            # For new node classes, determine node_name from the class itself
            self.node_name = getattr(self, "NODE_NAME", kwargs.get("node_name"))
            self.input = None

        def execute_stream(self, action_history_manager):
            behavior = behavior_map.get(self.node_name, [])
            if isinstance(behavior, Exception):
                return AsyncIteratorStub(exc=behavior)
            return AsyncIteratorStub(actions=behavior)

    monkeypatch.setattr(module, class_name, _NodeStub)


# =============================================================================
# Tests for dict_to_yaml_str
# =============================================================================


class TestDictToYamlStr:
    """Tests for dict_to_yaml_str helper function."""

    def test_converts_simple_dict(self):
        data = {"key": "value", "number": 42}
        result = dict_to_yaml_str(data)
        assert "key: value" in result
        assert "number: 42" in result

    def test_converts_nested_dict(self):
        data = {"outer": {"inner": "value"}}
        result = dict_to_yaml_str(data)
        assert "outer:" in result
        assert "inner: value" in result

    def test_returns_empty_string_on_invalid_input(self):
        # Create an object that cannot be serialized by yaml.safe_dump
        class Unserializable:
            pass

        data = {"obj": Unserializable()}
        result = dict_to_yaml_str(data)
        assert result == ""

    def test_handles_empty_dict(self):
        result = dict_to_yaml_str({})
        assert result == "{}\n"


# =============================================================================
# Tests for BenchmarkTutorial
# =============================================================================


class TestBenchmarkTutorialInit:
    """Tests for BenchmarkTutorial.__init__."""

    def test_init_with_config_path(self):
        tutorial = BenchmarkTutorial(config_path="/path/to/config.yml")
        assert tutorial.config_path == "/path/to/config.yml"
        assert tutorial.namespace_name == "california_schools"
        assert isinstance(tutorial.console, Console)

    def test_init_with_none_config_path(self):
        tutorial = BenchmarkTutorial(config_path=None)
        assert tutorial.config_path is None


class TestBenchmarkTutorialEnsureConfig:
    """Tests for BenchmarkTutorial._ensure_config."""

    def test_returns_false_when_config_file_not_found(self, tmp_path):
        tutorial = BenchmarkTutorial(config_path=str(tmp_path / "nonexistent.yml"))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        result = tutorial._ensure_config()

        assert result is False
        output = buffer.getvalue()
        assert "not found" in output

    def test_returns_true_when_config_exists_and_namespace_configured(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        mock_agent_config = MagicMock()
        mock_agent_config.home = str(tmp_path)
        mock_agent_config.benchmark_configs = {"california_schools": {}}
        mock_agent_config.namespaces = {"california_schools": {}}

        mock_path_manager = MagicMock()
        mock_path_manager.benchmark_dir = tmp_path / "benchmark"

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda config: mock_agent_config)
        monkeypatch.setattr(tutorial_module, "get_path_manager", lambda datus_home: mock_path_manager)

        tutorial = BenchmarkTutorial(config_path=str(config_file))

        result = tutorial._ensure_config()

        assert result is True
        assert tutorial.benchmark_path == tmp_path / "benchmark"

    def test_adds_namespace_and_benchmark_config_when_missing(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        mock_agent_config = MagicMock()
        mock_agent_config.home = str(tmp_path)
        mock_agent_config.benchmark_configs = {}
        mock_agent_config.namespaces = {}

        mock_path_manager = MagicMock()
        mock_path_manager.benchmark_dir = tmp_path / "benchmark"

        mock_config_manager = MagicMock()

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda config: mock_agent_config)
        monkeypatch.setattr(tutorial_module, "get_path_manager", lambda datus_home: mock_path_manager)
        # Mock the configuration_manager in the agent_config_loader module where it's imported from
        monkeypatch.setattr(
            "datus.configuration.agent_config_loader.configuration_manager",
            lambda: mock_config_manager,
        )

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        result = tutorial._ensure_config()

        assert result is True
        assert mock_config_manager.update_item.call_count == 2
        # First call for namespace config
        first_call = mock_config_manager.update_item.call_args_list[0]
        assert first_call[0][0] == "namespace"
        # Second call for benchmark config
        second_call = mock_config_manager.update_item.call_args_list[1]
        assert second_call[0][0] == "benchmark"


class TestBenchmarkTutorialEnsureFiles:
    """Tests for BenchmarkTutorial._ensure_files."""

    def test_creates_benchmark_directory_if_not_exists(self, tmp_path, monkeypatch):
        benchmark_path = tmp_path / "benchmark"

        mock_copy_data_file = MagicMock()
        monkeypatch.setattr(
            "datus.cli.interactive_init.copy_data_file",
            mock_copy_data_file,
        )

        tutorial = BenchmarkTutorial(config_path=None)
        tutorial.benchmark_path = benchmark_path

        tutorial._ensure_files()

        assert benchmark_path.exists()
        sub_path = benchmark_path / "california_schools"
        assert sub_path.exists()
        mock_copy_data_file.assert_called_once()

    def test_uses_existing_benchmark_directory(self, tmp_path, monkeypatch):
        benchmark_path = tmp_path / "benchmark"
        benchmark_path.mkdir()
        sub_path = benchmark_path / "california_schools"
        sub_path.mkdir()

        mock_copy_data_file = MagicMock()
        monkeypatch.setattr(
            "datus.cli.interactive_init.copy_data_file",
            mock_copy_data_file,
        )

        tutorial = BenchmarkTutorial(config_path=None)
        tutorial.benchmark_path = benchmark_path

        tutorial._ensure_files()

        mock_copy_data_file.assert_called_once_with(
            resource_path="sample_data/california_schools",
            target_dir=sub_path,
        )


class TestBenchmarkTutorialAddSubAgents:
    """Tests for BenchmarkTutorial.add_sub_agents."""

    def test_adds_two_sub_agents(self, tmp_path, monkeypatch):
        mock_agent_config = MagicMock()
        mock_manager = MagicMock()

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda reload: mock_agent_config)
        monkeypatch.setattr(tutorial_module, "configuration_manager", lambda config_path, reload: MagicMock())
        monkeypatch.setattr(tutorial_module, "SubAgentManager", lambda **kwargs: mock_manager)

        tutorial = BenchmarkTutorial(config_path=str(tmp_path / "config.yml"))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        tutorial.add_sub_agents()

        assert mock_manager.save_agent.call_count == 2

        # Check first agent (datus_schools)
        first_call = mock_manager.save_agent.call_args_list[0]
        first_config = first_call[0][0]
        assert first_config.system_prompt == "datus_schools"
        assert first_config.tools == "db_tools, date_parsing_tools"
        assert first_call[1]["previous_name"] == "datus_schools"

        # Check second agent (datus_schools_context)
        second_call = mock_manager.save_agent.call_args_list[1]
        second_config = second_call[0][0]
        assert second_config.system_prompt == "datus_schools_context"
        assert "context_search_tools" in second_config.tools
        assert second_call[1]["previous_name"] == "datus_schools_context"

        output = buffer.getvalue()
        assert "datus_schools" in output
        assert "datus_schools_context" in output


class TestBenchmarkTutorialAddWorkflows:
    """Tests for BenchmarkTutorial.add_workflows."""

    def test_adds_workflows(self, tmp_path, monkeypatch):
        mock_config_manager = MagicMock()

        monkeypatch.setattr(
            tutorial_module,
            "configuration_manager",
            lambda config_path, reload: mock_config_manager,
        )

        tutorial = BenchmarkTutorial(config_path=str(tmp_path / "config.yml"))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        tutorial.add_workflows()

        mock_config_manager.update_item.assert_called_once()
        call_args = mock_config_manager.update_item.call_args

        assert call_args[0][0] == "workflow"
        workflows = call_args[1]["value"]
        assert "datus_schools" in workflows
        assert "datus_schools_context" in workflows
        assert workflows["datus_schools"] == ["datus_schools", "execute_sql", "output"]
        assert workflows["datus_schools_context"] == ["datus_schools_context", "execute_sql", "output"]

        output = buffer.getvalue()
        assert "datus_schools" in output


class TestBenchmarkTutorialRun:
    """Tests for BenchmarkTutorial.run."""

    @pytest.mark.acceptance
    def test_run_returns_1_when_ensure_config_fails(self, tmp_path, monkeypatch):
        tutorial = BenchmarkTutorial(config_path=str(tmp_path / "nonexistent.yml"))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        result = tutorial.run()

        assert result == 1

    @pytest.mark.acceptance
    def test_run_returns_1_on_exception(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        def raise_error(config):
            raise RuntimeError("Test error")

        monkeypatch.setattr(tutorial_module, "load_agent_config", raise_error)
        monkeypatch.setattr(tutorial_module, "print_rich_exception", MagicMock())

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        result = tutorial.run()

        assert result == 1

    @pytest.mark.acceptance
    def test_run_completes_successfully(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        mock_agent_config = MagicMock()
        mock_agent_config.home = str(tmp_path)
        mock_agent_config.benchmark_configs = {"california_schools": {}}
        mock_agent_config.namespaces = {"california_schools": {}}

        mock_path_manager = MagicMock()
        benchmark_path = tmp_path / "benchmark"
        benchmark_path.mkdir()
        mock_path_manager.benchmark_dir = benchmark_path

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda config=None, reload=False: mock_agent_config)
        monkeypatch.setattr(tutorial_module, "get_path_manager", lambda datus_home: mock_path_manager)

        # Mock all the initialization functions
        mock_copy_data_file = MagicMock()
        monkeypatch.setattr("datus.cli.interactive_init.copy_data_file", mock_copy_data_file)

        mock_init_metadata = MagicMock()
        mock_init_sql = MagicMock()
        monkeypatch.setattr("datus.cli.interactive_init.init_metadata_and_log_result", mock_init_metadata)
        monkeypatch.setattr("datus.cli.interactive_init.init_sql_and_log_result", mock_init_sql)

        # Mock _init_metrics to return success
        monkeypatch.setattr(BenchmarkTutorial, "_init_metrics", lambda self, path: True)

        # Mock add_sub_agents and add_workflows
        monkeypatch.setattr(BenchmarkTutorial, "add_sub_agents", lambda self: None)
        monkeypatch.setattr(BenchmarkTutorial, "add_workflows", lambda self: None)

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        result = tutorial.run()

        assert result == 0
        output = buffer.getvalue()
        assert "Welcome to Datus benchmark" in output


class TestBenchmarkTutorialInitMetrics:
    """Tests for BenchmarkTutorial._init_metrics."""

    def test_init_metrics_success(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        mock_agent_config = MagicMock()
        mock_agent_config.current_namespace = "california_schools"
        mock_agent_config.rag_storage_path.return_value = str(tmp_path / "storage")

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda reload, config: mock_agent_config)

        mock_parse_subject_tree = MagicMock(return_value=[])
        monkeypatch.setattr(tutorial_module, "parse_subject_tree", mock_parse_subject_tree)

        mock_init_metrics = MagicMock(return_value=(True, "", ""))
        monkeypatch.setattr(
            "datus.storage.metric.metric_init.init_success_story_metrics",
            mock_init_metrics,
        )

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        tutorial.benchmark_path = tmp_path
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        success_path = tmp_path / "success_story.csv"
        success_path.write_text("sql,question\nSELECT 1,test")

        result = tutorial._init_metrics(success_path)

        assert result is True
        output = buffer.getvalue()
        assert "OK" in output or "Metrics initialized" in output

    def test_init_metrics_failure(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        mock_agent_config = MagicMock()
        mock_agent_config.current_namespace = "california_schools"
        mock_agent_config.rag_storage_path.return_value = str(tmp_path / "storage")

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda reload, config: mock_agent_config)

        mock_parse_subject_tree = MagicMock(return_value=[])
        monkeypatch.setattr(tutorial_module, "parse_subject_tree", mock_parse_subject_tree)

        mock_init_metrics = MagicMock(return_value=(False, "Failed to process metrics"))
        monkeypatch.setattr(
            "datus.storage.metric.metric_init.init_success_story_metrics",
            mock_init_metrics,
        )

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        tutorial.benchmark_path = tmp_path
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        success_path = tmp_path / "success_story.csv"
        success_path.write_text("sql,question\nSELECT 1,test")

        result = tutorial._init_metrics(success_path)

        assert result is False
        output = buffer.getvalue()
        assert "Error" in output or "failed" in output.lower()

    def test_init_metrics_handles_exception(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        def raise_error(*args, **kwargs):
            raise RuntimeError("Test exception")

        monkeypatch.setattr(tutorial_module, "load_agent_config", raise_error)
        monkeypatch.setattr(tutorial_module, "print_rich_exception", MagicMock())

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        tutorial.benchmark_path = tmp_path
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        success_path = tmp_path / "success_story.csv"

        result = tutorial._init_metrics(success_path)

        assert result is False

    def test_init_metrics_deletes_existing_storage(self, tmp_path, monkeypatch):
        config_file = tmp_path / "agent.yml"
        config_file.write_text("agent: {}")

        storage_path = tmp_path / "storage"
        storage_path.mkdir()
        semantic_model_path = storage_path / "semantic_model.lance"
        semantic_model_path.mkdir()
        metrics_path = storage_path / "metrics.lance"
        metrics_path.mkdir()

        mock_agent_config = MagicMock()
        mock_agent_config.current_namespace = "california_schools"
        mock_agent_config.rag_storage_path.return_value = str(storage_path)

        monkeypatch.setattr(tutorial_module, "load_agent_config", lambda reload, config: mock_agent_config)

        mock_parse_subject_tree = MagicMock(return_value=[])
        monkeypatch.setattr(tutorial_module, "parse_subject_tree", mock_parse_subject_tree)

        mock_init_metrics = MagicMock(return_value=(True, "", "mock.yaml"))
        monkeypatch.setattr(
            "datus.storage.metric.metric_init.init_success_story_metrics",
            mock_init_metrics,
        )

        tutorial = BenchmarkTutorial(config_path=str(config_file))
        tutorial.benchmark_path = tmp_path
        buffer = StringIO()
        tutorial.console = Console(file=buffer, force_terminal=False, color_system=None)

        success_path = tmp_path / "success_story.csv"
        success_path.write_text("sql,question\nSELECT 1,test")

        result = tutorial._init_metrics(success_path)

        assert result is True
        # Verify existing directories were deleted
        assert not semantic_model_path.exists()
        assert not metrics_path.exists()


# =============================================================================
# Tests for metric_init functions
# =============================================================================


@pytest.mark.acceptance
def test_init_success_story_metrics_returns_error_on_exception(monkeypatch):
    """Test that init_success_story_metrics returns error when GenMetricsAgenticNode fails."""
    df = pd.DataFrame(
        [
            {"sql": "SELECT * FROM schools", "question": "Q1"},
            {"sql": "SELECT * FROM students", "question": "Q2"},
        ]
    )
    monkeypatch.setattr(metric_init.pd, "read_csv", lambda path: df)

    # Create a node class that raises an exception during execute_stream
    class FailingNode:
        def __init__(self, *args, **kwargs):
            self.input = None

        def execute_stream(self, action_history_manager):
            return AsyncIteratorStub(exc=RuntimeError("metrics generation failed"))

    monkeypatch.setattr(metric_init, "GenMetricsAgenticNode", FailingNode)

    args = Namespace(success_story="anything.csv")
    success, error_message, _ = metric_init.init_success_story_metrics(args, DummyAgentConfig())

    assert success is False
    assert "metrics generation failed" in error_message


def test_init_success_story_metrics_success(monkeypatch):
    """Test that init_success_story_metrics returns success when processing completes."""
    df = pd.DataFrame(
        [
            {"sql": "SELECT * FROM schools", "question": "Q1"},
        ]
    )
    monkeypatch.setattr(metric_init.pd, "read_csv", lambda path: df)

    # Create a node class that succeeds
    class SuccessNode:
        def __init__(self, *args, **kwargs):
            self.input = None

        def execute_stream(self, action_history_manager):
            action = SimpleNamespace(
                status=ActionStatus.SUCCESS,
                output={"metrics": []},
                messages="Metrics extracted",
            )
            return AsyncIteratorStub(actions=[action])

    monkeypatch.setattr(metric_init, "GenMetricsAgenticNode", SuccessNode)

    args = Namespace(success_story="anything.csv")
    success, error_message, _ = metric_init.init_success_story_metrics(args, DummyAgentConfig())

    assert success is True
    assert error_message == ""


# =============================================================================
# Tests for reference_sql_init functions
# =============================================================================


@pytest.mark.asyncio
async def test_process_sql_item_returns_none_when_node_raises(monkeypatch):
    behavior_map = {"gen_sql_summary": RuntimeError("summary failure")}
    patch_node_class(monkeypatch, reference_sql_init, "SqlSummaryAgenticNode", behavior_map)

    item = {"sql": "SELECT * FROM schools", "comment": "desc", "filepath": "file.sql"}

    result = await reference_sql_init.process_sql_item(item, SimpleNamespace(), build_mode="overwrite")

    assert result is None


@pytest.mark.asyncio
async def test_process_sql_item_returns_none_without_output(monkeypatch):
    action = SimpleNamespace(status=ActionStatus.SUCCESS, output={}, messages="")
    behavior_map = {"gen_sql_summary": [action]}
    patch_node_class(monkeypatch, reference_sql_init, "SqlSummaryAgenticNode", behavior_map)

    item = {"sql": "SELECT * FROM schools", "comment": "desc", "filepath": "file.sql"}

    result = await reference_sql_init.process_sql_item(item, SimpleNamespace(), build_mode="overwrite")

    assert result is None


@pytest.mark.acceptance
def test_init_reference_sql_reports_process_errors(monkeypatch):
    valid_items = [{"sql": "SELECT * FROM schools", "comment": "desc"}]
    monkeypatch.setattr(reference_sql_init, "process_sql_files", lambda sql_dir: (valid_items, []))

    async def failing_process_sql_item(*args, **kwargs):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(reference_sql_init, "process_sql_item", failing_process_sql_item)

    storage = DummyReferenceStorage()

    result = reference_sql_init.init_reference_sql(
        storage,
        global_config=SimpleNamespace(),
        build_mode="overwrite",
        sql_dir="dummy",
        validate_only=False,
        pool_size=1,
        subject_tree=None,
    )

    assert result["status"] == "success"
    assert result["processed_entries"] == 0
    # Field name is process_errors (plural), not process_error
    assert result["process_errors"] is not None
    assert "SQL processing failed with exception `worker failure`" in result["process_errors"]
