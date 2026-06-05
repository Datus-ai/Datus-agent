# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.metric.metric_init."""

from enum import Enum
from unittest.mock import MagicMock

import pytest

from datus.storage.metric.metric_init import (
    BIZ_NAME,
    DEFAULT_METRICS_BATCH_SIZE,
    _action_status_value,
    _all_candidate_metrics_satisfied,
    _extract_metric_artifact_ids,
    _generate_metrics_batch,
    _source_provenance_from_row,
    _sync_metric_provenance,
    init_semantic_yaml_metrics,
)

# ---------------------------------------------------------------------------
# _action_status_value
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestActionStatusValue:
    """Tests for the _action_status_value helper."""

    def test_none_status(self):
        """Returns None when action.status is None."""
        action = MagicMock(status=None)
        assert _action_status_value(action) is None

    def test_no_status_attr(self):
        """Returns None when action has no status attribute."""
        action = object()
        assert _action_status_value(action) is None

    def test_enum_status(self):
        """Returns enum .value when status is an Enum."""

        class St(Enum):
            DONE = "done"

        action = MagicMock()
        action.status = St.DONE
        assert _action_status_value(action) == "done"

    def test_string_status(self):
        """Returns str(status) for plain string status."""
        action = MagicMock()
        action.status = "processing"
        assert _action_status_value(action) == "processing"

    def test_object_with_value_attr(self):
        """Returns status.value for objects with value attribute."""

        class CustomStatus:
            value = "custom"

        action = MagicMock()
        action.status = CustomStatus()
        assert _action_status_value(action) == "custom"


# ---------------------------------------------------------------------------
# BIZ_NAME constant
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestBizNameConstant:
    """Tests for module-level constant."""

    def test_biz_name(self):
        """BIZ_NAME is metric_init."""
        assert BIZ_NAME == "metric_init"


# ---------------------------------------------------------------------------
# init_semantic_yaml_metrics - file not found
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSemanticYamlMetrics:
    """Tests for init_semantic_yaml_metrics function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when YAML file does not exist."""
        nonexistent = str(tmp_path / "nonexistent.yaml")
        mock_config = MagicMock()

        success, error = init_semantic_yaml_metrics(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_existing_file_calls_process(self, tmp_path):
        """When file exists, delegates to process_semantic_yaml_file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("tables:\n  - name: test\n")
        mock_config = MagicMock()

        # The import happens inside the function body from the semantic_model package
        from unittest.mock import patch

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_process:
            success, error = init_semantic_yaml_metrics(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_semantic_objects=False)


# ---------------------------------------------------------------------------
# init_success_story_metrics_async - importability and coroutine check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStoryMetricsAsync:
    """Tests for init_success_story_metrics_async importability and interface."""

    def test_async_function_is_importable(self):
        """init_success_story_metrics_async can be imported from the module."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        assert callable(init_success_story_metrics_async)
        assert inspect.iscoroutinefunction(init_success_story_metrics_async)

    def test_async_function_is_coroutine(self):
        """init_success_story_metrics_async is a coroutine function (async def)."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        assert inspect.iscoroutinefunction(init_success_story_metrics_async)

    def test_async_function_signature_has_no_args_param(self):
        """init_success_story_metrics_async signature does not include argparse.Namespace args."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        sig = inspect.signature(init_success_story_metrics_async)
        param_names = list(sig.parameters.keys())
        assert "args" not in param_names
        assert "agent_config" in param_names
        assert "success_story" in param_names

    def test_async_optional_params_present(self):
        """init_success_story_metrics_async exposes subject_tree, emit, extra_instructions."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        sig = inspect.signature(init_success_story_metrics_async)
        param_names = list(sig.parameters.keys())
        assert "subject_tree" in param_names
        assert "emit" in param_names
        assert "extra_instructions" in param_names

    @pytest.mark.asyncio
    async def test_batch_flow_uses_latest_prompt_version(self):
        """Batch flow uses the latest gen_metrics prompt instead of pinning an old version."""
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        captured_input = {}

        mock_node = MagicMock()

        async def fake_execute_stream(action_manager):
            captured_input["input"] = mock_node.input
            action = MagicMock()
            action.status = ActionStatus.SUCCESS
            action.action_type = "gen_metrics_response"
            action.output = {"response": "done"}
            action.messages = "ok"
            yield action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="test_db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.2"

        import pandas as pd

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, error, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
            )

        assert success is True
        node_input = captured_input["input"]
        assert node_input.prompt_version == "1.2", f"Expected latest '1.2', got '{node_input.prompt_version}'"

    @pytest.mark.asyncio
    async def test_batch_flow_failed_action_returns_failure(self):
        """A final node failure must not be masked by earlier successful tool actions."""
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        mock_node = MagicMock()

        async def fake_execute_stream(action_manager):
            tool_action = MagicMock()
            tool_action.status = ActionStatus.SUCCESS
            tool_action.action_type = "write_file"
            tool_action.output = {"raw_output": {"success": 1}}
            tool_action.messages = "tool ok"
            yield tool_action

            failed_action = MagicMock()
            failed_action.status = ActionStatus.FAILED
            failed_action.action_type = "error"
            failed_action.output = {"error": "Metric generation did not publish to Knowledge Base"}
            failed_action.messages = "Metric generation did not publish to Knowledge Base"
            yield failed_action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="test_db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.2"

        import pandas as pd

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, error, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
            )

        assert success is False
        assert result is None
        assert "did not publish" in error

    @pytest.mark.asyncio
    async def test_batch_flow_accepts_standard_response_action_type(self):
        """Batch mode accepts the standard GenMetricsAgenticNode final response action."""
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        mock_node = MagicMock()

        async def fake_execute_stream(action_manager):
            final_action = MagicMock()
            final_action.status = ActionStatus.SUCCESS
            final_action.action_type = "gen_metrics_response"
            final_action.output = {"response": "done"}
            final_action.messages = "ok"
            yield final_action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="test_db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.2"

        import pandas as pd

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, error, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
            )

        assert success is True
        assert error == ""
        assert result == {"response": "done"}

    @pytest.mark.asyncio
    async def test_batch_flow_allows_recoverable_tool_failure(self):
        """A failed intermediate tool action should not abort a later successful metrics response."""
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        mock_node = MagicMock()

        async def fake_execute_stream(action_manager):
            failed_tool_action = MagicMock()
            failed_tool_action.status = ActionStatus.FAILED
            failed_tool_action.action_type = "validate_semantic"
            failed_tool_action.output = {"raw_output": {"success": 0, "error": "invalid yaml"}}
            failed_tool_action.messages = "validation failed"
            yield failed_tool_action

            final_action = MagicMock()
            final_action.status = ActionStatus.SUCCESS
            final_action.action_type = "gen_metrics_response"
            final_action.output = {"response": "done"}
            final_action.messages = "ok"
            yield final_action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="test_db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.2"

        import pandas as pd

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, error, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
            )

        assert success is True
        assert error == ""
        assert result == {"response": "done"}


# ---------------------------------------------------------------------------
# init_success_story_metrics sync wrapper - new signature
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStoryMetricsSync:
    """Tests for init_success_story_metrics sync wrapper with decoupled signature."""

    def test_sync_function_is_importable(self):
        """init_success_story_metrics can be imported."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        assert callable(init_success_story_metrics)
        assert not inspect.iscoroutinefunction(init_success_story_metrics)

    def test_sync_function_is_not_coroutine(self):
        """init_success_story_metrics is a plain sync function."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        assert not inspect.iscoroutinefunction(init_success_story_metrics)

    def test_sync_function_signature_has_no_args_param(self):
        """init_success_story_metrics signature does not include argparse.Namespace args."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        sig = inspect.signature(init_success_story_metrics)
        param_names = list(sig.parameters.keys())
        assert "args" not in param_names
        assert "agent_config" in param_names
        assert "success_story" in param_names

    def test_sync_returns_three_tuple(self, tmp_path):
        """Sync wrapper returns a 3-tuple (bool, str, Optional[dict]) for a missing CSV."""
        from unittest.mock import patch

        from datus.storage.metric.metric_init import init_success_story_metrics

        missing = str(tmp_path / "no_file.csv")
        mock_config = MagicMock()

        # Patch the async function to avoid creating an unawaited coroutine
        with patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            return_value=(False, "file error", None),
        ):
            result = init_success_story_metrics(mock_config, missing)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] is False, f"Expected failure for missing CSV, got success={result[0]}"

    def test_sync_accepts_all_kwargs(self, tmp_path):
        """Sync wrapper accepts subject_tree, emit, extra_instructions kwargs."""
        from unittest.mock import patch

        from datus.storage.metric.metric_init import init_success_story_metrics

        mock_config = MagicMock()
        emit_events = []

        # Patch the async function to avoid creating an unawaited coroutine
        with patch(
            "datus.storage.metric.metric_init.init_success_story_metrics_async",
            return_value=(True, "", {"metrics": []}),
        ):
            result = init_success_story_metrics(
                mock_config,
                "dummy.csv",
                subject_tree=["Finance"],
                emit=emit_events.append,
                extra_instructions="Focus on revenue metrics.",
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] is True, f"Expected success, got {result[0]}"


# ---------------------------------------------------------------------------
# init_success_story_metrics_async — overwrite truncate semantics
# ---------------------------------------------------------------------------


class TestInitSuccessStoryMetricsAsyncOverwriteTruncate:
    """Verify build_mode='overwrite' wipes the metrics store before LLM regeneration."""

    @pytest.mark.asyncio
    async def test_overwrite_calls_truncate_and_skips_existing_probe(self):
        """build_mode='overwrite' calls MetricRAG(...).truncate() once and never consults exists_metrics."""
        from unittest.mock import patch

        import pandas as pd

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        fake_rag_instance = MagicMock()
        rag_factory = MagicMock(return_value=fake_rag_instance)

        mock_node = MagicMock()

        async def fake_execute_stream(_action_manager):
            action = MagicMock()
            action.status = ActionStatus.SUCCESS
            action.action_type = "gen_metrics_response"
            action.output = {"response": "done"}
            action.messages = "ok"
            yield action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.project_name = "unit-test-project"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.store.MetricRAG", rag_factory),
            patch(
                "datus.storage.metric.init_utils.exists_metrics",
                MagicMock(return_value=set()),
            ) as mock_exists,
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, error, _ = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                build_mode="overwrite",
            )

        assert success is True
        assert rag_factory.call_count == 2
        rag_factory.assert_any_call(mock_config)
        fake_rag_instance.truncate.assert_called_once_with()
        mock_exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_incremental_does_not_call_truncate(self):
        """build_mode='incremental' must not call truncate or whole-store exists probe."""
        from unittest.mock import patch

        import pandas as pd

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        fake_rag_instance = MagicMock()
        rag_factory = MagicMock(return_value=fake_rag_instance)

        mock_node = MagicMock()

        async def fake_execute_stream(_action_manager):
            action = MagicMock()
            action.status = ActionStatus.SUCCESS
            action.action_type = "gen_metrics_response"
            action.output = {"response": "done"}
            action.messages = "ok"
            yield action

        mock_node.execute_stream = fake_execute_stream

        mock_config = MagicMock()
        mock_config.project_name = "unit-test-project"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.store.MetricRAG", rag_factory),
            patch(
                "datus.storage.metric.init_utils.exists_metrics",
                MagicMock(return_value=set()),
            ) as mock_exists,
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
            patch("datus.storage.metric.metric_init.pd.read_csv") as mock_read_csv,
        ):
            mock_read_csv.return_value = pd.DataFrame([{"question": "Revenue?", "sql": "SELECT SUM(a) FROM t"}])
            success, _error, _ = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                build_mode="incremental",
            )

        assert success is True
        fake_rag_instance.truncate.assert_not_called()
        mock_exists.assert_not_called()


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestMetricProvenanceHelpers:
    def test_source_provenance_from_row_reads_context_columns(self):
        row = {
            "source_context_id": "metric:seed:7; metric:task:21",
            "source_id": "seed_context.csv:7",
            "source_metadata": '{"task_id": "21"}',
            "question": "delivery activities",
        }

        result = _source_provenance_from_row(row, 7, "/tmp/seed_context.csv")

        assert result == {
            "source_id": "seed_context.csv:7",
            "source_type": "success_story",
            "source_context_ids": ["metric:seed:7", "metric:task:21"],
            "source_metadata": {
                "task_id": "21",
                "source_id": "seed_context.csv:7",
                "source_type": "success_story",
                "row_index": 7,
                "question": "delivery activities",
            },
        }

    def test_extract_metric_artifact_ids_recurses_nested_tool_result(self):
        payload = {
            "result": {
                "sync": {
                    "metric_artifact_ids": ["metric:Sales.activity_count", "metric:Sales.activity_count"],
                }
            }
        }

        assert _extract_metric_artifact_ids(payload) == ["metric:Sales.activity_count"]

    def test_extract_metric_artifact_ids_reads_synced_field(self):
        payload = {"_synced_metric_artifact_ids": ["metric:Sales.activity_count"]}

        assert _extract_metric_artifact_ids(payload) == ["metric:Sales.activity_count"]

    def test_sync_metric_provenance_writes_sidecar(self, tmp_path):
        from types import SimpleNamespace

        from datus.storage.knowledge_provenance import METRIC_ARTIFACT_TYPE, KnowledgeProvenanceStore

        config = SimpleNamespace(
            knowledge_base={"provenance": {"enabled": True}},
            path_manager=SimpleNamespace(project_data_dir=tmp_path),
        )

        written = _sync_metric_provenance(
            config,
            ["metric:Sales.activity_count"],
            [
                {
                    "source_id": "seed_context.csv:0",
                    "source_context_ids": ["metric:seed:0"],
                    "source_metadata": {"row_index": 0},
                }
            ],
        )

        found = KnowledgeProvenanceStore(config).find_by_artifact_ids(
            METRIC_ARTIFACT_TYPE, ["metric:Sales.activity_count"]
        )
        assert written == 1
        assert found["metric:Sales.activity_count"]["source_context_ids"] == ["metric:seed:0"]

    def test_sync_metric_provenance_skips_ambiguous_multi_source_batch(self, tmp_path):
        from types import SimpleNamespace

        from datus.storage.knowledge_provenance import METRIC_ARTIFACT_TYPE, KnowledgeProvenanceStore

        config = SimpleNamespace(
            knowledge_base={"provenance": {"enabled": True}},
            path_manager=SimpleNamespace(project_data_dir=tmp_path),
        )

        written = _sync_metric_provenance(
            config,
            ["metric:Sales.activity_count"],
            [
                {"source_id": "seed_context.csv:0", "source_context_ids": ["metric:seed:0"]},
                {"source_id": "seed_context.csv:1", "source_context_ids": ["metric:seed:1"]},
            ],
        )

        found = KnowledgeProvenanceStore(config).find_by_artifact_ids(
            METRIC_ARTIFACT_TYPE, ["metric:Sales.activity_count"]
        )
        assert written == 0
        assert found == {}


# ---------------------------------------------------------------------------
# DEFAULT_METRICS_BATCH_SIZE constant
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestDefaultMetricsBatchSize:
    def test_default_batch_size(self):
        assert DEFAULT_METRICS_BATCH_SIZE == 5


# ---------------------------------------------------------------------------
# _generate_metrics_batch
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestGenerateMetricsBatch:
    """Tests for the _generate_metrics_batch helper."""

    @pytest.mark.asyncio
    async def test_success_returns_result(self):
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchEventHelper

        mock_node = MagicMock()

        async def fake_stream(_ahm):
            action = MagicMock()
            action.status = ActionStatus.SUCCESS
            action.action_type = "gen_metrics_response"
            action.output = {"metrics": ["revenue"]}
            action.messages = "ok"
            yield action

        mock_node.execute_stream = fake_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
        ):
            ok, err, result = await _generate_metrics_batch(
                ["Query 1:\nQuestion: rev?\nSQL:\nSELECT 1"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
            )

        assert ok is True
        assert err == ""
        assert result == {"metrics": ["revenue"]}

    @pytest.mark.asyncio
    async def test_success_captures_synced_metric_artifact_ids(self):
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchEventHelper

        mock_node = MagicMock()

        async def fake_stream(_ahm):
            tool_action = MagicMock()
            tool_action.status = ActionStatus.SUCCESS
            tool_action.action_type = "end_metric_generation"
            tool_action.output = {"result": {"sync": {"metric_artifact_ids": ["metric:Sales.activity_count"]}}}
            tool_action.messages = "published"
            yield tool_action

            final_action = MagicMock()
            final_action.status = ActionStatus.SUCCESS
            final_action.action_type = "gen_metrics_response"
            final_action.output = {"metrics": ["activity_count"]}
            final_action.messages = "ok"
            yield final_action

        mock_node.execute_stream = fake_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
        ):
            ok, err, result = await _generate_metrics_batch(
                ["Query 1:\nQuestion: rev?\nSQL:\nSELECT 1"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
            )

        assert ok is True
        assert err == ""
        assert result["_synced_metric_artifact_ids"] == ["metric:Sales.activity_count"]

    @pytest.mark.asyncio
    async def test_terminal_error_returns_failure(self):
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchEventHelper

        mock_node = MagicMock()

        async def fake_stream(_ahm):
            action = MagicMock()
            action.status = ActionStatus.FAILED
            action.action_type = "error"
            action.output = None
            action.messages = "Max turns exceeded"
            yield action

        mock_node.execute_stream = fake_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
        ):
            ok, err, result = await _generate_metrics_batch(
                ["Query 1:\nQuestion: q?\nSQL:\nSELECT 1"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
            )

        assert ok is False
        assert "Max turns" in err
        assert result is None

    @pytest.mark.asyncio
    async def test_failed_final_response_returns_failure(self):
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchEventHelper

        mock_node = MagicMock()

        async def fake_stream(_ahm):
            action = MagicMock()
            action.status = ActionStatus.FAILED
            action.action_type = "gen_metrics_response"
            action.output = {"error": "publish failed"}
            action.messages = "publish failed"
            yield action

        mock_node.execute_stream = fake_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
        ):
            ok, err, result = await _generate_metrics_batch(
                ["Query 1:\nQuestion: q?\nSQL:\nSELECT 1"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
            )

        assert ok is False
        assert "publish failed" in err
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_failure(self):
        from unittest.mock import patch

        from datus.schemas.batch_events import BatchEventHelper

        mock_node = MagicMock()

        async def fake_stream(_ahm):
            raise RuntimeError("connection lost")
            yield  # noqa: F841 - async generator marker

        mock_node.execute_stream = fake_stream

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", return_value=mock_node),
        ):
            ok, err, result = await _generate_metrics_batch(
                ["Query 1:\nQuestion: q?\nSQL:\nSELECT 1"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
            )

        assert ok is False
        assert "connection lost" in err

    @pytest.mark.asyncio
    async def test_prompt_includes_precomputed_catalog_and_candidate_plan(self):
        from unittest.mock import patch

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchEventHelper

        captured = {}

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, _ahm):
                captured["message"] = self.input.user_message
                action = MagicMock()
                action.status = ActionStatus.SUCCESS
                action.action_type = "gen_metrics_response"
                action.output = {"metrics": ["revenue"]}
                action.messages = "ok"
                yield action

        mock_config = MagicMock()
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
        ):
            ok, err, _ = await _generate_metrics_batch(
                ["Query 1:\nQuestion: q?\nSQL:\nSELECT SUM(revenue) AS revenue FROM orders"],
                0,
                mock_config,
                None,
                None,
                BatchEventHelper("test", None),
                None,
                candidate_plan_json='{"direct_metric_candidates":[{"name":"revenue"}]}',
                existing_metric_catalog_json='[{"name":"orders"}]',
            )

        assert ok is True
        assert err == ""
        assert "Existing Metric Catalog JSON" in captured["message"]
        assert "Precomputed Metric Candidate Plan JSON" in captured["message"]
        assert "do not call analyze_metric_candidates_from_history again" in captured["message"]


# ---------------------------------------------------------------------------
# init_success_story_metrics_async — batch splitting & partial failure
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestBatchSplitting:
    """Tests for batch splitting and partial failure tolerance."""

    def _make_node_factory(self, fail_on_batches=None):
        """Return a mock GenMetricsAgenticNode factory.

        Args:
            fail_on_batches: set of batch indices (0-based) that should fail.
        """
        from datus.schemas.action_history import ActionStatus

        fail_on_batches = fail_on_batches or set()
        call_count = {"n": 0}

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None
                self._batch_idx = call_count["n"]
                call_count["n"] += 1

            async def execute_stream(self, _ahm):
                if self._batch_idx in fail_on_batches:
                    action = MagicMock()
                    action.status = ActionStatus.FAILED
                    action.action_type = "error"
                    action.output = None
                    action.messages = f"Batch {self._batch_idx} failed"
                    yield action
                else:
                    action = MagicMock()
                    action.status = ActionStatus.SUCCESS
                    action.action_type = "gen_metrics_response"
                    action.output = {"metrics": [f"m{self._batch_idx}"]}
                    action.messages = "ok"
                    yield action

        return MockNode

    @pytest.mark.asyncio
    async def test_queries_split_into_batches(self):
        """15 queries with batch_size=5 → 3 batches, 3 node instantiations."""
        from unittest.mock import patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        MockNode = self._make_node_factory()

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        rows = [{"question": f"q{i}", "sql": f"SELECT {i}"} for i in range(15)]

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                batch_size=5,
            )

        assert ok is True
        assert err == ""
        assert len(result["metrics"]) == 3

    @pytest.mark.asyncio
    async def test_refreshes_existing_metric_catalog_between_batches(self):
        from unittest.mock import patch

        import pandas as pd

        from datus.schemas.action_history import ActionStatus
        from datus.storage.metric.metric_init import init_success_story_metrics_async

        captured_messages = []
        call_count = {"n": 0}

        class MockNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, _ahm):
                captured_messages.append(self.input.user_message)
                batch_idx = call_count["n"]
                call_count["n"] += 1
                action = MagicMock()
                action.status = ActionStatus.SUCCESS
                action.action_type = "gen_metrics_response"
                action.output = {"metrics": [f"m{batch_idx}"]}
                action.messages = "ok"
                yield action

        catalog_snapshots = [
            [],
            [{"name": "first_metric"}],
            [{"name": "first_metric"}, {"name": "second_metric"}],
        ]

        def fake_catalog(_agent_config):
            if catalog_snapshots:
                return catalog_snapshots.pop(0)
            return [{"name": "first_metric"}, {"name": "second_metric"}]

        def fake_candidate_plan(_sql_entries, _catalog_json, _agent_config):
            return {"available": True, "direct_metric_candidates": [], "metric_candidates": []}

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"
        rows = [
            {"question": "q1", "sql": "SELECT COUNT(*) AS first_metric FROM orders"},
            {"question": "q2", "sql": "SELECT COUNT(*) AS second_metric FROM orders"},
        ]

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init._build_existing_metric_catalog", side_effect=fake_catalog),
            patch("datus.storage.metric.metric_init._build_candidate_plan", side_effect=fake_candidate_plan),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                batch_size=1,
            )

        assert ok is True
        assert err == ""
        assert result["metrics"] == ["m0", "m1"]
        assert len(captured_messages) == 2
        assert '"first_metric"' in captured_messages[1]

    @pytest.mark.asyncio
    async def test_metrics_bootstrap_uses_batch_semantic_model_creation(self):
        """Metrics bootstrap asks semantic model auto-create to batch missing tables."""
        from unittest.mock import AsyncMock, patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        MockNode = self._make_node_factory()

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.get_latest_version.return_value = "1.0"
        ensure_semantic = AsyncMock(return_value=(True, "", ["orders", "customers"]))

        rows = [{"question": "revenue", "sql": "SELECT SUM(amount) FROM orders"}]

        with (
            patch(
                "datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value={"orders", "customers"}
            ),
            patch("datus.storage.metric.metric_init.extract_table_sql_evidence", return_value={"orders": ["sql"]}),
            patch("datus.storage.metric.metric_init.ensure_semantic_models_exist", ensure_semantic),
            patch("datus.storage.metric.metric_init._build_existing_metric_catalog", return_value=[]),
            patch("datus.storage.metric.metric_init._build_candidate_plan", return_value={"available": False}),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_prompt_manager),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
            )

        assert ok is True
        assert err == ""
        assert result["metrics"] == ["m0"]
        ensure_semantic.assert_awaited_once()
        assert ensure_semantic.await_args.kwargs["batch_mode"] is True

    @pytest.mark.asyncio
    async def test_partial_batch_failure_continues(self):
        """3 batches, middle one fails → success=True, 2 batches of results merged."""
        from unittest.mock import patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        MockNode = self._make_node_factory(fail_on_batches={1})

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        rows = [{"question": f"q{i}", "sql": f"SELECT {i}"} for i in range(15)]

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                batch_size=5,
            )

        assert ok is True
        assert "batch 2" in err
        assert len(result["metrics"]) == 2
        assert "m0" in result["metrics"]
        assert "m2" in result["metrics"]

    @pytest.mark.asyncio
    async def test_all_batches_fail(self):
        """All batches fail → success=False."""
        from unittest.mock import patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        MockNode = self._make_node_factory(fail_on_batches={0, 1})

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        rows = [{"question": f"q{i}", "sql": f"SELECT {i}"} for i in range(10)]

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                batch_size=5,
            )

        assert ok is False
        assert "All" in err
        assert result is None

    @pytest.mark.asyncio
    async def test_single_row_single_batch(self):
        """1 query → 1 batch, no splitting overhead."""
        from unittest.mock import patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        MockNode = self._make_node_factory()

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")
        mock_pm = MagicMock()
        mock_pm.get_latest_version.return_value = "1.0"

        with (
            patch("datus.storage.metric.store.MetricRAG", MagicMock()),
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch("datus.storage.metric.metric_init.get_prompt_manager", return_value=mock_pm),
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode", MockNode),
            patch(
                "datus.storage.metric.metric_init.pd.read_csv",
                return_value=pd.DataFrame([{"question": "q1", "sql": "SELECT 1"}]),
            ),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                batch_size=10,
            )

        assert ok is True
        assert result == {"metrics": ["m0"]}

    @pytest.mark.asyncio
    async def test_batch_size_parameter_passed_through_sync(self):
        """Sync wrapper forwards batch_size to async function."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        sig = inspect.signature(init_success_story_metrics)
        assert "batch_size" in sig.parameters
        assert sig.parameters["batch_size"].default == DEFAULT_METRICS_BATCH_SIZE

    @pytest.mark.asyncio
    async def test_incremental_skips_batch_when_candidates_already_exist(self):
        """A fully satisfied candidate plan skips gen_metrics without creating an agent node."""
        from unittest.mock import patch

        import pandas as pd

        from datus.storage.metric.metric_init import init_success_story_metrics_async

        mock_config = MagicMock()
        mock_config.project_name = "test"
        mock_config.current_db_config.return_value = MagicMock(catalog="", database="db", schema="")

        rows = [{"question": "revenue", "sql": "SELECT SUM(revenue) AS total_revenue FROM orders"}]
        candidate_plan = {
            "available": True,
            "direct_metric_candidates": [
                {
                    "name": "total_revenue",
                    "source_sql_name": "sql_1",
                    "metric_type": "measure_proxy",
                    "base_measures": [{"name": "total_revenue"}],
                }
            ],
            "derived_metric_candidates": [],
            "metric_candidates": [],
        }

        with (
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list", return_value=[]),
            patch(
                "datus.storage.metric.metric_init._build_existing_metric_catalog",
                return_value=[{"name": "total_revenue", "type": "measure_proxy", "base_measures": ["total_revenue"]}],
            ) as catalog_builder,
            patch(
                "datus.storage.metric.metric_init._build_candidate_plan", return_value=candidate_plan
            ) as plan_builder,
            patch("datus.storage.metric.metric_init.GenMetricsAgenticNode") as node_factory,
            patch("datus.storage.metric.metric_init.pd.read_csv", return_value=pd.DataFrame(rows)),
        ):
            ok, err, result = await init_success_story_metrics_async(
                agent_config=mock_config,
                success_story="dummy.csv",
                build_mode="incremental",
            )

        assert ok is True
        assert err == ""
        assert result["skipped_batches"] == 1
        assert result["skipped_metric_candidates"] == ["total_revenue"]
        catalog_builder.assert_called_once_with(mock_config)
        plan_builder.assert_called_once()
        node_factory.assert_not_called()

    def test_candidate_metrics_satisfied_requires_definition_match(self):
        candidate_plan = {
            "direct_metric_candidates": [
                {
                    "name": "total_revenue",
                    "metric_type": "measure_proxy",
                    "base_measures": [{"name": "total_revenue"}],
                }
            ]
        }

        assert (
            _all_candidate_metrics_satisfied(
                candidate_plan,
                [{"name": "total_revenue", "type": "measure_proxy", "base_measures": ["total_revenue"]}],
            )
            is True
        )
        assert (
            _all_candidate_metrics_satisfied(
                candidate_plan,
                [{"name": "total_revenue", "type": "ratio", "base_measures": ["total_revenue"]}],
            )
            is False
        )
        assert _all_candidate_metrics_satisfied({"direct_metric_candidates": [{"name": "total_revenue"}]}, []) is False
