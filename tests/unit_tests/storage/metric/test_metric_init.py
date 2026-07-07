# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.metric.metric_init."""

from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.storage.metric.metric_init import (
    BIZ_NAME,
    DEFAULT_METRICS_BATCH_SIZE,
    _action_status_value,
    _ensure_semantic_models_for_metrics,
    _extract_metric_artifact_ids,
    _generate_metrics_batch,
    _source_provenance_from_row,
    _sync_metric_provenance,
    _unique_metric_catalog_by_name,
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

        with (
            patch(
                "datus.storage.metric.metric_init._metrics_authoring_format",
                return_value="metricflow",
            ),
            patch(
                "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
                return_value=(True, ""),
            ) as mock_process,
        ):
            success, error = init_semantic_yaml_metrics(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_semantic_objects=False)

    def test_osi_existing_file_calls_osi_metric_sync(self, tmp_path):
        """OSI metric YAML imports use the OSI sync path."""
        yaml_file = tmp_path / "metrics.yaml"
        yaml_file.write_text("metrics:\n  - name: revenue\n")
        mock_config = MagicMock()
        mock_tools = MagicMock()
        mock_tools._sync_osi_metric_to_db.return_value = {"success": True, "message": "synced"}

        with (
            patch(
                "datus.storage.metric.metric_init._metrics_authoring_format",
                return_value="osi",
            ),
            patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=mock_tools) as mock_cls,
        ):
            success, error = init_semantic_yaml_metrics(str(yaml_file), mock_config)

        assert success is True
        assert error == "synced"
        mock_cls.assert_called_once_with(agent_config=mock_config, authoring_format="osi")
        mock_tools._sync_osi_metric_to_db.assert_called_once_with(str(yaml_file))

    def test_osi_existing_file_returns_sync_error(self, tmp_path):
        """OSI sync failures are surfaced as init failures."""
        yaml_file = tmp_path / "metrics.yaml"
        yaml_file.write_text("metrics:\n  - name: revenue\n")
        mock_config = MagicMock()
        mock_tools = MagicMock()
        mock_tools._sync_osi_metric_to_db.return_value = {"success": False, "error": "invalid osi metrics"}

        with (
            patch(
                "datus.storage.metric.metric_init._metrics_authoring_format",
                return_value="osi",
            ),
            patch("datus.tools.func_tool.generation_tools.GenerationTools", return_value=mock_tools),
        ):
            success, error = init_semantic_yaml_metrics(str(yaml_file), mock_config)

        assert success is False
        assert error == "invalid osi metrics"


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
        assert result == {"response": "done", "metrics_count": 0, "final_metrics_count": 0}

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
        assert result == {"response": "done", "metrics_count": 0, "final_metrics_count": 0}


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
        rag_factory.assert_any_call(mock_config)
        fake_rag_instance.truncate.assert_called_once_with()
        mock_exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_incremental_does_not_call_truncate(self):
        """build_mode='incremental' must not call truncate."""
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
# semantic model prerequisites for metric bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestEnsureSemanticModelsForMetrics:
    @pytest.mark.asyncio
    async def test_osi_generates_domain_semantic_model_once(self, tmp_path):
        from unittest.mock import patch

        target_file = "subject/semantic_models/warehouse/warehouse.yml"
        target_path = tmp_path / target_file
        config = SimpleNamespace(
            project_root=str(tmp_path),
            current_datasource="warehouse",
            resolve_semantic_adapter=lambda requested=None: "osi",
        )
        semantic_rag = MagicMock()
        semantic_rag.get_size.return_value = 0
        action_callback = MagicMock()
        captured = {}

        async def fake_init(agent_config, success_story, emit=None, build_mode="overwrite", action_callback=None):
            target_path.parent.mkdir(parents=True)
            target_path.write_text("version: 0.2.0.dev0\nsemantic_model:\n  - name: warehouse\n", encoding="utf-8")
            captured["build_mode"] = build_mode
            captured["action_callback"] = action_callback
            return True, ""

        with (
            patch("datus.storage.semantic_model.store.SemanticModelRAG", return_value=semantic_rag),
            patch(
                "datus.storage.semantic_model.semantic_model_init.init_success_story_semantic_model_async",
                side_effect=fake_init,
            ) as init_mock,
            patch("datus.storage.metric.metric_init.extract_tables_from_sql_list") as extract_mock,
            patch("datus.storage.metric.metric_init.ensure_semantic_models_exist") as ensure_mock,
        ):
            ok, error, created = await _ensure_semantic_models_for_metrics(
                config,
                "success_story.csv",
                [{"sql": "SELECT COUNT(*) FROM orders", "question": "How many orders?"}],
                ["SELECT COUNT(*) FROM orders"],
                action_callback=action_callback,
            )

        assert ok is True
        assert error == ""
        assert created == [target_file]
        init_mock.assert_called_once()
        extract_mock.assert_not_called()
        ensure_mock.assert_not_called()
        assert captured["build_mode"] == "incremental"
        assert captured["action_callback"] is action_callback

    @pytest.mark.asyncio
    async def test_metricflow_keeps_per_table_semantic_model_auto_create(self):
        from unittest.mock import patch

        config = SimpleNamespace(
            resolve_semantic_adapter=lambda requested=None: "metricflow",
        )
        records = [{"sql": "SELECT COUNT(*) FROM orders JOIN customers USING (customer_id)", "question": "Orders?"}]
        sql_list = [records[0]["sql"]]

        async def fake_ensure(tables, agent_config, emit=None, sql_evidence_by_table=None):
            assert tables == ["customers", "orders"]
            assert sql_evidence_by_table == {"orders": records}
            return True, "", tables

        with (
            patch(
                "datus.storage.metric.metric_init.extract_tables_from_sql_list",
                return_value=["customers", "orders"],
            ),
            patch(
                "datus.storage.metric.metric_init.extract_table_sql_evidence",
                return_value={"orders": records},
            ),
            patch("datus.storage.metric.metric_init.ensure_semantic_models_exist", side_effect=fake_ensure),
        ):
            ok, error, created = await _ensure_semantic_models_for_metrics(
                config,
                "success_story.csv",
                records,
                sql_list,
            )

        assert ok is True
        assert error == ""
        assert created == ["customers", "orders"]


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
        assert result == {"metrics": ["m0"], "metrics_count": 0, "final_metrics_count": 0}

    @pytest.mark.asyncio
    async def test_batch_size_parameter_passed_through_sync(self):
        """Sync wrapper forwards batch_size to async function."""
        import inspect

        from datus.storage.metric.metric_init import init_success_story_metrics

        sig = inspect.signature(init_success_story_metrics)
        assert "batch_size" in sig.parameters
        assert sig.parameters["batch_size"].default == DEFAULT_METRICS_BATCH_SIZE


class TestBatchHasNoMetricCandidates:
    """Tests for _batch_has_no_metric_candidates early-skip logic."""

    def test_returns_false_when_plan_unavailable(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        assert _batch_has_no_metric_candidates({}) is False
        assert _batch_has_no_metric_candidates({"available": False}) is False
        assert _batch_has_no_metric_candidates(None) is False

    def test_returns_false_when_direct_candidates_exist(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "direct_metric_candidates": [{"name": "revenue"}],
            "non_metric_evidence": [{"name": "detail_query"}],
        }
        assert _batch_has_no_metric_candidates(plan) is False

    def test_returns_false_when_derived_candidates_exist(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "derived_metric_candidates": [{"name": "mom_delta"}],
            "non_metric_evidence": [{"name": "detail_query"}],
        }
        assert _batch_has_no_metric_candidates(plan) is False

    def test_returns_false_when_llm_review_candidates_exist(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "llm_review_candidates": [
                {
                    "name": "price_per_quantity",
                    "candidate_classification": "llm_review_candidate",
                    "equivalence": "lifted",
                }
            ],
            "non_metric_evidence": [{"name": "detail_query"}],
        }
        assert _batch_has_no_metric_candidates(plan) is False

    def test_returns_true_when_only_non_metric_evidence(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "direct_metric_candidates": [],
            "derived_metric_candidates": [],
            "non_metric_evidence": [{"name": "row_number_query"}],
        }
        assert _batch_has_no_metric_candidates(plan) is True

    def test_returns_true_when_only_identity_references(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "identity_metric_references": [{"name": "activity_count"}],
        }
        assert _batch_has_no_metric_candidates(plan) is True

    def test_returns_true_when_only_derived_datasource_recommendations(self):
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {
            "available": True,
            "derived_datasource_recommendations": [{"sql_query": "SELECT ..."}],
        }
        assert _batch_has_no_metric_candidates(plan) is True

    def test_returns_false_when_no_evidence_at_all(self):
        """Available plan with zero candidates AND zero evidence should NOT skip."""
        from datus.storage.metric.metric_init import _batch_has_no_metric_candidates

        plan = {"available": True}
        assert _batch_has_no_metric_candidates(plan) is False


@pytest.mark.ci
class TestMetricCatalogHelpers:
    def test_unique_metric_catalog_by_name(self):
        unique, ambiguous = _unique_metric_catalog_by_name(
            [
                "not-dict",
                {"name": "activity_count"},
                {"name": "activity_count"},
                {"name": "order_count"},
            ]
        )
        assert unique == {"order_count": {"name": "order_count"}}
        assert ambiguous == {"activity_count"}


@pytest.mark.ci
class TestSourceNames:
    """Tests for _source_names helper."""

    def test_empty_value_returns_empty_set(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names("") == set()
        assert _source_names(None) == set()
        assert _source_names([]) == set()

    def test_list_of_strings(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names(["sql_1", "sql_2"]) == {"sql_1", "sql_2"}

    def test_tuple_of_strings(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names(("sql_1", "sql_2")) == {"sql_1", "sql_2"}

    def test_set_of_strings(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names({"sql_1"}) == {"sql_1"}

    def test_comma_separated_string(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names("sql_1, sql_2") == {"sql_1", "sql_2"}

    def test_single_string(self):
        from datus.storage.metric.metric_init import _source_names

        assert _source_names("sql_1") == {"sql_1"}


@pytest.mark.ci
class TestSourceScopedItems:
    """Tests for _source_scoped_items helper."""

    def test_non_list_returns_empty(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        assert _source_scoped_items(None, {"sql_1"}) == []
        assert _source_scoped_items("not_a_list", {"sql_1"}) == []

    def test_non_dict_items_skipped(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        assert _source_scoped_items(["not_dict", 42], {"sql_1"}) == []

    def test_item_without_source_always_included(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        item = {"name": "m1"}
        result = _source_scoped_items([item], {"sql_1"})
        assert item in result

    def test_item_with_matching_source_included(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        item = {"name": "m1", "source_sql_name": "sql_1"}
        result = _source_scoped_items([item], {"sql_1"})
        assert item in result

    def test_item_with_non_matching_source_excluded(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        item = {"name": "m1", "source_sql_name": "sql_99"}
        result = _source_scoped_items([item], {"sql_1"})
        assert result == []

    def test_uses_source_key_as_fallback(self):
        from datus.storage.metric.metric_init import _source_scoped_items

        item = {"name": "m1", "source": "sql_1"}
        result = _source_scoped_items([item], {"sql_1"})
        assert item in result


@pytest.mark.ci
class TestNormalizedScalar:
    """Tests for _normalized_scalar helper."""

    def test_strips_and_lowercases(self):
        from datus.storage.metric.metric_init import _normalized_scalar

        assert _normalized_scalar("  MeasureProxy  ") == "measureproxy"

    def test_none_returns_empty(self):
        from datus.storage.metric.metric_init import _normalized_scalar

        assert _normalized_scalar(None) == ""

    def test_empty_string(self):
        from datus.storage.metric.metric_init import _normalized_scalar

        assert _normalized_scalar("") == ""


@pytest.mark.ci
class TestNormalizedMetricType:
    """Tests for _normalized_metric_type helper."""

    def test_simple_mapped_to_measure_proxy(self):
        from datus.storage.metric.metric_init import _normalized_metric_type

        assert _normalized_metric_type("simple") == "measure_proxy"

    def test_other_types_unchanged(self):
        from datus.storage.metric.metric_init import _normalized_metric_type

        assert _normalized_metric_type("ratio") == "ratio"
        assert _normalized_metric_type("derived") == "derived"

    def test_none_returns_empty(self):
        from datus.storage.metric.metric_init import _normalized_metric_type

        assert _normalized_metric_type(None) == ""


@pytest.mark.ci
class TestNormalizedMeasureNames:
    """Tests for _normalized_measure_names helper."""

    def test_non_list_returns_empty(self):
        from datus.storage.metric.metric_init import _normalized_measure_names

        assert _normalized_measure_names(None) == set()
        assert _normalized_measure_names("not_a_list") == set()

    def test_list_of_strings(self):
        from datus.storage.metric.metric_init import _normalized_measure_names

        assert _normalized_measure_names(["Revenue", "Cost"]) == {"revenue", "cost"}

    def test_list_of_dicts_with_name_key(self):
        from datus.storage.metric.metric_init import _normalized_measure_names

        result = _normalized_measure_names([{"name": "Revenue"}, {"name": "Cost"}])
        assert result == {"revenue", "cost"}

    def test_list_of_dicts_with_measure_key(self):
        from datus.storage.metric.metric_init import _normalized_measure_names

        result = _normalized_measure_names([{"measure": "Revenue"}])
        assert result == {"revenue"}

    def test_ignores_non_string_non_dict_items(self):
        from datus.storage.metric.metric_init import _normalized_measure_names

        result = _normalized_measure_names([42, None, "revenue"])
        assert result == {"revenue"}


@pytest.mark.ci
class TestCandidateHasDefinitionEvidence:
    """Tests for _candidate_has_definition_evidence helper."""

    def test_empty_candidate_returns_false(self):
        from datus.storage.metric.metric_init import _candidate_has_definition_evidence

        assert _candidate_has_definition_evidence({}) is False

    def test_candidate_with_metric_type(self):
        from datus.storage.metric.metric_init import _candidate_has_definition_evidence

        assert _candidate_has_definition_evidence({"metric_type": "measure_proxy"}) is True

    def test_candidate_with_semantic_model(self):
        from datus.storage.metric.metric_init import _candidate_has_definition_evidence

        assert _candidate_has_definition_evidence({"semantic_model": "orders"}) is True

    def test_candidate_with_base_measures(self):
        from datus.storage.metric.metric_init import _candidate_has_definition_evidence

        assert _candidate_has_definition_evidence({"base_measures": ["order_count"]}) is True

    def test_candidate_with_referenced_metrics(self):
        from datus.storage.metric.metric_init import _candidate_has_definition_evidence

        assert _candidate_has_definition_evidence({"referenced_metrics": ["revenue"]}) is True


@pytest.mark.ci
class TestCandidateMatchesExistingMetric:
    """Tests for _candidate_matches_existing_metric helper."""

    def test_no_definition_evidence_returns_false(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        assert _candidate_matches_existing_metric({}, {"name": "revenue", "type": "measure_proxy"}) is False

    def test_matching_type_and_measures(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        candidate = {"metric_type": "measure_proxy", "base_measures": ["revenue_total"]}
        existing = {"type": "measure_proxy", "base_measures": ["revenue_total"]}
        assert _candidate_matches_existing_metric(candidate, existing) is True

    def test_mismatched_metric_type_returns_false(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        candidate = {"metric_type": "ratio", "base_measures": ["a", "b"]}
        existing = {"type": "measure_proxy", "base_measures": ["a"]}
        assert _candidate_matches_existing_metric(candidate, existing) is False

    def test_mismatched_semantic_model_returns_false(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        candidate = {"metric_type": "measure_proxy", "semantic_model_name": "orders"}
        existing = {"type": "measure_proxy", "semantic_model_name": "customers"}
        assert _candidate_matches_existing_metric(candidate, existing) is False

    def test_mismatched_base_measures_returns_false(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        candidate = {"metric_type": "measure_proxy", "base_measures": ["revenue"]}
        existing = {"type": "measure_proxy", "base_measures": ["cost"]}
        assert _candidate_matches_existing_metric(candidate, existing) is False

    def test_simple_type_normalized_to_measure_proxy(self):
        from datus.storage.metric.metric_init import _candidate_matches_existing_metric

        candidate = {"metric_type": "simple", "base_measures": ["revenue"]}
        existing = {"type": "measure_proxy", "base_measures": ["revenue"]}
        assert _candidate_matches_existing_metric(candidate, existing) is True


@pytest.mark.ci
class TestAllCandidateMetricsSatisfied:
    """Tests for _all_candidate_metrics_satisfied helper."""

    def test_empty_candidates_returns_false(self):
        from datus.storage.metric.metric_init import _all_candidate_metrics_satisfied

        assert _all_candidate_metrics_satisfied({}, []) is False

    def test_candidate_not_in_existing_returns_false(self):
        from datus.storage.metric.metric_init import _all_candidate_metrics_satisfied

        plan = {"direct_metric_candidates": [{"name": "new_metric", "metric_type": "measure_proxy"}]}
        assert _all_candidate_metrics_satisfied(plan, []) is False

    def test_all_candidates_satisfied(self):
        from datus.storage.metric.metric_init import _all_candidate_metrics_satisfied

        plan = {
            "direct_metric_candidates": [
                {"name": "revenue_total", "metric_type": "measure_proxy", "base_measures": ["revenue"]}
            ]
        }
        existing = [{"name": "revenue_total", "type": "measure_proxy", "base_measures": ["revenue"]}]
        assert _all_candidate_metrics_satisfied(plan, existing) is True

    def test_one_unsatisfied_returns_false(self):
        from datus.storage.metric.metric_init import _all_candidate_metrics_satisfied

        plan = {
            "direct_metric_candidates": [
                {"name": "revenue_total", "metric_type": "measure_proxy", "base_measures": ["revenue"]},
                {"name": "new_metric", "metric_type": "measure_proxy", "base_measures": ["something"]},
            ]
        }
        existing = [{"name": "revenue_total", "type": "measure_proxy", "base_measures": ["revenue"]}]
        assert _all_candidate_metrics_satisfied(plan, existing) is False
