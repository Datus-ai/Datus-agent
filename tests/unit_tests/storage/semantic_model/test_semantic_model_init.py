# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.semantic_model.semantic_model_init."""

from unittest.mock import MagicMock, patch

import pytest

from datus.storage.semantic_model.semantic_model_init import (
    init_semantic_yaml_semantic_model,
    process_semantic_yaml_file,
)

# ---------------------------------------------------------------------------
# init_semantic_yaml_semantic_model
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSemanticYamlSemanticModel:
    """Tests for init_semantic_yaml_semantic_model function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when YAML file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = init_semantic_yaml_semantic_model(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_existing_file_delegates_to_process(self, tmp_path):
        """When file exists, calls process_semantic_yaml_file with include_metrics=False."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: test\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.process_semantic_yaml_file",
            return_value=(True, ""),
        ) as mock_process:
            success, error = init_semantic_yaml_semantic_model(str(yaml_file), mock_config)

        assert success is True
        assert error == ""
        mock_process.assert_called_once_with(str(yaml_file), mock_config, include_metrics=False)


# ---------------------------------------------------------------------------
# process_semantic_yaml_file
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestProcessSemanticYamlFile:
    """Tests for process_semantic_yaml_file function."""

    def test_file_not_found(self, tmp_path):
        """Returns (False, error) when file does not exist."""
        nonexistent = str(tmp_path / "missing.yaml")
        mock_config = MagicMock()

        success, error = process_semantic_yaml_file(nonexistent, mock_config)

        assert success is False
        assert "not found" in error

    def test_sync_success(self, tmp_path):
        """Returns (True, '') when sync succeeds."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "synced"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is True
        assert error == ""

    def test_sync_failure(self, tmp_path):
        """Returns (False, error) when sync reports failure."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False, "error": "validation failed"},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "validation failed" in error

    def test_sync_exception(self, tmp_path):
        """Returns (False, error) when sync raises an exception."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            side_effect=RuntimeError("connection error"),
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "connection error" in error

    def test_default_includes_both(self, tmp_path):
        """By default, include_semantic_objects and include_metrics are both True."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=True,
        )

    def test_exclude_metrics(self, tmp_path):
        """include_metrics=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_metrics=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=True,
            include_metrics=False,
        )

    def test_exclude_semantic_objects(self, tmp_path):
        """include_semantic_objects=False is forwarded to _sync_semantic_to_db."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": True, "message": "ok"},
        ) as mock_sync:
            process_semantic_yaml_file(str(yaml_file), mock_config, include_semantic_objects=False)

        mock_sync.assert_called_once_with(
            str(yaml_file),
            mock_config,
            include_semantic_objects=False,
            include_metrics=True,
        )

    def test_sync_unknown_error(self, tmp_path):
        """When sync returns failure with no error key, uses 'Unknown error'."""
        yaml_file = tmp_path / "model.yaml"
        yaml_file.write_text("tables:\n  - name: orders\n")
        mock_config = MagicMock()

        with patch(
            "datus.storage.semantic_model.semantic_model_init.GenerationHooks._sync_semantic_to_db",
            return_value={"success": False},
        ):
            success, error = process_semantic_yaml_file(str(yaml_file), mock_config)

        assert success is False
        assert "Unknown error" in error


# ---------------------------------------------------------------------------
# init_success_story_semantic_model_async - importability and coroutine check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelAsync:
    """Tests for init_success_story_semantic_model_async importability and interface."""

    def test_async_function_is_importable(self):
        """init_success_story_semantic_model_async can be imported from the module."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        assert init_success_story_semantic_model_async is not None

    def test_async_function_is_coroutine(self):
        """init_success_story_semantic_model_async is a coroutine function (async def)."""
        import inspect

        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        assert inspect.iscoroutinefunction(init_success_story_semantic_model_async)

    @pytest.mark.asyncio
    async def test_async_returns_false_for_missing_csv(self, tmp_path):
        """Awaiting init_success_story_semantic_model_async with a missing CSV returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        missing = str(tmp_path / "no_such_file.csv")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, missing)

        assert success is False
        assert "not found" in error.lower() or missing in error

    @pytest.mark.asyncio
    async def test_async_returns_false_for_empty_csv(self, tmp_path):
        """Awaiting with an empty CSV (no rows) returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_async_returns_false_for_missing_columns(self, tmp_path):
        """Awaiting with a CSV missing required columns returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "bad_cols.csv"
        csv_path.write_text("question\nWhat is revenue?\n")
        mock_config = MagicMock()

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "missing" in error.lower() or "sql" in error.lower()


# ---------------------------------------------------------------------------
# init_success_story_semantic_model sync wrapper - new signature
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelSync:
    """Tests for init_success_story_semantic_model sync wrapper with decoupled signature."""

    def test_sync_function_is_importable(self):
        """init_success_story_semantic_model can be imported."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        assert init_success_story_semantic_model is not None

    def test_sync_function_is_not_coroutine(self):
        """init_success_story_semantic_model is a plain sync function, not a coroutine."""
        import inspect

        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        assert not inspect.iscoroutinefunction(init_success_story_semantic_model)

    def test_sync_returns_tuple_for_missing_csv(self, tmp_path):
        """Sync wrapper returns (bool, str) tuple for a missing CSV path."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        missing = str(tmp_path / "no_file.csv")
        mock_config = MagicMock()

        result = init_success_story_semantic_model(mock_config, missing)

        assert isinstance(result, tuple)
        assert len(result) == 2
        success, error = result
        assert success is False
        assert isinstance(error, str)

    def test_sync_accepts_agent_config_and_success_story_args(self, tmp_path):
        """Sync wrapper accepts (agent_config, success_story) without argparse.Namespace."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()

        # Should call without raising TypeError about unexpected args
        result = init_success_story_semantic_model(mock_config, str(csv_path))
        assert isinstance(result, tuple)

    def test_sync_accepts_optional_emit_kwarg(self, tmp_path):
        """Sync wrapper accepts optional emit keyword argument."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\n")
        mock_config = MagicMock()
        emit_calls = []

        result = init_success_story_semantic_model(mock_config, str(csv_path), emit=emit_calls.append)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# init_success_story_semantic_model_async - LLM execution path (lines 95-157)
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitSuccessStorySemanticModelAsyncLLMPath:
    """Tests for the async LLM execution path inside init_success_story_semantic_model_async."""

    @pytest.mark.asyncio
    async def test_success_path_with_semantic_models_list(self, tmp_path, monkeypatch):
        """Success path: agentic node yields action with semantic_models list → returns (True, '')."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,What is one?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = "cat"
        mock_db_config.database = "db"
        mock_db_config.schema = "public"
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output={"semantic_models": ["model1.yaml", "model2.yaml"]},
                    messages="Generated models",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_success_path_emits_events(self, tmp_path, monkeypatch):
        """Success path: emit callback is called for ITEM_PROCESSING and TASK_COMPLETED stages."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,What is one?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output={"semantic_models": ["model.yaml"]},
                    messages="Generated model",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is True
        assert BatchStage.TASK_STARTED in emitted_stages
        assert BatchStage.TASK_COMPLETED in emitted_stages

    @pytest.mark.asyncio
    async def test_success_path_single_model_string(self, tmp_path, monkeypatch):
        """Success path: semantic_models as a single string (not list) is also collected."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                # single string instead of list
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output={"semantic_models": "single_model.yaml"},
                    messages="Generated",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_empty_result_path_returns_false(self, tmp_path, monkeypatch):
        """Empty result path: no generated files → returns (False, error)."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                # Yields an action with SUCCESS but no semantic_models key
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output={"other_key": "value"},
                    messages="Nothing useful",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert error != ""

    @pytest.mark.asyncio
    async def test_empty_result_emits_task_failed(self, tmp_path, monkeypatch):
        """Empty result path: emit callback receives TASK_FAILED event."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output={},
                    messages="",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is False
        assert BatchStage.TASK_FAILED in emitted_stages

    @pytest.mark.asyncio
    async def test_exception_path_returns_false(self, tmp_path, monkeypatch):
        """Exception path: execute_stream raises → returns (False, error)."""
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                raise RuntimeError("LLM backend error")
                yield  # make it an async generator

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        assert success is False
        assert "LLM backend error" in error

    @pytest.mark.asyncio
    async def test_exception_emits_task_failed(self, tmp_path, monkeypatch):
        """Exception path: emit receives TASK_FAILED when execute_stream raises."""
        from datus.schemas.batch_events import BatchStage
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                raise ValueError("unexpected error")
                yield  # async generator marker

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        emitted_stages = []

        def capture_emit(event):
            emitted_stages.append(event.stage)

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path), emit=capture_emit)

        assert success is False
        assert BatchStage.TASK_FAILED in emitted_stages

    @pytest.mark.asyncio
    async def test_action_with_none_output_skipped(self, tmp_path, monkeypatch):
        """Action with output=None should not cause error and counts as empty result."""
        from types import SimpleNamespace

        from datus.schemas.action_history import ActionStatus
        from datus.storage.semantic_model.semantic_model_init import init_success_story_semantic_model_async

        csv_path = tmp_path / "story.csv"
        csv_path.write_text("sql,question\nSELECT 1,Q?\n")

        mock_config = MagicMock()
        mock_db_config = MagicMock()
        mock_db_config.catalog = ""
        mock_db_config.database = "db"
        mock_db_config.schema = ""
        mock_config.current_db_config.return_value = mock_db_config

        class MockSemanticNode:
            def __init__(self, *args, **kwargs):
                self.input = None

            async def execute_stream(self, action_history_manager):
                action = SimpleNamespace(
                    status=ActionStatus.SUCCESS,
                    output=None,
                    messages="",
                )
                yield action

        monkeypatch.setattr(
            "datus.storage.semantic_model.semantic_model_init.GenSemanticModelAgenticNode",
            MockSemanticNode,
        )

        success, error = await init_success_story_semantic_model_async(mock_config, str(csv_path))

        # No files generated → failure
        assert success is False
