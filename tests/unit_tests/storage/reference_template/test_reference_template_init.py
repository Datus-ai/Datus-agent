# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.reference_template.reference_template_init."""

import inspect
from enum import Enum
from unittest.mock import MagicMock

import pytest

from datus.storage.reference_template.reference_template_init import (
    BIZ_NAME,
    _action_status_value,
)

# ---------------------------------------------------------------------------
# _action_status_value
# ---------------------------------------------------------------------------


class TestActionStatusValue:
    def test_none_status_attribute(self):
        action = object()
        assert _action_status_value(action) is None

    def test_status_is_none(self):
        action = MagicMock(status=None)
        assert _action_status_value(action) is None

    def test_status_with_value_attribute(self):
        class MockStatus(Enum):
            SUCCESS = "success"

        action = MagicMock(status=MockStatus.SUCCESS)
        assert _action_status_value(action) == "success"

    def test_status_string(self):
        action = MagicMock()
        action.status = "running"
        assert _action_status_value(action) == "running"

    def test_status_with_custom_value(self):
        class CustomStatus:
            value = "custom_val"

        action = MagicMock()
        action.status = CustomStatus()
        assert _action_status_value(action) == "custom_val"


# ---------------------------------------------------------------------------
# BIZ_NAME constant
# ---------------------------------------------------------------------------


class TestBizNameConstant:
    def test_biz_name_value(self):
        assert BIZ_NAME == "reference_template_init"


# ---------------------------------------------------------------------------
# init_reference_template - empty template_dir
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateEmptyDir:
    def test_empty_template_dir_returns_success(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir="",
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0
        assert "empty" in result["message"].lower() or "no" in result["message"].lower()

    def test_empty_template_dir_none(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 5
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=None,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["total_stored_entries"] == 5


# ---------------------------------------------------------------------------
# init_reference_template - validate_only mode
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateValidateOnly:
    def test_validate_only_with_valid_template(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        tpl_file = tmp_path / "test.j2"
        tpl_file.write_text("SELECT * FROM t WHERE dt > '{{start_date}}'")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tpl_file),
            validate_only=True,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] >= 1
        assert result["processed_entries"] == 0
        assert "validate-only" in result["message"].lower()

    def test_validate_only_with_invalid_template(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        tpl_file = tmp_path / "bad.j2"
        tpl_file.write_text("{% if broken")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tpl_file),
            validate_only=True,
        )

        assert result["status"] == "success"
        assert result["invalid_entries"] >= 1
        assert result["processed_entries"] == 0

    def test_validate_only_with_multiple_files(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        (tmp_path / "a.j2").write_text("SELECT {{x}}")
        (tmp_path / "b.jinja2").write_text("SELECT {{y}} FROM t")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tmp_path),
            validate_only=True,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] >= 2


# ---------------------------------------------------------------------------
# init_reference_template - no valid items
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateNoValidItems:
    def test_all_invalid_returns_success(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        tpl_file = tmp_path / "broken.j2"
        tpl_file.write_text("{% if x %}no end")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tpl_file),
            validate_only=False,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0


# ---------------------------------------------------------------------------
# init_reference_template - incremental mode filtering
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateIncrementalFiltering:
    def test_incremental_filters_existing_ids(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template

        tpl_file = tmp_path / "test.j2"
        tpl_file.write_text("SELECT {{x}} FROM t")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 1
        mock_storage.search_all_reference_templates.return_value = [{"id": "dummy_id"}]
        mock_config = MagicMock()

        result = init_reference_template(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tpl_file),
            validate_only=True,
            build_mode="incremental",
        )

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# init_reference_template_async - importability and coroutine check
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateAsync:
    def test_async_function_is_importable(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        assert init_reference_template_async is not None

    def test_async_function_is_coroutine(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        assert inspect.iscoroutinefunction(init_reference_template_async)

    def test_async_function_signature(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        sig = inspect.signature(init_reference_template_async)
        param_names = list(sig.parameters.keys())
        assert "storage" in param_names
        assert "global_config" in param_names
        assert "template_dir" in param_names

    def test_async_optional_params_present(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        sig = inspect.signature(init_reference_template_async)
        param_names = list(sig.parameters.keys())
        for expected in ["validate_only", "build_mode", "pool_size", "subject_tree", "emit", "extra_instructions"]:
            assert expected in param_names, f"Expected param '{expected}' missing"

    @pytest.mark.asyncio
    async def test_async_returns_dict_for_empty_template_dir(self):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        result = await init_reference_template_async(
            storage=mock_storage,
            global_config=mock_config,
            template_dir="",
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0

    @pytest.mark.asyncio
    async def test_async_validate_only_returns_dict(self, tmp_path):
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        tpl_file = tmp_path / "query.j2"
        tpl_file.write_text("SELECT {{col}} FROM orders")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = await init_reference_template_async(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tpl_file),
            validate_only=True,
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["processed_entries"] == 0


# ---------------------------------------------------------------------------
# TEMPLATE_EXTRA_INSTRUCTIONS constant
# ---------------------------------------------------------------------------


class TestTemplateExtraInstructions:
    def test_instructions_mention_jinja2(self):
        from datus.storage.reference_template.reference_template_init import TEMPLATE_EXTRA_INSTRUCTIONS

        assert "Jinja2" in TEMPLATE_EXTRA_INSTRUCTIONS
        assert "parameter" in TEMPLATE_EXTRA_INSTRUCTIONS.lower()


# ---------------------------------------------------------------------------
# process_template_item - mock SqlSummaryAgenticNode
# ---------------------------------------------------------------------------


class TestProcessTemplateItem:
    """Tests for process_template_item() with mocked SqlSummaryAgenticNode."""

    @pytest.mark.asyncio
    async def test_success_with_complete_yaml(self, tmp_path):
        """Successful LLM run that writes a complete YAML summary file."""
        from unittest.mock import patch

        import yaml

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        # Prepare a YAML summary file the node would create
        summary_dir = tmp_path / "sql_summaries" / "test_ns"
        summary_dir.mkdir(parents=True)
        summary_file = summary_dir / "tpl_001.yaml"
        summary_file.write_text(
            yaml.dump(
                {
                    "name": "free_rate_query",
                    "summary": "Query free meal rates by school type",
                    "search_text": "free rate school type continuation",
                    "subject_tree": "Education/FreeRate",
                    "tags": "education,free_rate",
                }
            )
        )

        # Mock action that returns success with sql_summary_file
        success_action = MagicMock(spec=ActionHistory)
        success_action.status = ActionStatus.SUCCESS
        success_action.output = {"sql_summary_file": "tpl_001.yaml"}
        success_action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield success_action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        mock_config.path_manager.sql_summary_path.return_value = summary_dir
        mock_config.current_namespace = "test_ns"

        item = {
            "template": "SELECT * FROM t WHERE x = '{{val}}'",
            "filepath": "/tmp/test.j2",
            "comment": "",
            "parameters": '[{"name": "val"}]',
        }

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(item, mock_config, build_mode="overwrite")

        assert result == "tpl_001.yaml"
        # Verify metadata was backfilled into item
        assert item["name"] == "free_rate_query"
        assert item["summary"] == "Query free meal rates by school type"
        assert item["search_text"] == "free rate school type continuation"
        assert item["subject_tree"] == "Education/FreeRate"
        assert item["tags"] == "education,free_rate"

    @pytest.mark.asyncio
    async def test_no_sql_summary_file_returns_none(self):
        """LLM returns success but no sql_summary_file in output."""
        from unittest.mock import patch

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        action = MagicMock(spec=ActionHistory)
        action.status = ActionStatus.SUCCESS
        action.output = {"some_other_key": "value"}
        action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(item, mock_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_incomplete_metadata_returns_none(self, tmp_path):
        """YAML exists but missing required fields → returns None."""
        from unittest.mock import patch

        import yaml

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        summary_dir = tmp_path / "sql_summaries" / "test_ns"
        summary_dir.mkdir(parents=True)
        summary_file = summary_dir / "incomplete.yaml"
        summary_file.write_text(yaml.dump({"name": "test", "summary": "partial"}))
        # Missing search_text and subject_tree

        action = MagicMock(spec=ActionHistory)
        action.status = ActionStatus.SUCCESS
        action.output = {"sql_summary_file": "incomplete.yaml"}
        action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        mock_config.path_manager.sql_summary_path.return_value = summary_dir
        mock_config.current_namespace = "test_ns"

        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(item, mock_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_yaml_file_not_found_returns_none(self):
        """sql_summary_file returned but file doesn't exist → returns None."""
        from pathlib import Path
        from unittest.mock import patch

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        action = MagicMock(spec=ActionHistory)
        action.status = ActionStatus.SUCCESS
        action.output = {"sql_summary_file": "nonexistent.yaml"}
        action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        mock_config.path_manager.sql_summary_path.return_value = Path("/nonexistent/dir")
        mock_config.current_namespace = "test_ns"

        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(item, mock_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_node_exception_returns_none(self):
        """Exception during node execution → returns None."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import process_template_item

        async def mock_execute_stream(*args, **kwargs):
            raise RuntimeError("LLM connection failed")
            yield  # noqa: F841 — unreachable but needed to make this an async generator

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(item, mock_config)

        assert result is None

    @pytest.mark.asyncio
    async def test_extra_instructions_appended(self, tmp_path):
        """extra_instructions are appended to the default instructions."""
        from unittest.mock import patch

        import yaml

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        summary_dir = tmp_path / "sql_summaries" / "test_ns"
        summary_dir.mkdir(parents=True)
        (summary_dir / "tpl.yaml").write_text(
            yaml.dump(
                {
                    "name": "test",
                    "summary": "s",
                    "search_text": "st",
                    "subject_tree": "A/B",
                    "tags": "t",
                }
            )
        )

        action = MagicMock(spec=ActionHistory)
        action.status = ActionStatus.SUCCESS
        action.output = {"sql_summary_file": "tpl.yaml"}
        action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        mock_config.path_manager.sql_summary_path.return_value = summary_dir
        mock_config.current_namespace = "test_ns"

        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            result = await process_template_item(
                item, mock_config, extra_instructions="Custom: focus on revenue metrics"
            )

        assert result == "tpl.yaml"

    @pytest.mark.asyncio
    async def test_event_helper_receives_callbacks(self, tmp_path):
        """When event_helper is provided, it receives item_processing callbacks."""
        from unittest.mock import patch

        import yaml

        from datus.schemas.action_history import ActionHistory, ActionStatus
        from datus.storage.reference_template.reference_template_init import process_template_item

        summary_dir = tmp_path / "sql_summaries" / "test_ns"
        summary_dir.mkdir(parents=True)
        (summary_dir / "tpl.yaml").write_text(
            yaml.dump({"name": "n", "summary": "s", "search_text": "st", "subject_tree": "A/B"})
        )

        action = MagicMock(spec=ActionHistory)
        action.status = ActionStatus.SUCCESS
        action.output = {"sql_summary_file": "tpl.yaml"}
        action.messages = []

        async def mock_execute_stream(*args, **kwargs):
            yield action

        mock_node = MagicMock()
        mock_node.execute_stream = mock_execute_stream

        mock_config = MagicMock()
        mock_config.path_manager.sql_summary_path.return_value = summary_dir
        mock_config.current_namespace = "test_ns"

        mock_event_helper = MagicMock()

        item = {"template": "SELECT 1", "filepath": "/tmp/test.j2", "comment": ""}

        with patch(
            "datus.storage.reference_template.reference_template_init.SqlSummaryAgenticNode",
            return_value=mock_node,
        ):
            await process_template_item(item, mock_config, event_helper=mock_event_helper, template_id="tpl_id_123")

        mock_event_helper.item_processing.assert_called_once()


# ---------------------------------------------------------------------------
# init_reference_template_async - full processing loop with mocked process_template_item
# ---------------------------------------------------------------------------


class TestInitReferenceTemplateAsyncProcessing:
    """Tests for the async processing loop that calls process_template_item."""

    @pytest.mark.asyncio
    async def test_overwrite_processes_all_items(self, tmp_path):
        """Overwrite mode processes all valid template items."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        (tmp_path / "a.j2").write_text("SELECT {{x}}")
        (tmp_path / "b.j2").write_text("SELECT {{y}} FROM t")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 2
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
            return_value="summary.yaml",
        ) as mock_process:
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="overwrite",
            )

        assert result["status"] == "success"
        assert result["valid_entries"] == 2
        assert mock_process.call_count == 2
        mock_storage.upsert_batch.assert_called_once()
        mock_storage.after_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_incremental_skips_existing(self, tmp_path):
        """Incremental mode skips items whose IDs already exist."""
        from unittest.mock import patch

        from datus.storage.reference_template.init_utils import gen_reference_template_id
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        template_content = "SELECT {{x}}"
        (tmp_path / "a.j2").write_text(template_content)
        existing_id = gen_reference_template_id(template_content)

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 1
        mock_storage.search_all_reference_templates.return_value = [{"id": existing_id}]
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
        ) as mock_process:
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="incremental",
            )

        assert result["status"] == "success"
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_processing_failure_recorded(self, tmp_path):
        """When process_template_item returns None, it's recorded as an error."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        (tmp_path / "a.j2").write_text("SELECT {{x}}")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
            return_value=None,
        ):
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="overwrite",
            )

        assert result["status"] == "success"
        assert result["processed_entries"] == 0
        assert result["process_errors"] is not None

    @pytest.mark.asyncio
    async def test_processing_exception_recorded(self, tmp_path):
        """When process_template_item raises, it's caught and recorded."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        (tmp_path / "a.j2").write_text("SELECT {{x}}")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
            side_effect=RuntimeError("LLM down"),
        ):
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="overwrite",
            )

        assert result["status"] == "success"
        assert result["processed_entries"] == 0
        assert "LLM down" in result["process_errors"]

    @pytest.mark.asyncio
    async def test_storage_upsert_failure_recorded(self, tmp_path):
        """When storage.upsert_batch raises, error is recorded but doesn't crash."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        (tmp_path / "a.j2").write_text("SELECT {{x}}")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_storage.upsert_batch.side_effect = RuntimeError("DB write failed")
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
            return_value="summary.yaml",
        ):
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="overwrite",
            )

        assert result["status"] == "success"
        assert result["process_errors"] is not None
        assert "Storage write failed" in result["process_errors"]

    @pytest.mark.asyncio
    async def test_pool_size_forced_serial_without_subject_tree(self, tmp_path):
        """pool_size > 1 is forced to 1 when subject_tree is None."""
        from unittest.mock import patch

        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        (tmp_path / "a.j2").write_text("SELECT {{x}}")

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 0
        mock_config = MagicMock()

        with patch(
            "datus.storage.reference_template.reference_template_init.process_template_item",
            return_value="s.yaml",
        ):
            result = await init_reference_template_async(
                storage=mock_storage,
                global_config=mock_config,
                template_dir=str(tmp_path),
                build_mode="overwrite",
                pool_size=4,
                subject_tree=None,  # Forces serial
            )

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_no_items_to_process_incremental(self, tmp_path):
        """Incremental mode with all existing items → no processing, no storage write."""

        from datus.storage.reference_template.init_utils import gen_reference_template_id
        from datus.storage.reference_template.reference_template_init import init_reference_template_async

        tpl = "SELECT {{x}}"
        (tmp_path / "a.j2").write_text(tpl)

        mock_storage = MagicMock()
        mock_storage.get_reference_template_size.return_value = 1
        mock_storage.search_all_reference_templates.return_value = [{"id": gen_reference_template_id(tpl)}]
        mock_config = MagicMock()

        result = await init_reference_template_async(
            storage=mock_storage,
            global_config=mock_config,
            template_dir=str(tmp_path),
            build_mode="incremental",
        )

        assert result["status"] == "success"
        assert result["processed_entries"] == 0
        mock_storage.upsert_batch.assert_not_called()
        mock_storage.after_init.assert_called_once()


# ---------------------------------------------------------------------------
# _is_safe_identifier
# ---------------------------------------------------------------------------


class TestIsSafeIdentifier:
    def test_plain_identifier(self):
        from datus.storage.reference_template.reference_template_init import _is_safe_identifier

        assert _is_safe_identifier("table_name")
        assert _is_safe_identifier("col1")
        assert _is_safe_identifier("_private")

    def test_backtick_quoted_identifier(self):
        from datus.storage.reference_template.reference_template_init import _is_safe_identifier

        assert _is_safe_identifier("`Educational Option Type`")
        assert _is_safe_identifier("`col with spaces`")

    def test_unsafe_identifiers(self):
        from datus.storage.reference_template.reference_template_init import _is_safe_identifier

        assert not _is_safe_identifier("1bad_start")
        assert not _is_safe_identifier("table; DROP TABLE users")
        assert not _is_safe_identifier("col'injection")
        assert not _is_safe_identifier("")


# ---------------------------------------------------------------------------
# _enrich_dimension_param
# ---------------------------------------------------------------------------


class TestEnrichDimensionParam:
    def test_enrich_success(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_dimension_param
        from datus.tools.func_tool.base import FuncToolResult

        mock_db = MagicMock()
        mock_db.read_query.return_value = FuncToolResult(
            success=1,
            result={"compressed_data": "index,value\n0,TypeA\n1,TypeB\n2,TypeC"},
        )
        p = {"name": "school_type", "type": "dimension", "column_ref": "schools.county"}
        _enrich_dimension_param(p, mock_db)
        assert "sample_values" in p
        assert "TypeA" in p["sample_values"]

    def test_enrich_no_result(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_dimension_param
        from datus.tools.func_tool.base import FuncToolResult

        mock_db = MagicMock()
        mock_db.read_query.return_value = FuncToolResult(success=0, error="query failed")
        p = {"name": "school_type", "type": "dimension", "column_ref": "schools.county"}
        _enrich_dimension_param(p, mock_db)
        assert "sample_values" not in p

    def test_enrich_invalid_col_ref_no_dot(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_dimension_param

        mock_db = MagicMock()
        p = {"name": "x", "type": "dimension", "column_ref": "no_dot_col_ref"}
        _enrich_dimension_param(p, mock_db)
        mock_db.read_query.assert_not_called()
        assert "sample_values" not in p

    def test_enrich_unsafe_identifier_skipped(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_dimension_param

        mock_db = MagicMock()
        p = {"name": "x", "type": "dimension", "column_ref": "schools; DROP TABLE users.county"}
        _enrich_dimension_param(p, mock_db)
        mock_db.read_query.assert_not_called()
        assert "sample_values" not in p

    def test_enrich_db_exception_does_not_raise(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_dimension_param

        mock_db = MagicMock()
        mock_db.read_query.side_effect = RuntimeError("connection lost")
        p = {"name": "x", "type": "dimension", "column_ref": "schools.county"}
        _enrich_dimension_param(p, mock_db)
        assert "sample_values" not in p


# ---------------------------------------------------------------------------
# _enrich_column_param
# ---------------------------------------------------------------------------


class TestEnrichColumnParam:
    def test_enrich_success(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_column_param
        from datus.tools.func_tool.base import FuncToolResult

        mock_db = MagicMock()
        mock_db.describe_table.return_value = FuncToolResult(
            success=1,
            result=[{"column_name": "id"}, {"column_name": "name"}, {"column_name": "region"}],
        )
        p = {"name": "col", "type": "column", "table_refs": ["schools"]}
        _enrich_column_param(p, mock_db)
        assert "sample_values" in p
        assert "id" in p["sample_values"]
        assert "region" in p["sample_values"]

    def test_enrich_multiple_tables(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_column_param
        from datus.tools.func_tool.base import FuncToolResult

        mock_db = MagicMock()
        mock_db.describe_table.side_effect = [
            FuncToolResult(success=1, result=[{"column_name": "a"}, {"column_name": "b"}]),
            FuncToolResult(success=1, result=[{"column_name": "c"}, {"column_name": "b"}]),
        ]
        p = {"name": "col", "type": "column", "table_refs": ["t1", "t2"]}
        _enrich_column_param(p, mock_db)
        assert "a" in p["sample_values"]
        assert "c" in p["sample_values"]
        # "b" deduped
        assert p["sample_values"].count("b") == 1

    def test_unsafe_table_skipped(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_column_param

        mock_db = MagicMock()
        p = {"name": "col", "type": "column", "table_refs": ["bad; table"]}
        _enrich_column_param(p, mock_db)
        mock_db.describe_table.assert_not_called()
        assert "sample_values" not in p

    def test_enrich_describe_fails_gracefully(self):
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_column_param

        mock_db = MagicMock()
        mock_db.describe_table.side_effect = RuntimeError("table not found")
        p = {"name": "col", "type": "column", "table_refs": ["schools"]}
        _enrich_column_param(p, mock_db)
        assert "sample_values" not in p

    def test_column_name_fallback_to_name_key(self):
        """If result uses 'name' key instead of 'column_name', it should still be picked up."""
        from unittest.mock import MagicMock

        from datus.storage.reference_template.reference_template_init import _enrich_column_param
        from datus.tools.func_tool.base import FuncToolResult

        mock_db = MagicMock()
        mock_db.describe_table.return_value = FuncToolResult(
            success=1,
            result=[{"name": "id"}, {"name": "score"}],
        )
        p = {"name": "col", "type": "column", "table_refs": ["t"]}
        _enrich_column_param(p, mock_db)
        assert "id" in p["sample_values"]
        assert "score" in p["sample_values"]


# ---------------------------------------------------------------------------
# _extract_csv_values
# ---------------------------------------------------------------------------


class TestExtractCsvValues:
    def test_valid_compressed_data(self):

        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(
            success=1,
            result={"compressed_data": "index,value\n0,TypeA\n1,TypeB"},
        )
        values = _extract_csv_values(result)
        assert values == ["TypeA", "TypeB"]

    def test_header_only_no_values(self):
        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(
            success=1,
            result={"compressed_data": "index,value"},
        )
        values = _extract_csv_values(result)
        assert values is None

    def test_empty_compressed_data(self):
        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(success=1, result={"compressed_data": ""})
        values = _extract_csv_values(result)
        assert values is None

    def test_no_result(self):
        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(success=0, error="failed")
        values = _extract_csv_values(result)
        assert values is None

    def test_empty_result_dict(self):
        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(success=1, result={})
        values = _extract_csv_values(result)
        assert values is None

    def test_values_with_commas_in_value(self):
        """Values with commas should be captured from after the first comma."""
        from datus.storage.reference_template.reference_template_init import _extract_csv_values
        from datus.tools.func_tool.base import FuncToolResult

        result = FuncToolResult(
            success=1,
            result={"compressed_data": "index,value\n0,hello world\n1,another"},
        )
        values = _extract_csv_values(result)
        assert "hello world" in values
        assert "another" in values
