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
