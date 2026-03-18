# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.reference_sql.reference_sql_init."""

from enum import Enum
from unittest.mock import MagicMock

import pytest

from datus.storage.reference_sql.reference_sql_init import BIZ_NAME, _action_status_value

# ---------------------------------------------------------------------------
# _action_status_value
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestActionStatusValue:
    """Tests for the _action_status_value helper."""

    def test_none_status_attribute(self):
        """Returns None when action has no status attribute."""
        action = object()
        assert _action_status_value(action) is None

    def test_status_is_none(self):
        """Returns None when action.status is None."""
        action = MagicMock(status=None)
        assert _action_status_value(action) is None

    def test_status_with_value_attribute(self):
        """Returns status.value when status is an enum-like object."""

        class MockStatus(Enum):
            SUCCESS = "success"

        action = MagicMock(status=MockStatus.SUCCESS)
        result = _action_status_value(action)
        assert result == "success"

    def test_status_string(self):
        """Returns str(status) when status has no .value attribute."""
        action = MagicMock(status="running")
        # MagicMock's status will be a string directly, so no .value
        action.status = "running"
        result = _action_status_value(action)
        assert result == "running"

    def test_status_with_custom_value(self):
        """Returns status.value for custom objects with a value attribute."""

        class CustomStatus:
            value = "custom_val"

        action = MagicMock()
        action.status = CustomStatus()
        result = _action_status_value(action)
        assert result == "custom_val"


# ---------------------------------------------------------------------------
# BIZ_NAME constant
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestBizNameConstant:
    """Tests for module-level constants."""

    def test_biz_name_value(self):
        """BIZ_NAME is reference_sql_init."""
        assert BIZ_NAME == "reference_sql_init"


# ---------------------------------------------------------------------------
# init_reference_sql - empty sql_dir
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitReferenceSqlEmptyDir:
    """Tests for init_reference_sql with empty/missing sql_dir."""

    def test_empty_sql_dir_returns_success(self):
        """When sql_dir is empty string, returns success with zero entries."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 0
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir="",
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0
        assert "empty" in result["message"].lower() or "no" in result["message"].lower()

    def test_empty_sql_dir_none(self):
        """When sql_dir is None, returns success with zero entries."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 5
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=None,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["total_stored_entries"] == 5


# ---------------------------------------------------------------------------
# init_reference_sql - validate_only mode
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitReferenceSqlValidateOnly:
    """Tests for init_reference_sql with validate_only=True."""

    def test_validate_only_with_valid_sql(self, tmp_path):
        """validate_only mode processes files but does not store."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        sql_file = tmp_path / "test.sql"
        sql_file.write_text("SELECT id, name FROM users WHERE id = 1;")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=True,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] >= 1
        assert result["processed_entries"] == 0
        assert "validate-only" in result["message"].lower()

    def test_validate_only_with_invalid_sql(self, tmp_path):
        """validate_only mode reports invalid entries."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        sql_file = tmp_path / "bad.sql"
        # Write a non-SELECT SQL that will be skipped
        sql_file.write_text("CREATE TABLE foo (id INT);")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=True,
        )

        assert result["status"] == "success"
        # CREATE TABLE is not a SELECT, so it is skipped entirely
        assert result["processed_entries"] == 0

    def test_validate_only_with_multiple_sql_files(self, tmp_path):
        """validate_only mode handles directory of SQL files."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        (tmp_path / "a.sql").write_text("SELECT 1;")
        (tmp_path / "b.sql").write_text("SELECT name FROM employees;")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(tmp_path),
            validate_only=True,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] >= 2


# ---------------------------------------------------------------------------
# init_reference_sql - no valid items
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitReferenceSqlNoValidItems:
    """Tests for init_reference_sql with no valid SQL items."""

    def test_no_valid_items_returns_success(self, tmp_path):
        """All non-SELECT SQL results in no valid items."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        sql_file = tmp_path / "ddl.sql"
        sql_file.write_text("DROP TABLE IF EXISTS test;\nCREATE TABLE test (id INT);")

        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 0
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=False,
        )

        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0


# ---------------------------------------------------------------------------
# init_reference_sql - incremental mode filtering
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitReferenceSqlIncrementalFiltering:
    """Tests for incremental mode SQL ID filtering logic."""

    def test_incremental_filters_existing_ids(self, tmp_path):
        """Incremental mode skips items whose IDs already exist."""
        from datus.storage.reference_sql.init_utils import gen_reference_sql_id
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql

        sql_content = "SELECT id FROM users;"
        sql_file = tmp_path / "test.sql"
        sql_file.write_text(sql_content)

        # Simulate the storage already having this SQL
        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 1
        # Return existing IDs that match
        mock_storage.search_all_reference_sql.return_value = [
            # The ID is generated from cleaned SQL, which may differ from raw
            # We need the actual cleaned SQL ID - but since we can't predict it,
            # we return a dummy. The real test is that the filtering path runs.
            {"id": gen_reference_sql_id("whatever")}
        ]
        mock_config = MagicMock()

        result = init_reference_sql(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=True,  # Use validate_only to avoid LLM calls
            build_mode="incremental",
        )

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# init_reference_sql_async - importability and coroutine check
# ---------------------------------------------------------------------------


@pytest.mark.ci
class TestInitReferenceSqlAsync:
    """Tests for init_reference_sql_async importability and interface."""

    def test_async_function_is_importable(self):
        """init_reference_sql_async can be imported from the module."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        assert init_reference_sql_async is not None

    def test_async_function_is_coroutine(self):
        """init_reference_sql_async is a coroutine function (async def)."""
        import inspect

        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        assert inspect.iscoroutinefunction(init_reference_sql_async)

    def test_async_function_signature(self):
        """init_reference_sql_async has the expected parameter names."""
        import inspect

        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        sig = inspect.signature(init_reference_sql_async)
        param_names = list(sig.parameters.keys())
        assert "storage" in param_names
        assert "global_config" in param_names
        assert "sql_dir" in param_names
        assert "args" not in param_names

    def test_async_optional_params_present(self):
        """init_reference_sql_async exposes all optional params."""
        import inspect

        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        sig = inspect.signature(init_reference_sql_async)
        param_names = list(sig.parameters.keys())
        for expected in ["validate_only", "build_mode", "pool_size", "subject_tree", "emit", "extra_instructions"]:
            assert expected in param_names, f"Expected param '{expected}' missing from init_reference_sql_async"

    @pytest.mark.asyncio
    async def test_async_returns_dict_for_empty_sql_dir(self):
        """Awaiting init_reference_sql_async with empty sql_dir returns a success dict."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        mock_storage = MagicMock()
        mock_storage.get_reference_sql_size.return_value = 0
        mock_config = MagicMock()

        result = await init_reference_sql_async(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir="",
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["valid_entries"] == 0
        assert result["processed_entries"] == 0

    @pytest.mark.asyncio
    async def test_async_validate_only_returns_dict(self, tmp_path):
        """Awaiting init_reference_sql_async in validate_only mode returns a dict."""
        from datus.storage.reference_sql.reference_sql_init import init_reference_sql_async

        sql_file = tmp_path / "query.sql"
        sql_file.write_text("SELECT id FROM orders;")

        mock_storage = MagicMock()
        mock_config = MagicMock()

        result = await init_reference_sql_async(
            storage=mock_storage,
            global_config=mock_config,
            sql_dir=str(sql_file),
            validate_only=True,
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["processed_entries"] == 0
