# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/service_manager.py"""

from unittest.mock import MagicMock, patch

from datus.utils.exceptions import DatusException, ErrorCode


def _make_agent_config(databases=None):
    """Build a minimal mock AgentConfig with service.databases."""
    db_map = databases if databases is not None else {}
    service = MagicMock()
    service.databases = db_map
    service.default_database = next(iter(db_map), None)
    service.bi_tools = {}
    service.schedulers = {}
    agent_config = MagicMock()
    agent_config.service = service
    return agent_config


def _make_db_config(db_type="sqlite", uri="path/to/db.sqlite", default=False):
    cfg = MagicMock()
    cfg.type = db_type
    cfg.uri = uri
    cfg.host = ""
    cfg.account = ""
    cfg.port = ""
    cfg.default = default
    cfg.logic_name = ""
    return cfg


class TestServiceManagerInit:
    """Tests for ServiceManager.__init__."""

    def test_init_with_valid_config_sets_agent_config(self, tmp_path):
        """When load_agent_config succeeds, agent_config is populated."""
        mock_config = _make_agent_config()
        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager(str(tmp_path / "agent.yml"))
            assert sm.agent_config is mock_config

    def test_init_with_missing_file_prints_error(self, tmp_path):
        """When load_agent_config raises FILE_NOT_FOUND, agent_config is None."""
        exc = DatusException(ErrorCode.COMMON_FILE_NOT_FOUND, message_args={"config_name": "agent", "file_name": "x"})
        with (
            patch("datus.cli.service_manager.load_agent_config", side_effect=exc),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager(str(tmp_path / "missing.yml"))
            assert sm.agent_config is None
            mock_console.print.assert_called()

    def test_init_with_generic_exception_sets_agent_config_none(self):
        """When load_agent_config raises generic exception, agent_config is None."""
        with (
            patch("datus.cli.service_manager.load_agent_config", side_effect=RuntimeError("unexpected")),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("any_path.yml")
            assert sm.agent_config is None
            mock_console.print.assert_called()


class TestServiceManagerList:
    """Tests for ServiceManager.list()."""

    def test_list_with_databases_shows_table(self):
        """list() prints a table when databases are configured."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"my_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0
            mock_console.print.assert_called()

    def test_list_with_empty_databases_shows_message(self):
        """list() prints a 'no databases' message when none are configured."""
        mock_config = _make_agent_config({})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("No databases" in c for c in calls)

    def test_list_shows_bi_tools_when_present(self):
        """list() prints BI tools section when bi_tools are configured."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"my_db": db_cfg})
        mock_config.service.bi_tools = {"tableau": {"url": "http://tableau"}}

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0
            mock_console.print.assert_called()


class TestServiceManagerDelete:
    """Tests for ServiceManager.delete()."""

    def test_delete_nonexistent_db_name_returns_1(self):
        """delete() returns 1 when given database name doesn't exist."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"real_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.cli.service_manager.Prompt.ask", return_value="nonexistent_db"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.delete()
            assert ret == 1
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("does not exist" in c for c in calls)

    def test_delete_empty_databases_returns_1(self):
        """delete() returns 1 when there are no databases to delete."""
        mock_config = _make_agent_config({})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.delete()
            assert ret == 1
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("No databases" in c for c in calls)

    def test_delete_cancelled_by_user_returns_1(self):
        """delete() returns 1 when user declines the confirmation prompt."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"my_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
            patch("datus.cli.service_manager.Prompt.ask", return_value="my_db"),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.delete()
            assert ret == 1


class TestServiceManagerSaveConfiguration:
    """Tests for ServiceManager._save_configuration()."""

    def test_save_configuration_builds_correct_structure(self):
        """_save_configuration() calls configure_manager.update with correct service section."""
        db_cfg = _make_db_config(db_type="sqlite", uri="data/db.sqlite", default=True)
        mock_config = _make_agent_config({"main_db": db_cfg})

        mock_cm = MagicMock()
        mock_cm.data = {}

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.configuration_manager", return_value=mock_cm),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            result = sm._save_configuration()
            assert result is True
            mock_cm.update.assert_called_once()
            call_kwargs = mock_cm.update.call_args
            updates = call_kwargs[1]["updates"] if call_kwargs[1] else call_kwargs[0][0]
            assert "service" in updates
            assert "databases" in updates["service"]

    def test_save_configuration_returns_false_on_exception(self):
        """_save_configuration() returns False when configuration_manager raises."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"main_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.configuration_manager", side_effect=RuntimeError("write error")),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            result = sm._save_configuration()
            assert result is False


class TestValidateDbName:
    """Tests for _validate_db_name()."""

    def test_valid_name_returns_true(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("my_database")
        assert ok is True
        assert msg == ""

    def test_empty_name_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("   ")
        assert ok is False
        assert "empty" in msg

    def test_name_with_space_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("bad name")
        assert ok is False

    def test_name_with_slash_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("path/name")
        assert ok is False


class TestValidatePort:
    """Tests for _validate_port()."""

    def test_valid_port_returns_true(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("5432")
        assert ok is True

    def test_port_below_range_returns_false(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("0")
        assert ok is False

    def test_port_above_range_returns_false(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("99999")
        assert ok is False

    def test_non_numeric_port_returns_false(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("abc")
        assert ok is False
