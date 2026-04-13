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


def _make_db_config(db_type="sqlite", uri="path/to/db.sqlite", default=False, host="", account="", port=""):
    cfg = MagicMock()
    cfg.type = db_type
    cfg.uri = uri
    cfg.host = host
    cfg.account = account
    cfg.port = port
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

    def test_init_with_other_datus_exception_prints_message(self):
        """When load_agent_config raises a non-FILE_NOT_FOUND DatusException, message is printed."""
        exc = DatusException(ErrorCode.COMMON_CONFIG_ERROR, message_args={})
        with (
            patch("datus.cli.service_manager.load_agent_config", side_effect=exc),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("any_path.yml")
            assert sm.agent_config is None
            mock_console.print.assert_called()


class TestServiceManagerRun:
    """Tests for ServiceManager.run()."""

    def _make_sm(self, mock_config):
        with patch("datus.cli.service_manager.load_agent_config", return_value=mock_config):
            from datus.cli.service_manager import ServiceManager

            return ServiceManager("agent.yml")

    def test_run_returns_1_when_no_config(self):
        """run() returns 1 when agent_config is None."""
        with (
            patch("datus.cli.service_manager.load_agent_config", side_effect=RuntimeError("fail")),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            assert sm.run("list") == 1

    def test_run_list_command_calls_list(self):
        """run('list') delegates to list() and returns its return value."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)
        with patch.object(sm, "list", return_value=0) as mock_list:
            assert sm.run("list") == 0
            mock_list.assert_called_once()

    def test_run_add_command_calls_add(self):
        """run('add') delegates to add() and returns its return value."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)
        with patch.object(sm, "add", return_value=0) as mock_add:
            assert sm.run("add") == 0
            mock_add.assert_called_once()

    def test_run_delete_command_calls_delete(self):
        """run('delete') delegates to delete() and returns its return value."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)
        with patch.object(sm, "delete", return_value=1) as mock_delete:
            assert sm.run("delete") == 1
            mock_delete.assert_called_once()

    def test_run_unknown_command_returns_1(self):
        """run() with unknown command prints error and returns 1."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)
        with patch("datus.cli.service_manager.console") as mock_console:
            result = sm.run("unknown_cmd")
        assert result == 1
        calls_str = [str(c) for c in mock_console.print.call_args_list]
        assert any("Unknown command" in c for c in calls_str)


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

    def test_list_shows_host_port_connection(self):
        """list() uses host:port as connection string when uri is empty but host is set."""
        db_cfg = _make_db_config(db_type="postgresql", uri="", host="localhost", port="5432")
        mock_config = _make_agent_config({"pg_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0

    def test_list_shows_account_connection(self):
        """list() uses account= as connection string when uri and host are both empty."""
        db_cfg = _make_db_config(db_type="snowflake", uri="", host="", account="myaccount.us-east-1")
        mock_config = _make_agent_config({"sf_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0

    def test_list_shows_schedulers_when_present(self):
        """list() prints schedulers section when schedulers are configured."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"my_db": db_cfg})
        mock_config.service.schedulers = {"airflow": {"url": "http://airflow"}}

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Schedulers" in c or "airflow" in c for c in calls)

    def test_list_marks_default_database(self):
        """list() marks the default database with '*' in the Default column."""
        db_cfg_default = _make_db_config(default=True)
        db_cfg_other = _make_db_config(default=False)
        mock_config = _make_agent_config({"default_db": db_cfg_default, "other_db": db_cfg_other})
        mock_config.service.default_database = "default_db"

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.list()
            assert ret == 0


class TestServiceManagerAdd:
    """Tests for ServiceManager.add()."""

    def _make_sm(self, mock_config):
        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            return ServiceManager("agent.yml")

    def test_add_invalid_db_name_returns_1(self):
        """add() returns 1 when given an invalid database name (contains space)."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        with (
            patch("datus.cli.service_manager.Prompt.ask", return_value="bad name"),
            patch("datus.cli.service_manager.console"),
        ):
            result = sm.add()
        assert result == 1

    def test_add_empty_name_returns_1(self):
        """add() returns 1 when given an empty database name."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        with (
            patch("datus.cli.service_manager.Prompt.ask", return_value="   "),
            patch("datus.cli.service_manager.console"),
        ):
            result = sm.add()
        assert result == 1

    def test_add_duplicate_name_returns_1(self):
        """add() returns 1 when the database name already exists."""
        existing_db = _make_db_config()
        mock_config = _make_agent_config({"existing_db": existing_db})
        sm = self._make_sm(mock_config)

        with (
            patch("datus.cli.service_manager.Prompt.ask", return_value="existing_db"),
            patch("datus.cli.service_manager.console") as mock_console,
        ):
            result = sm.add()
        assert result == 1
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("already exists" in c for c in calls)

    def test_add_no_adapters_returns_1(self):
        """add() returns 1 when no database adapters are available."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {}

        with (
            patch("datus.cli.service_manager.Prompt.ask", return_value="new_db"),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.tools.db_tools.connector_registry", mock_registry),
        ):
            result = sm.add()
        assert result == 1
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("No database adapters" in c for c in calls)

    def test_add_no_config_fields_returns_1(self):
        """add() returns 1 when adapter has no configuration schema fields."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {}
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        prompt_values = iter(["new_db", "duckdb"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.tools.db_tools.connector_registry", mock_registry),
        ):
            result = sm.add()
        assert result == 1
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("does not have a configuration schema" in c for c in calls)

    def test_add_connectivity_failure_returns_1(self):
        """add() returns 1 when database connectivity test fails."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "text"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        prompt_values = iter(["new_db", "duckdb", "/tmp/test.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(False, "connection refused")),
        ):
            result = sm.add()
        assert result == 1
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("connectivity test failed" in c for c in calls)

    def test_add_successful_save_returns_0(self):
        """add() returns 0 on successful connectivity test and config save."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "text"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "duckdb", "/tmp/test.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_save_failure_returns_1(self):
        """add() returns 1 when connectivity succeeds but config save fails."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "text"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "duckdb", "/tmp/test.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=False),
        ):
            result = sm.add()
        assert result == 1

    def test_add_first_db_auto_set_as_default(self):
        """add() automatically sets the first database as default without prompting."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "text"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        captured_config_data = {}

        def fake_filter_kwargs(cls, data):
            captured_config_data.update(data)
            return fake_db_config

        prompt_values = iter(["first_db", "duckdb", "/tmp/test.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", side_effect=fake_filter_kwargs),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0
        assert captured_config_data.get("default") is True

    def test_add_subsequent_db_prompts_for_default(self):
        """add() prompts the user about setting as default when other DBs exist."""
        existing_db = _make_db_config()
        mock_config = _make_agent_config({"existing_db": existing_db})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "text"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "duckdb", "/tmp/test.db"])
        confirm_values = iter([True])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", side_effect=lambda *a, **kw: next(confirm_values)),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_password_field_uses_getpass(self):
        """add() uses getpass() for password-type fields instead of Prompt.ask."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "password": {"required": False, "input_type": "password"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"postgresql": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "postgresql"])
        getpass_value = "secret"

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.getpass", return_value=getpass_value) as mock_getpass,
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        mock_getpass.assert_called_once()
        assert result == 0

    def test_add_file_path_field_with_sample_file(self):
        """add() uses sample directory path for file_path fields that have a default_sample."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": True, "input_type": "file_path", "default_sample": "sample.db"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        mock_path_manager = MagicMock()
        mock_path_manager.sample_dir = MagicMock()
        mock_path_manager.sample_dir.__truediv__ = MagicMock(return_value="/data/samples/sample.db")

        prompt_values = iter(["new_db", "duckdb", "/data/samples/sample.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.get_path_manager", return_value=mock_path_manager),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_file_path_field_without_sample_file(self):
        """add() uses default_value for file_path fields without default_sample."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "uri": {"required": False, "input_type": "file_path", "default": "/tmp/default.db"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"duckdb": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "duckdb", "/tmp/default.db"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_int_field_invalid_port_returns_1(self):
        """add() returns 1 when an invalid port number is entered."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "port": {"required": False, "type": "int", "default": 5432},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"postgresql": mock_adapter}

        prompt_values = iter(["new_db", "postgresql", "99999"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
        ):
            result = sm.add()
        assert result == 1

    def test_add_int_field_non_integer_returns_1(self):
        """add() returns 1 when a non-integer value is entered for an int field."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "workers": {"required": False, "type": "int", "default": 4},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"custom": mock_adapter}

        prompt_values = iter(["new_db", "custom", "not_a_number"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
        ):
            result = sm.add()
        assert result == 1

    def test_add_optional_field_with_default(self):
        """add() uses Prompt.ask with default for optional fields that have a default value."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "schema_name": {"required": False, "default": "public"},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"postgresql": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "postgresql", "public"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_optional_field_without_default(self):
        """add() uses Prompt.ask with empty default for optional fields with no default."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "extra_option": {"required": False},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"custom": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "custom", ""])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_required_field_no_default(self):
        """add() uses Prompt.ask without default for required fields."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "host": {"required": True},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"postgresql": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        prompt_values = iter(["new_db", "postgresql", "localhost"])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0

    def test_add_int_field_empty_value_uses_default(self):
        """add() uses default value when int field is left empty."""
        mock_config = _make_agent_config({})
        sm = self._make_sm(mock_config)

        mock_adapter = MagicMock()
        mock_adapter.get_config_fields.return_value = {
            "type": {"required": True},
            "workers": {"required": False, "type": "int", "default": 4},
        }
        mock_registry = MagicMock()
        mock_registry.list_available_adapters.return_value = {"custom": mock_adapter}

        from datus.configuration.agent_config import DbConfig

        fake_db_config = MagicMock(spec=DbConfig)
        fake_db_config.logic_name = ""
        fake_db_config.default = False

        # Return empty string to trigger the "else: value = default_value" path
        prompt_values = iter(["new_db", "custom", ""])

        with (
            patch("datus.cli.service_manager.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_values)),
            patch("datus.cli.service_manager.Confirm.ask", return_value=False),
            patch("datus.cli.service_manager.console"),
            patch("datus.tools.db_tools.connector_registry", mock_registry),
            patch("datus.cli.service_manager.detect_db_connectivity", return_value=(True, "")),
            patch("datus.cli.service_manager.DbConfig.filter_kwargs", return_value=fake_db_config),
            patch.object(sm, "_save_configuration", return_value=True),
        ):
            result = sm.add()
        assert result == 0


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

    def test_delete_empty_name_input_returns_1(self):
        """delete() returns 1 when user enters an empty database name."""
        db_cfg = _make_db_config()
        mock_config = _make_agent_config({"my_db": db_cfg})

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.cli.service_manager.Prompt.ask", return_value="   "),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.delete()
            assert ret == 1
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("cannot be empty" in c for c in calls)

    def test_delete_confirmed_saves_and_returns_0(self):
        """delete() removes the database, saves config, and returns 0 on success."""
        db_cfg = _make_db_config()
        databases = {"my_db": db_cfg}
        mock_config = _make_agent_config(databases)

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.cli.service_manager.Prompt.ask", return_value="my_db"),
            patch("datus.cli.service_manager.Confirm.ask", return_value=True),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")

            mock_cm = MagicMock()
            mock_cm.data = {}

            with patch("datus.cli.service_manager.configuration_manager", return_value=mock_cm):
                ret = sm.delete()
            assert ret == 0
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("deleted successfully" in c for c in calls)

    def test_delete_confirmed_save_failure_returns_1(self):
        """delete() returns 1 when save fails after confirmed deletion."""
        db_cfg = _make_db_config()
        databases = {"my_db": db_cfg}
        mock_config = _make_agent_config(databases)

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.console") as mock_console,
            patch("datus.cli.service_manager.Prompt.ask", return_value="my_db"),
            patch("datus.cli.service_manager.Confirm.ask", return_value=True),
            patch("datus.cli.service_manager.configuration_manager", side_effect=RuntimeError("disk error")),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            ret = sm.delete()
        assert ret == 1
        calls = [str(c) for c in mock_console.print.call_args_list]
        assert any("Failed to save configuration" in c for c in calls)


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

    def test_save_configuration_non_sqlite_uses_to_dict(self):
        """_save_configuration() uses to_dict() path for non-SQLite/DuckDB databases."""
        db_cfg = MagicMock()
        db_cfg.type = "postgresql"
        db_cfg.uri = ""
        db_cfg.host = "localhost"
        db_cfg.port = "5432"
        db_cfg.default = False
        db_cfg.logic_name = "pg_logical"
        db_cfg.to_dict.return_value = {
            "type": "postgresql",
            "host": "localhost",
            "port": "5432",
            "username": "admin",
            "password": "secret",
            "database": "mydb",
            "logic_name": "pg_logical",
            "path_pattern": "",
            "extra": None,
            "default": False,
        }

        mock_config = _make_agent_config({"pg_db": db_cfg})

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
        db_cfg.to_dict.assert_called_once()
        call_kwargs = mock_cm.update.call_args
        updates = call_kwargs[1]["updates"] if call_kwargs[1] else call_kwargs[0][0]
        pg_entry = updates["service"]["databases"]["pg_db"]
        # Internal fields should be removed
        assert "logic_name" not in pg_entry
        assert "path_pattern" not in pg_entry

    def test_save_configuration_removes_legacy_namespace_key(self):
        """_save_configuration() removes legacy 'namespace' key when present."""
        db_cfg = _make_db_config(db_type="sqlite", uri="data/db.sqlite")
        mock_config = _make_agent_config({"main_db": db_cfg})

        mock_cm = MagicMock()
        mock_cm.data = {"namespace": {"old_db": {}}, "service": {}}

        with (
            patch("datus.cli.service_manager.load_agent_config", return_value=mock_config),
            patch("datus.cli.service_manager.configuration_manager", return_value=mock_cm),
            patch("datus.cli.service_manager.console"),
        ):
            from datus.cli.service_manager import ServiceManager

            sm = ServiceManager("agent.yml")
            result = sm._save_configuration()

        assert result is True
        assert "namespace" not in mock_cm.data
        mock_cm.save.assert_called_once()

    def test_save_configuration_with_default_db_includes_default_flag(self):
        """_save_configuration() includes 'default: True' in output when db has default=True."""
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
        call_kwargs = mock_cm.update.call_args
        updates = call_kwargs[1]["updates"] if call_kwargs[1] else call_kwargs[0][0]
        main_entry = updates["service"]["databases"]["main_db"]
        assert main_entry.get("default") is True

    def test_save_configuration_non_default_sqlite_no_logic_name_diff(self):
        """_save_configuration() skips adding 'name' field when logic_name matches db_name."""
        db_cfg = MagicMock()
        db_cfg.type = "sqlite"
        db_cfg.uri = "data/db.sqlite"
        db_cfg.default = False
        db_cfg.logic_name = "main_db"  # Same as the dict key

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

    def test_save_configuration_sqlite_with_different_logic_name_adds_name_field(self):
        """_save_configuration() adds 'name' field when logic_name differs from db_name."""
        db_cfg = MagicMock()
        db_cfg.type = "sqlite"
        db_cfg.uri = "data/db.sqlite"
        db_cfg.default = False
        db_cfg.logic_name = "alias_name"  # Different from the dict key "main_db"

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
        call_kwargs = mock_cm.update.call_args
        updates = call_kwargs[1]["updates"] if call_kwargs[1] else call_kwargs[0][0]
        main_entry = updates["service"]["databases"]["main_db"]
        assert main_entry.get("name") == "alias_name"


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

    def test_name_with_backslash_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("path\\name")
        assert ok is False

    def test_name_with_colon_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("db:name")
        assert ok is False

    def test_name_with_tab_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("db\tname")
        assert ok is False

    def test_name_with_newline_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("db\nname")
        assert ok is False

    def test_name_with_asterisk_returns_false(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("db*name")
        assert ok is False

    def test_valid_name_with_hyphens_and_numbers(self):
        from datus.cli.service_manager import _validate_db_name

        ok, msg = _validate_db_name("my-database-01")
        assert ok is True


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

    def test_port_boundary_1_returns_true(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("1")
        assert ok is True

    def test_port_boundary_65535_returns_true(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("65535")
        assert ok is True

    def test_negative_port_returns_false(self):
        from datus.cli.service_manager import _validate_port

        ok, msg = _validate_port("-1")
        assert ok is False
