# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/init_workspace.py"""

from unittest.mock import MagicMock, patch


class TestScanDirectory:
    """Tests for _scan_directory()."""

    def test_returns_tree_output_for_temp_dir(self, tmp_path):
        """_scan_directory returns a non-empty tree string for a real directory."""
        from datus.cli.init_workspace import _scan_directory

        # Create some files and a subdirectory
        (tmp_path / "README.md").write_text("hello")
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("pass")

        result = _scan_directory(str(tmp_path))
        assert isinstance(result, str)
        assert len(result) > 0
        # Root dir marker
        assert "./" in result

    def test_ignores_hidden_directories(self, tmp_path):
        """_scan_directory skips hidden and noise directories."""
        from datus.cli.init_workspace import _scan_directory

        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "HEAD").write_text("ref: refs/heads/main")

        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("")

        (tmp_path / "main.py").write_text("pass")

        result = _scan_directory(str(tmp_path))
        assert ".git" not in result
        assert ".venv" not in result
        assert "main.py" in result

    def test_respects_max_depth(self, tmp_path):
        """_scan_directory does not recurse beyond max_depth."""
        from datus.cli.init_workspace import _scan_directory

        # Create a 4-level deep directory structure
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep_file.txt").write_text("deep")

        result = _scan_directory(str(tmp_path), max_depth=2)
        # deep_file.txt is at depth 4, should not appear
        assert "deep_file.txt" not in result

    def test_truncates_many_files_with_ellipsis(self, tmp_path):
        """_scan_directory uses ... when more than 8 files exist in a dir."""
        from datus.cli.init_workspace import _scan_directory

        for i in range(12):
            (tmp_path / f"file_{i:02d}.txt").write_text("")

        result = _scan_directory(str(tmp_path))
        assert "more files" in result


class TestDetectProjectType:
    """Tests for _detect_project_type()."""

    def test_detects_python_pyproject_toml(self, tmp_path):
        """Returns Python (pyproject.toml) when pyproject.toml exists."""
        from datus.cli.init_workspace import _detect_project_type

        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        result = _detect_project_type(str(tmp_path))
        assert "Python (pyproject.toml)" in result

    def test_detects_nodejs(self, tmp_path):
        """Returns Node.js when package.json exists."""
        from datus.cli.init_workspace import _detect_project_type

        (tmp_path / "package.json").write_text('{"name": "test"}')
        result = _detect_project_type(str(tmp_path))
        assert "Node.js" in result

    def test_returns_unknown_for_empty_dir(self, tmp_path):
        """Returns 'Unknown' when no recognizable project files are found."""
        from datus.cli.init_workspace import _detect_project_type

        result = _detect_project_type(str(tmp_path))
        assert result == "Unknown"

    def test_detects_multiple_project_types(self, tmp_path):
        """Returns comma-separated types when multiple indicators exist."""
        from datus.cli.init_workspace import _detect_project_type

        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "Dockerfile").write_text("")
        result = _detect_project_type(str(tmp_path))
        assert "Python (pyproject.toml)" in result
        assert "Docker" in result


class TestBuildServicesSection:
    """Tests for _build_services_section()."""

    def test_empty_dict_returns_no_services_message(self):
        """Returns a 'No services configured' string for empty input."""
        from datus.cli.init_workspace import _build_services_section

        result = _build_services_section({})
        assert "No services configured" in result

    def test_with_db_config_entries_produces_table(self):
        """Returns a markdown table with database entries."""
        from datus.cli.init_workspace import _build_services_section

        db_cfg = MagicMock()
        db_cfg.type = "sqlite"
        db_cfg.uri = "path/to/data.sqlite"
        db_cfg.host = ""
        db_cfg.account = ""

        result = _build_services_section({"my_db": db_cfg})
        assert "my_db" in result
        assert "sqlite" in result
        assert "path/to/data.sqlite" in result

    def test_host_based_db_shows_host_port(self):
        """For host-based DBs, connection shows host:port."""
        from datus.cli.init_workspace import _build_services_section

        db_cfg = MagicMock()
        db_cfg.type = "postgresql"
        db_cfg.uri = ""
        db_cfg.host = "localhost"
        db_cfg.port = "5432"
        db_cfg.account = ""

        result = _build_services_section({"pg_db": db_cfg})
        assert "localhost:5432" in result

    def test_account_based_db_shows_account(self):
        """For account-based DBs (Snowflake), connection shows account=..."""
        from datus.cli.init_workspace import _build_services_section

        db_cfg = MagicMock()
        db_cfg.type = "snowflake"
        db_cfg.uri = ""
        db_cfg.host = ""
        db_cfg.account = "myaccount"

        result = _build_services_section({"sf_db": db_cfg})
        assert "account=myaccount" in result


class TestInitWorkspaceRun:
    """Tests for InitWorkspace.run()."""

    def test_run_with_missing_config_returns_1(self, tmp_path):
        """run() returns 1 when load_agent_config raises an exception."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        args.config = str(tmp_path / "missing.yml")
        args.database = ""

        with patch(
            "datus.configuration.agent_config_loader.load_agent_config", side_effect=Exception("not found")
        ):
            iw = InitWorkspace(args)
            iw.project_dir = str(tmp_path)
            iw.project_name = "test_project"
            iw.agents_md_path = str(tmp_path / "AGENTS.md")
            ret = iw.run()
        assert ret == 1

    def test_run_cancel_when_agents_md_exists(self, tmp_path):
        """run() returns 0 without overwriting when user selects 'cancel'."""
        from datus.cli.init_workspace import InitWorkspace

        # Pre-create AGENTS.md
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# existing")

        args = MagicMock()
        args.config = ""
        args.database = ""

        mock_config = MagicMock()
        mock_config.service.databases = {}

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch("datus.cli.init_workspace.Prompt.ask", return_value="cancel"),
        ):
            iw = InitWorkspace(args)
            iw.project_dir = str(tmp_path)
            iw.project_name = "test_project"
            iw.agents_md_path = str(agents_md)
            ret = iw.run()

        assert ret == 0
        # File content unchanged
        assert agents_md.read_text() == "# existing"


class TestInitWorkspaceGenerateTemplate:
    """Tests for InitWorkspace._generate_template()."""

    def test_template_contains_project_name(self, tmp_path):
        """Generated template contains the project name as heading."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)
        iw.project_name = "my_cool_project"

        content = iw._generate_template("./\n  main.py", "Python (pyproject.toml)", "No services configured\n")
        assert "my_cool_project" in content

    def test_template_contains_section_headers(self, tmp_path):
        """Generated template contains Architecture, Directory Map, Services, Artifacts headers."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)
        iw.project_name = "test"

        content = iw._generate_template(".", "Unknown", "No services configured\n")
        assert "## Architecture" in content
        assert "## Directory Map" in content
        assert "## Services" in content
        assert "## Artifacts" in content


class TestInitWorkspaceProbeDatabase:
    """Tests for InitWorkspace._probe_database()."""

    def test_probe_database_returns_table_list(self):
        """_probe_database returns formatted table list from connector."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_db_cfg.type = "sqlite"

        mock_agent_config = MagicMock()
        mock_agent_config.service.databases = {"test_db": mock_db_cfg}

        mock_connector = MagicMock()
        mock_connector.get_tables.return_value = [
            {"table_name": "orders"},
            {"table_name": "customers"},
        ]

        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        with patch("datus.tools.db_tools.db_manager.DBManager", return_value=mock_db_manager):
            result = iw._probe_database(mock_agent_config, "test_db")

        assert "orders" in result
        assert "customers" in result
        assert "test_db" in result

    def test_probe_database_missing_db_name_returns_empty(self):
        """_probe_database returns empty string when db_name not in config."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_agent_config = MagicMock()
        mock_agent_config.service.databases = {}

        result = iw._probe_database(mock_agent_config, "nonexistent_db")
        assert result == ""

    def test_probe_database_exception_returns_empty(self):
        """_probe_database returns empty string when DBManager raises."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_agent_config = MagicMock()
        mock_agent_config.service.databases = {"db": mock_db_cfg}

        with patch("datus.tools.db_tools.db_manager.DBManager", side_effect=RuntimeError("connect failed")):
            result = iw._probe_database(mock_agent_config, "db")

        assert result == ""
