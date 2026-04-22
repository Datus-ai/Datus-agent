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
        args.datasource = ""

        with patch("datus.configuration.agent_config_loader.load_agent_config", side_effect=Exception("not found")):
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
        args.datasource = ""

        mock_config = MagicMock()
        mock_config.services.datasources = {}

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
        mock_agent_config.services.datasources = {"test_db": mock_db_cfg}

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
        mock_agent_config.services.datasources = {}

        result = iw._probe_database(mock_agent_config, "nonexistent_db")
        assert result == ""

    def test_probe_database_exception_returns_empty(self):
        """_probe_database returns empty string when DBManager raises."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_agent_config = MagicMock()
        mock_agent_config.services.datasources = {"db": mock_db_cfg}

        with patch("datus.tools.db_tools.db_manager.DBManager", side_effect=RuntimeError("connect failed")):
            result = iw._probe_database(mock_agent_config, "db")

        assert result == ""

    def test_probe_database_empty_tables_returns_no_tables_message(self):
        """_probe_database returns 'no tables found' when connector.get_tables() returns []."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_db_cfg.type = "sqlite"

        mock_agent_config = MagicMock()
        mock_agent_config.services.datasources = {"mydb": mock_db_cfg}

        mock_connector = MagicMock()
        mock_connector.get_tables.return_value = []

        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        with patch("datus.tools.db_tools.db_manager.DBManager", return_value=mock_db_manager):
            result = iw._probe_database(mock_agent_config, "mydb")

        assert "no tables found" in result
        assert "mydb" in result

    def test_probe_database_more_than_30_tables_truncates(self):
        """_probe_database truncates the table list when more than 30 tables are returned."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_db_cfg.type = "postgresql"

        mock_agent_config = MagicMock()
        mock_agent_config.services.datasources = {"bigdb": mock_db_cfg}

        # 35 tables — triggers the > 30 truncation branch
        tables = [{"table_name": f"table_{i}"} for i in range(35)]
        mock_connector = MagicMock()
        mock_connector.get_tables.return_value = tables

        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        with patch("datus.tools.db_tools.db_manager.DBManager", return_value=mock_db_manager):
            result = iw._probe_database(mock_agent_config, "bigdb")

        assert "and 5 more tables" in result
        assert "bigdb" in result

    def test_probe_database_table_name_fallback_to_name_key(self):
        """_probe_database resolves table name from 'name' key if 'table_name' is absent."""
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)

        mock_db_cfg = MagicMock()
        mock_db_cfg.type = "duckdb"

        mock_agent_config = MagicMock()
        mock_agent_config.services.datasources = {"duck": mock_db_cfg}

        mock_connector = MagicMock()
        mock_connector.get_tables.return_value = [{"name": "sales"}, {"name": "users"}]

        mock_db_manager = MagicMock()
        mock_db_manager.get_conn.return_value = mock_connector

        with patch("datus.tools.db_tools.db_manager.DBManager", return_value=mock_db_manager):
            result = iw._probe_database(mock_agent_config, "duck")

        assert "sales" in result
        assert "users" in result


def _make_mock_llm_module(model_class_name: str, llm_instance):
    """Inject a fake model module into sys.modules so __import__ in _generate_with_llm works."""
    import sys

    module_name = "datus.models.openai_model"
    mock_mod = MagicMock()
    setattr(mock_mod, model_class_name, MagicMock(return_value=llm_instance))
    sys.modules[module_name] = mock_mod
    return module_name


class TestInitWorkspaceGenerateWithLLM:
    """Tests for InitWorkspace._generate_with_llm()."""

    def _make_iw(self, tmp_path):
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        iw = InitWorkspace(args)
        iw.project_dir = str(tmp_path)
        iw.project_name = "test_proj"
        iw.agents_md_path = str(tmp_path / "AGENTS.md")
        return iw

    def _inject_model_module(self, model_type: str, model_class_name: str, llm_instance):
        """Inject a fake datus.models.{model_type}_model module into sys.modules."""
        import sys

        module_name = f"datus.models.{model_type}_model"
        mock_mod = MagicMock()
        setattr(mock_mod, model_class_name, MagicMock(return_value=llm_instance))
        sys.modules[module_name] = mock_mod
        return module_name

    def test_generate_with_llm_returns_content_when_llm_succeeds(self, tmp_path):
        """_generate_with_llm returns the LLM response when it is long enough."""
        import sys

        iw = self._make_iw(tmp_path)

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "x" * 200  # > 100 chars

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}):
                result = iw._generate_with_llm(mock_agent_config, "./\n  main.py", "Python", "No services\n")
        finally:
            sys.modules.pop(module_name, None)

        assert result is not None
        assert len(result) > 100

    def test_generate_with_llm_returns_none_when_response_too_short(self, tmp_path):
        """_generate_with_llm returns None when LLM response is <= 100 chars."""
        import sys

        iw = self._make_iw(tmp_path)

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "short"

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}):
                result = iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")
        finally:
            sys.modules.pop(module_name, None)

        assert result is None

    def test_generate_with_llm_returns_none_for_unknown_model_type(self, tmp_path):
        """_generate_with_llm returns None when model type is not in MODEL_TYPE_MAP."""
        iw = self._make_iw(tmp_path)

        mock_model_config = MagicMock()
        mock_model_config.type = "unknown_provider"

        mock_agent_config = MagicMock()
        mock_agent_config.active_model.return_value = mock_model_config

        with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {}):
            result = iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")

        assert result is None

    def test_generate_with_llm_returns_none_on_exception(self, tmp_path):
        """_generate_with_llm returns None when LLM call raises an exception."""
        iw = self._make_iw(tmp_path)

        mock_agent_config = MagicMock()
        mock_agent_config.active_model.side_effect = RuntimeError("model error")

        result = iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")
        assert result is None

    def test_generate_with_llm_reads_readme_when_present(self, tmp_path):
        """_generate_with_llm reads README.md from project dir and includes it in the prompt."""
        import sys

        (tmp_path / "README.md").write_text("This is the project readme." * 20)
        iw = self._make_iw(tmp_path)

        captured_prompts = []

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = lambda p: (captured_prompts.append(p), "x" * 200)[1]

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}):
                iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")
        finally:
            sys.modules.pop(module_name, None)

        assert len(captured_prompts) == 1
        assert "README excerpt" in captured_prompts[0]

    def test_generate_with_llm_includes_db_schema_section_when_provided(self, tmp_path):
        """_generate_with_llm includes the db_schema_info in the prompt and Data Tables section."""
        import sys

        iw = self._make_iw(tmp_path)

        captured_prompts = []

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = lambda p: (captured_prompts.append(p), "x" * 200)[1]

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            db_schema = "Database 'mydb' (sqlite) — 3 tables:\n  - orders\n  - users\n  - products\n"

            with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}):
                iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n", db_schema)
        finally:
            sys.modules.pop(module_name, None)

        assert "Data Tables" in captured_prompts[0]
        assert "Database schema" in captured_prompts[0]

    def test_generate_with_llm_returns_none_when_llm_generate_raises(self, tmp_path):
        """_generate_with_llm returns None when llm.generate() raises an exception."""
        import sys

        iw = self._make_iw(tmp_path)

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = RuntimeError("API error")

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            with patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}):
                result = iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")
        finally:
            sys.modules.pop(module_name, None)

        assert result is None

    def test_generate_with_llm_silently_skips_unreadable_readme(self, tmp_path):
        """_generate_with_llm skips README gracefully when open() raises an exception."""
        import sys

        # Create a README.md so the os.path.exists check passes, but mock open to fail
        readme_path = tmp_path / "README.md"
        readme_path.write_text("content")

        iw = self._make_iw(tmp_path)

        captured_prompts = []

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.side_effect = lambda p: (captured_prompts.append(p), "x" * 200)[1]

        module_name = self._inject_model_module("openai", "OpenAIModel", mock_llm_instance)
        try:
            mock_model_config = MagicMock()
            mock_model_config.type = "openai"

            mock_agent_config = MagicMock()
            mock_agent_config.active_model.return_value = mock_model_config

            real_open = open

            def open_that_fails_for_readme(path, *a, **kw):
                if str(path).endswith("README.md"):
                    raise OSError("permission denied")
                return real_open(path, *a, **kw)

            with (
                patch("datus.models.base.LLMBaseModel.MODEL_TYPE_MAP", {"openai": "OpenAIModel"}),
                patch("builtins.open", side_effect=open_that_fails_for_readme),
            ):
                result = iw._generate_with_llm(mock_agent_config, "./", "Python", "No services\n")
        finally:
            sys.modules.pop(module_name, None)

        # LLM was still called despite README read failure
        assert result is not None
        assert len(result) > 100
        # README excerpt should NOT appear in the prompt
        assert "README excerpt" not in captured_prompts[0]


class TestInitWorkspaceRunFullFlow:
    """Tests for InitWorkspace.run() covering the full happy path and edge cases."""

    def _setup_iw(self, tmp_path, database=""):
        from datus.cli.init_workspace import InitWorkspace

        args = MagicMock()
        args.config = ""
        args.datasource = database

        iw = InitWorkspace(args)
        iw.project_dir = str(tmp_path)
        iw.project_name = "myproject"
        iw.agents_md_path = str(tmp_path / "AGENTS.md")
        return iw

    def test_run_creates_agents_md_using_llm_content(self, tmp_path):
        """run() writes LLM-generated content to AGENTS.md and returns 0."""
        iw = self._setup_iw(tmp_path)

        mock_config = MagicMock()
        mock_config.services.datasources = {}

        llm_content = "# myproject\n\n## Architecture\n\nSome content here." * 10

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch.object(iw, "_generate_with_llm", return_value=llm_content),
        ):
            ret = iw.run()

        assert ret == 0
        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        assert llm_content in agents_md.read_text()

    def test_run_falls_back_to_template_when_llm_returns_none(self, tmp_path):
        """run() uses template when _generate_with_llm returns None."""
        iw = self._setup_iw(tmp_path)

        mock_config = MagicMock()
        mock_config.services.datasources = {}

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch.object(iw, "_generate_with_llm", return_value=None),
        ):
            ret = iw.run()

        assert ret == 0
        content = (tmp_path / "AGENTS.md").read_text()
        assert "## Architecture" in content

    def test_run_probes_database_when_database_arg_set(self, tmp_path):
        """run() calls _probe_database when args.datasource is non-empty."""
        iw = self._setup_iw(tmp_path, database="mydb")

        mock_config = MagicMock()
        mock_config.services.datasources = {}

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch.object(iw, "_probe_database", return_value="db schema info") as mock_probe,
            patch.object(iw, "_generate_with_llm", return_value="x" * 200),
        ):
            ret = iw.run()

        assert ret == 0
        mock_probe.assert_called_once_with(mock_config, "mydb")

    def test_run_overwrites_existing_agents_md_when_user_selects_overwrite(self, tmp_path):
        """run() overwrites AGENTS.md when user selects 'overwrite'."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# old content")

        iw = self._setup_iw(tmp_path)

        mock_config = MagicMock()
        mock_config.services.datasources = {}

        new_content = "# myproject\n\n## Architecture\nNew content." * 10

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch("datus.cli.init_workspace.Prompt.ask", return_value="overwrite"),
            patch.object(iw, "_generate_with_llm", return_value=new_content),
        ):
            ret = iw.run()

        assert ret == 0
        assert agents_md.read_text() == new_content

    def test_run_returns_1_on_keyboard_interrupt(self, tmp_path):
        """run() returns 1 when KeyboardInterrupt is raised during execution."""
        iw = self._setup_iw(tmp_path)

        with patch(
            "datus.configuration.agent_config_loader.load_agent_config",
            side_effect=KeyboardInterrupt,
        ):
            ret = iw.run()

        assert ret == 1

    def test_run_returns_1_on_unexpected_exception(self, tmp_path):
        """run() returns 1 and logs error when an unexpected exception is raised."""
        iw = self._setup_iw(tmp_path)

        mock_config = MagicMock()
        mock_config.services.datasources = {}

        with (
            patch("datus.configuration.agent_config_loader.load_agent_config", return_value=mock_config),
            patch.object(iw, "_generate_with_llm", side_effect=RuntimeError("unexpected")),
        ):
            ret = iw.run()

        assert ret == 1
