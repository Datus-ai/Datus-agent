# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/main.py.

CI-level: zero external dependencies. All I/O and subprocess calls mocked.
"""

import sys
from unittest.mock import MagicMock, patch

from datus.main import create_parser, main

# ---------------------------------------------------------------------------
# create_parser
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_parser_created_successfully(self):
        parser = create_parser()
        assert parser.description == "Datus: AI-powered SQL Agent for data engineering"

    def test_no_action_parses_ok(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.action is None

    def test_run_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--datasource",
                "myns",
                "--task",
                "count rows",
                "--task_db_name",
                "mydb",
            ]
        )
        assert args.action == "run"
        assert args.datasource == "myns"
        assert args.task == "count rows"
        assert args.task_db_name == "mydb"

    def test_benchmark_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "--benchmark",
                "bird_dev",
                "--datasource",
                "testns",
            ]
        )
        assert args.action == "benchmark"
        assert args.benchmark == "bird_dev"

    def test_check_db_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "check-db",
                "--datasource",
                "testns",
            ]
        )
        assert args.action == "check-db"
        assert args.datasource == "testns"

    def test_probe_llm_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(["probe-llm"])
        assert args.action == "probe-llm"

    def test_service_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(["service", "list"])
        assert args.action == "service"
        assert args.command == "list"

    def test_skill_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(["skill", "list"])
        assert args.action == "skill"
        assert args.subcommand == "list"

    def test_eval_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "eval",
                "--datasource",
                "myns",
                "--benchmark",
                "bird_dev",
            ]
        )
        assert args.action == "eval"

    def test_generate_dataset_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "generate-dataset",
                "--dataset_name",
                "my_dataset",
            ]
        )
        assert args.action == "generate-dataset"
        assert args.dataset_name == "my_dataset"

    def test_debug_flag_global(self):
        parser = create_parser()
        # --debug is a global option; must appear before subcommand
        args = parser.parse_args(["probe-llm", "--debug"])
        assert args.debug is True

    def test_schema_linking_rate_default(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--datasource",
                "ns",
                "--task",
                "do something",
                "--task_db_name",
                "db",
            ]
        )
        assert args.schema_linking_rate == "fast"

    def test_max_steps_default(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "run",
                "--datasource",
                "ns",
                "--task",
                "do something",
                "--task_db_name",
                "db",
            ]
        )
        assert args.max_steps == 20

    def test_platform_doc_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(["platform-doc"])
        assert args.action == "platform-doc"

    def test_bootstrap_kb_action_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "bootstrap-kb",
                "--datasource",
                "bird_school",
                "--components",
                "metadata",
                "reference_sql",
                "--kb_update_strategy",
                "overwrite",
                "--sql_dir",
                "/tmp/sql",
                "--subject_tree",
                "a/b,c/d",
                "--yes",
            ]
        )
        assert args.action == "bootstrap-kb"
        assert args.datasource == "bird_school"
        assert args.components == ["metadata", "reference_sql"]
        assert args.kb_update_strategy == "overwrite"
        assert args.sql_dir == "/tmp/sql"
        assert args.subject_tree == "a/b,c/d"
        assert args.yes is True
        assert args.pool_size == 4

    def test_bootstrap_kb_requires_datasource(self):
        import pytest

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bootstrap-kb", "--components", "metadata"])

    def test_bootstrap_kb_accepts_table_lineage_component(self):
        parser = create_parser()
        args = parser.parse_args(["bootstrap-kb", "--datasource", "ds", "--components", "table_lineage"])
        assert args.components == ["table_lineage"]

    def test_bootstrap_kb_accepts_check_strategy(self):
        parser = create_parser()
        args = parser.parse_args(["bootstrap-kb", "--datasource", "ds", "--kb_update_strategy", "check"])
        assert args.kb_update_strategy == "check"

    def test_bootstrap_kb_accepts_refresh_profile_strategy(self):
        parser = create_parser()
        args = parser.parse_args(["bootstrap-kb", "--datasource", "ds", "--kb_update_strategy", "refresh-profile"])
        assert args.kb_update_strategy == "refresh-profile"

    def test_bootstrap_kb_default_components_and_strategy(self):
        parser = create_parser()
        args = parser.parse_args(["bootstrap-kb", "--datasource", "ds"])
        assert args.components == ["metadata"]
        assert args.kb_update_strategy == "check"

    def test_bootstrap_kb_extended_flags_parsed(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "bootstrap-kb",
                "--datasource",
                "ds",
                "--benchmark",
                "spider2",
                "--from_adapter",
                "metricflow",
                "--semantic_yaml",
                "/tmp/s.yaml",
                "--validate-only",
                "--catalog",
                "main",
                "--database_name",
                "demo",
                "--schema_linking_type",
                "view",
            ]
        )
        assert args.benchmark == "spider2"
        assert args.from_adapter == "metricflow"
        assert args.semantic_yaml == "/tmp/s.yaml"
        assert args.validate_only is True
        assert args.catalog == "main"
        assert args.database_name == "demo"
        assert args.schema_linking_type == "view"


# ---------------------------------------------------------------------------
# main() – action routing
# ---------------------------------------------------------------------------


def _run_main(argv):
    """Helper: patch sys.argv and call main()."""
    with patch.object(sys, "argv", ["datus"] + argv):
        return main()


class TestMainNoAction:
    def test_no_action_prints_help_and_returns_1(self):
        parser = create_parser()
        with patch.object(sys, "argv", ["datus"]):
            with patch("datus.main.create_parser", return_value=parser):
                with patch.object(parser, "print_help"):
                    result = main()
        assert result == 1


class TestMainServiceAction:
    def test_service_list_runs_service_manager(self):
        mock_mgr = MagicMock()
        mock_mgr.run.return_value = 0
        with (
            patch("datus.main.configure_logging"),
            patch("datus.cli.service_manager.ServiceManager", return_value=mock_mgr),
            patch.object(sys, "argv", ["datus", "service", "list"]),
        ):
            main()
        mock_mgr.run.assert_called_with("list")


class TestMainSkillAction:
    def test_skill_action_calls_run_skill_command(self):
        mock_run = MagicMock(return_value=0)
        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.run_skill_command", mock_run, create=True),
            patch.object(sys, "argv", ["datus", "skill", "list"]),
        ):
            # Patch the import inside main
            with patch.dict("sys.modules", {"datus.cli.skill_cli": MagicMock(run_skill_command=mock_run)}):
                result = main()
        assert result == 0


class TestMainCheckDbAction:
    def test_check_db_action(self):
        mock_agent = MagicMock()
        mock_agent.check_db.return_value = {"status": "success"}
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(sys, "argv", ["datus", "check-db", "--datasource", "myns"]),
        ):
            result = main()

        mock_agent.check_db.assert_called_once()
        assert result == 0


class TestMainProbeLlmAction:
    def test_probe_llm_action(self):
        mock_agent = MagicMock()
        mock_agent.probe_llm.return_value = {"status": "success", "response": "ok"}
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(sys, "argv", ["datus", "probe-llm"]),
        ):
            result = main()

        mock_agent.probe_llm.assert_called_once()
        assert result == 0


class TestMainRunAction:
    def test_run_action_creates_sql_task(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_config = MagicMock()
        mock_config.current_db_name_type.return_value = ("mydb", "sqlite")
        mock_config.output_dir = "/tmp/output"
        mock_config.current_datasource = "ns"

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                ["datus", "run", "--datasource", "ns", "--task", "count rows", "--task_db_name", "mydb"],
            ),
        ):
            result = main()

        mock_agent.run.assert_called_once()
        assert result == 0

    def test_run_action_with_load_cp_skips_task_creation(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_config = MagicMock()
        mock_config.current_db_name_type.return_value = ("mydb", "sqlite")
        mock_config.output_dir = "/tmp/output"

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                [
                    "datus",
                    "run",
                    "--datasource",
                    "ns",
                    "--task",
                    "count rows",
                    "--task_db_name",
                    "mydb",
                    "--load_cp",
                    "cp.json",
                ],
            ),
        ):
            main()

        # When load_cp is set, run is called with check_storage=True
        call_kwargs = mock_agent.run.call_args
        assert call_kwargs[1].get("check_storage") is True


class TestMainBenchmarkAction:
    def test_benchmark_action(self):
        mock_agent = MagicMock()
        mock_agent.benchmark.return_value = None
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                ["datus", "benchmark", "--benchmark", "bird_dev", "--datasource", "ns"],
            ),
        ):
            result = main()

        mock_agent.benchmark.assert_called_once()
        assert result == 0


class TestMainEvalAction:
    def test_eval_action(self):
        mock_agent = MagicMock()
        mock_agent.evaluation.return_value = None
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                ["datus", "eval", "--datasource", "ns", "--benchmark", "bird_dev"],
            ),
        ):
            result = main()

        mock_agent.evaluation.assert_called_once()
        assert result == 0


class TestMainPlatformDocAction:
    def test_platform_doc_action(self):
        mock_config = MagicMock()
        mock_bootstrap = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.agent.agent.bootstrap_platform_doc", mock_bootstrap),
            patch.dict(
                "sys.modules",
                {"datus.agent.agent": __import__("datus.agent.agent", fromlist=["bootstrap_platform_doc"])},
            ),
            patch.object(sys, "argv", ["datus", "platform-doc"]),
        ):
            # Patch the local import inside main()
            import datus.agent.agent as _agent_mod

            original = getattr(_agent_mod, "bootstrap_platform_doc", None)
            _agent_mod.bootstrap_platform_doc = mock_bootstrap
            try:
                result = main()
            finally:
                if original is not None:
                    _agent_mod.bootstrap_platform_doc = original
                elif hasattr(_agent_mod, "bootstrap_platform_doc"):
                    del _agent_mod.bootstrap_platform_doc

        mock_bootstrap.assert_called_once()
        assert result == 0


class TestMainBootstrapKbAction:
    def test_bootstrap_kb_action_calls_agent_bootstrap_kb(self):
        mock_agent = MagicMock()
        mock_agent.bootstrap_kb.return_value = {"status": "success", "message": "ok"}
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                ["datus", "bootstrap-kb", "--datasource", "ds", "--components", "metadata"],
            ),
        ):
            result = main()

        mock_agent.bootstrap_kb.assert_called_once()
        assert result == 0


class TestMainGenerateDatasetAction:
    def test_generate_dataset_action(self):
        mock_agent = MagicMock()
        mock_agent.generate_dataset.return_value = None
        mock_config = MagicMock()

        with (
            patch("datus.main.configure_logging"),
            patch("datus.main.setup_exception_handler"),
            patch("datus.main.load_agent_config", return_value=mock_config),
            patch("datus.main.Agent", return_value=mock_agent),
            patch.object(
                sys,
                "argv",
                ["datus", "generate-dataset", "--dataset_name", "ds1"],
            ),
        ):
            result = main()

        mock_agent.generate_dataset.assert_called_once()
        assert result == 0
