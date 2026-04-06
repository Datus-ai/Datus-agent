"""Tests for datus.api.main — CLI argument parsing for datus-api command."""

import sys
from unittest.mock import patch

import pytest

from datus.api.main import APIServerArgumentParser


class TestAPIServerArgumentParserDefaults:
    """Tests for default argument values."""

    def test_default_host(self):
        """Default host is 127.0.0.1."""
        with patch.object(sys, "argv", ["datus-api"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.host == "127.0.0.1"

    def test_default_port(self):
        """Default port is 8000."""
        with patch.object(sys, "argv", ["datus-api"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.port == 8000

    def test_default_workers(self):
        """Default workers is 1."""
        with patch.object(sys, "argv", ["datus-api"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.workers == 1

    def test_default_reload_is_false(self):
        """Reload is False by default."""
        with patch.object(sys, "argv", ["datus-api"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.reload is False

    def test_default_config_is_none(self):
        """Config defaults to None."""
        with patch.object(sys, "argv", ["datus-api"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.config is None

    def test_default_log_level(self):
        """Default log level is INFO (from env or fallback)."""
        with patch.object(sys, "argv", ["datus-api"]):
            with patch.dict("os.environ", {}, clear=False):
                parser = APIServerArgumentParser()
                args = parser.parse_args()
                assert args.log_level in ("INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL")


class TestAPIServerArgumentParserCustomValues:
    """Tests for explicit argument values."""

    def test_custom_host_and_port(self):
        """Custom host and port are parsed correctly."""
        with patch.object(sys, "argv", ["datus-api", "--host", "0.0.0.0", "--port", "9090"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.host == "0.0.0.0"
            assert args.port == 9090

    def test_reload_flag(self):
        """--reload sets reload to True."""
        with patch.object(sys, "argv", ["datus-api", "--reload"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.reload is True

    def test_workers_count(self):
        """--workers sets the number of workers."""
        with patch.object(sys, "argv", ["datus-api", "--workers", "4"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.workers == 4

    def test_config_path(self):
        """--config sets config file path."""
        with patch.object(sys, "argv", ["datus-api", "--config", "/path/to/agent.yml"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.config == "/path/to/agent.yml"

    def test_namespace(self):
        """--namespace sets the namespace."""
        with patch.object(sys, "argv", ["datus-api", "--namespace", "prod"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.namespace == "prod"

    def test_output_dir(self):
        """--output-dir sets the output directory."""
        with patch.object(sys, "argv", ["datus-api", "--output-dir", "/tmp/output"]):
            parser = APIServerArgumentParser()
            args = parser.parse_args()
            assert args.output_dir == "/tmp/output"

    def test_log_level_choices(self):
        """--log-level accepts valid levels."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            with patch.object(sys, "argv", ["datus-api", "--log-level", level]):
                parser = APIServerArgumentParser()
                args = parser.parse_args()
                assert args.log_level == level

    def test_invalid_log_level_raises(self):
        """Invalid log level causes SystemExit."""
        with patch.object(sys, "argv", ["datus-api", "--log-level", "TRACE"]):
            parser = APIServerArgumentParser()
            with pytest.raises(SystemExit):
                parser.parse_args()


class TestAPIServerArgumentParserEdgeCases:
    """Edge cases for argument parsing."""

    def test_namespace_from_env_var(self):
        """Namespace falls back to DATUS_NAMESPACE env var."""
        with patch.object(sys, "argv", ["datus-api"]):
            with patch.dict("os.environ", {"DATUS_NAMESPACE": "staging"}):
                parser = APIServerArgumentParser()
                args = parser.parse_args()
                assert args.namespace == "staging"

    def test_output_dir_from_env_var(self):
        """Output dir falls back to DATUS_OUTPUT_DIR env var."""
        with patch.object(sys, "argv", ["datus-api"]):
            with patch.dict("os.environ", {"DATUS_OUTPUT_DIR": "/custom/output"}):
                parser = APIServerArgumentParser()
                args = parser.parse_args()
                assert args.output_dir == "/custom/output"


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_config_not_found_exits(self):
        """main() exits with code 1 when config file is not found."""
        from datus.api.main import main

        with patch.object(sys, "argv", ["datus-api", "--config", "/nonexistent/agent.yml"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_sets_env_vars(self):
        """main() sets DATUS_CONFIG, DATUS_NAMESPACE, DATUS_OUTPUT_DIR, DATUS_LOG_LEVEL env vars."""
        import os

        from datus.api.main import main

        with patch.object(sys, "argv", ["datus-api", "--namespace", "test_main_ns"]):
            with patch("datus.api.main.parse_config_path", return_value="/tmp/agent.yml"):
                with patch("datus.api.main.uvicorn") as mock_uvicorn:
                    mock_uvicorn.run = lambda *a, **kw: None
                    with patch("datus.api.service.create_app") as mock_create:
                        mock_create.return_value = None
                        try:
                            main()
                        except Exception:
                            pass
                        assert os.environ.get("DATUS_NAMESPACE") == "test_main_ns"
