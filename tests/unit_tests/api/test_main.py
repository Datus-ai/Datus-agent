"""Tests for datus.api.main — CLI argument parsing for datus-api command."""

import argparse
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
