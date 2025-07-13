"""Tests for configuration management."""

from pathlib import Path

from src.mcp_metricflow_server.config import MetricFlowConfig


class TestMetricFlowConfig:
    """Test MetricFlow configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetricFlowConfig()

        assert config.mf_path == "/path/to/mf"
        assert config.project_dir is None
        assert config.verbose is False

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("MF_PATH", "/usr/local/bin/mf")
        monkeypatch.setenv("MF_PROJECT_DIR", "/path/to/project")
        monkeypatch.setenv("MF_VERBOSE", "true")

        config = MetricFlowConfig.from_env()

        assert config.mf_path == "/usr/local/bin/mf"
        assert config.project_dir == Path("/path/to/project")
        assert config.verbose is True

    def test_config_from_env_defaults(self, monkeypatch):
        """Test configuration from environment with defaults."""
        # Clear environment variables
        monkeypatch.delenv("MF_PATH", raising=False)
        monkeypatch.delenv("MF_PROJECT_DIR", raising=False)
        monkeypatch.delenv("MF_VERBOSE", raising=False)

        config = MetricFlowConfig.from_env()

        assert config.mf_path == "/path/to/mf"
        assert config.project_dir is None
        assert config.verbose is False
