from unittest.mock import patch

from datus.cli.web.config_manager import get_available_datasources


def test_get_available_datasources_from_wrapped_agent_config() -> None:
    config = {"agent": {"services": {"datasources": {"duckdb": {}, "snowflake": {}}}}}

    with patch("datus.cli.web.config_manager._load_config_cached", return_value=config):
        assert get_available_datasources("conf/agent.yml") == ["duckdb", "snowflake"]


def test_get_available_datasources_from_top_level_config() -> None:
    config = {"services": {"datasources": {"bird_sqlite": {}, "superset": {}}}}

    with patch("datus.cli.web.config_manager._load_config_cached", return_value=config):
        assert get_available_datasources("conf/agent.yml") == ["bird_sqlite", "superset"]


def test_get_available_datasources_returns_empty_on_missing_file() -> None:
    with (
        patch("datus.cli.web.config_manager._load_config_cached", side_effect=FileNotFoundError("missing")),
        patch("datus.cli.web.config_manager.logger") as mock_logger,
    ):
        assert get_available_datasources("missing.yml") == []
        mock_logger.error.assert_called_once()


def test_get_available_datasources_returns_empty_on_parse_error() -> None:
    with (
        patch("datus.cli.web.config_manager._load_config_cached", side_effect=ValueError("bad yaml")),
        patch("datus.cli.web.config_manager.logger") as mock_logger,
    ):
        assert get_available_datasources("bad.yml") == []
        mock_logger.error.assert_called_once()
