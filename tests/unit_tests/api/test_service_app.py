"""Tests for datus.api.service — FastAPI app creation and DatusAPIService."""

import argparse

from datus.api.service import DatusAPIService, create_app


class TestCreateApp:
    """Tests for create_app — FastAPI application factory."""

    def test_create_app_returns_fastapi(self):
        """create_app returns a FastAPI application."""
        from fastapi import FastAPI

        args = argparse.Namespace(config="", datasource="default", output_dir="./output", log_level="INFO")
        app = create_app(args)
        assert isinstance(app, FastAPI)

    def test_create_app_has_routes(self):
        """create_app registers expected route paths."""
        args = argparse.Namespace(config="", datasource="default", output_dir="./output", log_level="INFO")
        app = create_app(args)
        route_paths = [route.path for route in app.routes]
        assert "/" in route_paths
        assert "/health" in route_paths

    def test_create_app_stores_agent_args(self):
        """create_app stores agent_args in app.state."""
        args = argparse.Namespace(config="", datasource="test_ns", output_dir="./output", log_level="INFO")
        app = create_app(args)
        assert app.state.agent_args is args
        assert app.state.agent_args.datasource == "test_ns"

    def test_create_app_includes_cors_middleware(self):
        """create_app adds CORS middleware."""
        args = argparse.Namespace(config="", datasource="default", output_dir="./output", log_level="INFO")
        app = create_app(args)
        # CORS middleware is in the middleware stack
        # FastAPI stores user middleware as Middleware objects
        assert len(app.user_middleware) >= 1

    def test_create_app_registers_v1_routers(self):
        """create_app registers API v1 route prefixes."""
        args = argparse.Namespace(config="", datasource="default", output_dir="./output", log_level="INFO")
        app = create_app(args)
        route_paths = {route.path for route in app.routes}
        # Check that at least some v1 routes are registered
        has_api_routes = any("/api/" in p for p in route_paths)
        assert has_api_routes


class TestDatusAPIServiceInit:
    """Tests for DatusAPIService initialization."""

    def test_init_sets_defaults(self):
        """DatusAPIService starts with empty agents, no config, no task store."""
        args = argparse.Namespace(config="", namespace="default")
        svc = DatusAPIService(args)
        assert svc.agents == {}
        assert svc.agent_config is None
        assert svc.task_store is None
        assert svc.args is args


class TestDatusAPIServiceParseCsv:
    """Tests for _parse_csv_to_list — CSV parsing utility."""

    def test_parse_valid_csv(self):
        """_parse_csv_to_list parses valid CSV string."""
        svc = DatusAPIService(argparse.Namespace())
        result = svc._parse_csv_to_list("name,age\nAlice,30\nBob,25")
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == "30"
        assert result[1]["name"] == "Bob"

    def test_parse_empty_csv(self):
        """_parse_csv_to_list returns empty list for empty string."""
        svc = DatusAPIService(argparse.Namespace())
        assert svc._parse_csv_to_list("") == []

    def test_parse_none_csv(self):
        """_parse_csv_to_list returns empty list for None."""
        svc = DatusAPIService(argparse.Namespace())
        assert svc._parse_csv_to_list(None) == []

    def test_parse_whitespace_csv(self):
        """_parse_csv_to_list returns empty list for whitespace."""
        svc = DatusAPIService(argparse.Namespace())
        assert svc._parse_csv_to_list("   ") == []

    def test_parse_single_row_csv(self):
        """_parse_csv_to_list parses single row CSV."""
        svc = DatusAPIService(argparse.Namespace())
        result = svc._parse_csv_to_list("col1,col2\nval1,val2")
        assert len(result) == 1


class TestDatusAPIServiceGenerateTaskId:
    """Tests for _generate_task_id."""

    def test_generate_task_id_contains_client_id(self):
        """_generate_task_id includes client_id in result."""
        svc = DatusAPIService(argparse.Namespace())
        task_id = svc._generate_task_id("client_abc")
        assert "client_abc" in task_id

    def test_generate_task_id_contains_timestamp(self):
        """_generate_task_id includes timestamp format."""
        svc = DatusAPIService(argparse.Namespace())
        task_id = svc._generate_task_id("client")
        # Format: client_YYYYMMDDHHmmss
        parts = task_id.split("_")
        assert len(parts) >= 2
        assert len(parts[-1]) == 14  # timestamp length

    def test_generate_task_id_unique(self):
        """_generate_task_id produces unique IDs for same client."""
        import time

        svc = DatusAPIService(argparse.Namespace())
        id1 = svc._generate_task_id("c1")
        time.sleep(1.1)
        id2 = svc._generate_task_id("c1")
        assert id1 != id2
