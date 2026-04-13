# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/web/chatbot.py (FastAPI-based web chatbot)."""

import argparse
import os
from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# 1. _build_agent_args
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestBuildAgentArgs:
    """Tests for the _build_agent_args bridge function."""

    def test_bridges_required_fields(self):
        from datus.cli.web.chatbot import _build_agent_args

        args = argparse.Namespace(
            namespace="myns",
            config="conf/agent.yml",
            debug=False,
        )
        result = _build_agent_args(args)

        assert result.namespace == "myns"
        assert result.config == "conf/agent.yml"
        assert result.source == "web"
        assert result.interactive is True
        assert result.workflow == "chat_agentic"
        assert result.log_level == "INFO"

    def test_debug_mode(self):
        from datus.cli.web.chatbot import _build_agent_args

        args = argparse.Namespace(namespace="ns", config=None, debug=True)
        result = _build_agent_args(args)

        assert result.log_level == "DEBUG"

    def test_missing_optional_fields(self):
        """Fields not present on CLI args should have safe defaults."""
        from datus.cli.web.chatbot import _build_agent_args

        args = argparse.Namespace(namespace="ns")
        result = _build_agent_args(args)

        assert result.config is None
        assert result.debug is False
        assert result.max_steps == 20


# ═══════════════════════════════════════════════════════════════════════════
# 2. _read_template
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestReadTemplate:
    """Tests for the _read_template helper."""

    def test_reads_html(self):
        from datus.cli.web.chatbot import _read_template

        html = _read_template()
        assert "DatusChatbot" in html
        assert "chatbot-root" in html
        assert "{{ request_origin_json }}" in html
        assert "{{ user_name_json }}" in html
        assert "{{ chatbot_js }}" in html
        assert "{{ chatbot_css }}" in html
        assert "{{ react_js }}" in html

    def test_returns_string(self):
        from datus.cli.web.chatbot import _read_template

        html = _read_template()
        assert isinstance(html, str)
        assert len(html) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. create_web_app
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestCreateWebApp:
    """Tests for create_web_app function."""

    def test_returns_fastapi_app(self):
        from datus.cli.web.chatbot import create_web_app

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist="/nonexistent/path",
            session_scope=None,
        )

        with patch("datus.cli.web.chatbot.create_app") as mock_create_app:
            from fastapi import FastAPI

            mock_app = FastAPI()
            mock_create_app.return_value = mock_app

            app = create_web_app(args)
            assert app is mock_app
            mock_create_app.assert_called_once()

    def test_mounts_chatbot_assets_when_dist_exists(self, tmp_path):
        """Static files should be mounted when dist directory exists."""
        from datus.cli.web.chatbot import create_web_app

        # Create a fake dist directory
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "datus-chatbot.umd.js").write_text("// js")
        (dist_dir / "datus-chatbot.css").write_text("/* css */")

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist=str(dist_dir),
            session_scope=None,
        )

        with patch("datus.cli.web.chatbot.create_app") as mock_create_app:
            from fastapi import FastAPI

            mock_create_app.return_value = FastAPI()
            app = create_web_app(args)

            # Check that /chatbot-assets route was mounted
            route_paths = [r.path for r in app.routes if hasattr(r, "path")]
            assert "/chatbot-assets" in route_paths or any("/chatbot-assets" in str(r) for r in app.routes)

    def test_uses_cdn_when_no_chatbot_dist(self):
        """Without --chatbot-dist, should use CDN URLs in rendered HTML."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from datus.cli.web.chatbot import _CDN_CHATBOT_JS, _CDN_REACT_JS, create_web_app

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist=None,
            session_scope=None,
        )

        with patch("datus.cli.web.chatbot.create_app") as mock_create_app:
            mock_create_app.return_value = FastAPI()
            app = create_web_app(args)

            # Should NOT mount /chatbot-assets
            route_paths = [r.path for r in app.routes if hasattr(r, "path")]
            assert "/chatbot-assets" not in route_paths

            # Verify CDN URLs are actually rendered in the HTML response
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/")
            assert resp.status_code == 200
            assert _CDN_CHATBOT_JS in resp.text
            assert _CDN_REACT_JS in resp.text

    def test_warns_when_dist_missing(self):
        """Should warn and fall back to CDN when dist path doesn't exist."""
        from datus.cli.web.chatbot import create_web_app

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist="/definitely/not/a/real/path",
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_app") as mock_create_app,
            patch("datus.cli.web.chatbot.logger") as mock_logger,
        ):
            from fastapi import FastAPI

            mock_create_app.return_value = FastAPI()
            app = create_web_app(args)

            assert app is not None
            mock_logger.warning.assert_called_once()

    def test_falls_back_to_cdn_when_dist_incomplete(self, tmp_path):
        """Should fall back to CDN when dist dir exists but bundle files are missing."""
        from datus.cli.web.chatbot import create_web_app

        # Create dir with only CSS, missing the JS bundle
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        (dist_dir / "datus-chatbot.css").write_text("/* css */")

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist=str(dist_dir),
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_app") as mock_create_app,
            patch("datus.cli.web.chatbot.logger") as mock_logger,
        ):
            from fastapi import FastAPI

            mock_create_app.return_value = FastAPI()
            app = create_web_app(args)

            # Should NOT mount /chatbot-assets
            route_paths = [r.path for r in app.routes if hasattr(r, "path")]
            assert "/chatbot-assets" not in route_paths
            mock_logger.warning.assert_called_once()

    def test_html_template_rendered(self):
        """The root route should serve HTML with config values substituted."""
        from datus.cli.web.chatbot import create_web_app

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="myhost",
            port=9999,
            debug=False,
            subagent="",
            chatbot_dist=None,
            session_scope=None,
        )

        with patch("datus.cli.web.chatbot.create_app") as mock_create_app:
            from fastapi import FastAPI

            mock_create_app.return_value = FastAPI()
            app = create_web_app(args)

            # Find the root route handler
            root_routes = [r for r in app.routes if hasattr(r, "path") and r.path == "/"]
            assert len(root_routes) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. run_web_interface
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestRunWebInterface:
    """Tests for run_web_interface entry point."""

    def test_calls_uvicorn_run(self):
        from datus.cli.web.chatbot import run_web_interface

        args = argparse.Namespace(
            namespace="test",
            config="conf/agent.yml",
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist=None,
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_web_app") as mock_create,
            patch("datus.cli.web.chatbot.uvicorn") as mock_uvicorn,
            patch("datus.cli.web.chatbot._schedule_browser_open"),
            patch("datus.cli.web.config_manager.get_home_from_config", return_value="~/.datus"),
            patch("datus.utils.path_manager.set_current_path_manager"),
        ):
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            mock_uvicorn.run.return_value = None

            run_web_interface(args)

            mock_create.assert_called_once_with(args)
            mock_uvicorn.run.assert_called_once_with(
                mock_app,
                host="localhost",
                port=8501,
                log_level="info",
            )

    def test_debug_mode_log_level(self):
        from datus.cli.web.chatbot import run_web_interface

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=True,
            subagent="",
            chatbot_dist=None,
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_web_app"),
            patch("datus.cli.web.chatbot.uvicorn") as mock_uvicorn,
            patch("datus.cli.web.chatbot._schedule_browser_open"),
            patch("datus.cli.web.config_manager.get_home_from_config", return_value="~/.datus"),
            patch("datus.utils.path_manager.set_current_path_manager"),
        ):
            mock_uvicorn.run.return_value = None
            run_web_interface(args)

            assert mock_uvicorn.run.call_args[1]["log_level"] == "debug"

    def test_keyboard_interrupt_handled(self):
        from datus.cli.web.chatbot import run_web_interface

        args = argparse.Namespace(
            namespace="test",
            config=None,
            host="localhost",
            port=8501,
            debug=False,
            subagent="",
            chatbot_dist=None,
            session_scope=None,
        )

        with (
            patch("datus.cli.web.chatbot.create_web_app"),
            patch("datus.cli.web.chatbot.uvicorn") as mock_uvicorn,
            patch("datus.cli.web.chatbot._schedule_browser_open"),
            patch("datus.cli.web.config_manager.get_home_from_config", return_value="~/.datus"),
            patch("datus.utils.path_manager.set_current_path_manager"),
        ):
            mock_uvicorn.run.side_effect = KeyboardInterrupt
            # Should not raise
            run_web_interface(args)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Template file existence
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestTemplateFile:
    """Verify the HTML template file exists and is valid."""

    def test_template_exists(self):
        from datus.cli.web.chatbot import _TEMPLATES_DIR

        template_path = os.path.join(_TEMPLATES_DIR, "index.html")
        assert os.path.isfile(template_path)

    def test_template_contains_chatbot_init(self):
        from datus.cli.web.chatbot import _TEMPLATES_DIR

        template_path = os.path.join(_TEMPLATES_DIR, "index.html")
        with open(template_path, encoding="utf-8") as f:
            content = f.read()
        assert "DatusChatbot.initChatbot" in content
        assert "chatbot-root" in content
        assert "{{ chatbot_js }}" in content
        assert "{{ chatbot_css }}" in content
        assert "{{ react_js }}" in content
        assert "{{ react_dom_js }}" in content
        assert "{{ request_origin_json }}" in content
        assert "{{ user_name_json }}" in content


# ═══════════════════════════════════════════════════════════════════════════
# _schedule_browser_open
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.ci
class TestScheduleBrowserOpen:
    """Ensure the helper kicks off a daemon thread that opens the URL after a delay."""

    def test_delays_then_calls_webbrowser_open(self):
        from datus.cli.web.chatbot import _schedule_browser_open

        with (
            patch("datus.cli.web.chatbot.webbrowser") as mock_webbrowser,
            patch("time.sleep") as mock_sleep,
            patch("threading.Thread") as mock_thread,
        ):
            _schedule_browser_open("http://example.test:1234", delay=0.25)

            mock_thread.assert_called_once()
            kwargs = mock_thread.call_args.kwargs
            assert kwargs["daemon"] is True
            inner = kwargs["target"]

            mock_thread.return_value.start.assert_called_once()

            # Invoke the inner thread target directly to cover the sleep+open path.
            inner()
            mock_sleep.assert_called_once_with(0.25)
            mock_webbrowser.open.assert_called_once_with("http://example.test:1234")
