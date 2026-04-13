# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Web Chatbot server for Datus Agent.

Serves a React-based chatbot frontend (``@datus/web-chatbot`` UMD bundle)
backed by the standard Datus Agent API routes.  Replaces the former
Streamlit-based implementation with a lightweight FastAPI static-file server.
"""

import argparse
import json
import os
import webbrowser

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from datus.api.service import create_app
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# CDN URLs used when --chatbot-dist is NOT provided (production mode).
_CDN_REACT_JS = "https://unpkg.com/react@18/umd/react.production.min.js"
_CDN_REACT_DOM_JS = "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"
_CDN_CHATBOT_CSS = "https://unpkg.com/@datus/web-chatbot/dist/datus-chatbot.css"
_CDN_CHATBOT_JS = "https://unpkg.com/@datus/web-chatbot/dist/datus-chatbot.umd.js"

# Dev URLs used when --chatbot-dist IS provided (local development mode).
_DEV_REACT_JS = "https://unpkg.com/react@18/umd/react.development.js"
_DEV_REACT_DOM_JS = "https://unpkg.com/react-dom@18/umd/react-dom.development.js"
_DEV_CHATBOT_CSS = "/chatbot-assets/datus-chatbot.css"
_DEV_CHATBOT_JS = "/chatbot-assets/datus-chatbot.umd.js"


def _build_agent_args(args: argparse.Namespace) -> argparse.Namespace:
    """Bridge CLI ``datus web`` arguments to the shape expected by
    ``create_app`` / ``DatusAPIService``.
    """
    agent_args = argparse.Namespace(
        namespace=args.namespace,
        config=getattr(args, "config", None),
        debug=getattr(args, "debug", False),
        # Fields expected by DatusAPIService but not present in CLI args
        max_steps=20,
        workflow="chat_agentic",
        load_cp=None,
        source="web",
        interactive=True,
        output_dir=getattr(args, "output_dir", "./output"),
        log_level="DEBUG" if getattr(args, "debug", False) else "INFO",
    )
    return agent_args


def _read_template() -> str:
    """Read the HTML template from disk (once)."""
    template_path = os.path.join(_TEMPLATES_DIR, "index.html")
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def create_web_app(args: argparse.Namespace) -> FastAPI:
    """Create the FastAPI application that serves both the API and the chatbot
    frontend.

    Parameters
    ----------
    args:
        Parsed CLI arguments from ``datus web``.
    """
    agent_args = _build_agent_args(args)
    app = create_app(agent_args)

    # ── Remove the default JSON root route registered by create_app()
    #    so we can replace it with the chatbot HTML page.
    app.routes[:] = [r for r in app.routes if not (hasattr(r, "path") and r.path == "/" and hasattr(r, "methods"))]

    # ── Resolve asset mode: local dist vs CDN ──────────────────────
    chatbot_dist = getattr(args, "chatbot_dist", None)
    use_local = False

    if chatbot_dist:
        chatbot_dist = os.path.abspath(os.path.expanduser(chatbot_dist))
        required_assets = [
            os.path.join(chatbot_dist, "datus-chatbot.css"),
            os.path.join(chatbot_dist, "datus-chatbot.umd.js"),
        ]
        if not os.path.isdir(chatbot_dist) or not all(os.path.isfile(p) for p in required_assets):
            logger.warning(f"Chatbot dist directory missing or incomplete: {chatbot_dist}. Falling back to CDN.")
        else:
            app.mount(
                "/chatbot-assets",
                StaticFiles(directory=chatbot_dist),
                name="chatbot-assets",
            )
            use_local = True
            logger.info(f"Dev mode: serving chatbot assets from {chatbot_dist}")

    if not use_local:
        logger.info("Production mode: loading chatbot assets from CDN")

    # ── Pick asset URLs based on mode ──────────────────────────────
    if use_local:
        react_js, react_dom_js = _DEV_REACT_JS, _DEV_REACT_DOM_JS
        chatbot_css, chatbot_js = _DEV_CHATBOT_CSS, _DEV_CHATBOT_JS
    else:
        react_js, react_dom_js = _CDN_REACT_JS, _CDN_REACT_DOM_JS
        chatbot_css, chatbot_js = _CDN_CHATBOT_CSS, _CDN_CHATBOT_JS

    # ── Render the HTML template ───────────────────────────────────
    html_template = _read_template()
    user_name = getattr(args, "user_name", None) or os.getenv("USER", "User")

    # Pre-render the static parts (asset URLs); dynamic parts (origin, user)
    # are rendered per-request so reverse proxies and alternate hostnames work.
    static_html = (
        html_template.replace("{{ react_js }}", react_js)
        .replace("{{ react_dom_js }}", react_dom_js)
        .replace("{{ chatbot_css }}", chatbot_css)
        .replace("{{ chatbot_js }}", chatbot_js)
    )

    # Override the default JSON root endpoint with the chatbot page
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def chatbot_page(request: Request) -> HTMLResponse:
        # Derive requestOrigin from the actual request so it works behind
        # reverse proxies, with --host 0.0.0.0, or alternate hostnames.
        # json.dumps escapes quotes/special chars to prevent XSS.
        rendered = static_html.replace(
            "{{ request_origin_json }}", json.dumps(str(request.base_url).rstrip("/"))
        ).replace("{{ user_name_json }}", json.dumps(user_name))
        return HTMLResponse(content=rendered)

    return app


def run_web_interface(args: argparse.Namespace) -> None:
    """Entry point called by ``datus web``.

    Creates the FastAPI app and starts uvicorn.
    """
    from datus.cli.web.config_manager import get_home_from_config
    from datus.utils.path_manager import set_current_path_manager

    config_path = getattr(args, "config", None) or "conf/agent.yml"
    set_current_path_manager(get_home_from_config(config_path))

    host = getattr(args, "host", "localhost")
    port = getattr(args, "port", 8501)
    url = f"http://{host}:{port}"

    if getattr(args, "subagent", ""):
        url += f"/?subagent={args.subagent}"

    logger.info("Starting Datus Web Interface...")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Server URL: {url}")

    app = create_web_app(args)

    # Open the browser after a short delay
    def _open_browser():
        import threading
        import time

        def _open():
            time.sleep(1.5)
            webbrowser.open(url)

        t = threading.Thread(target=_open, daemon=True)
        t.start()

    _open_browser()

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="debug" if getattr(args, "debug", False) else "info",
        )
    except KeyboardInterrupt:
        logger.info("Web server stopped")
