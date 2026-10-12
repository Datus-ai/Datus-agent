"""Shared validation for local assets used by the Web UI browser regression suite."""

import os
from pathlib import Path


def local_chatbot_dist() -> Path:
    """Require actual chatbot assets rather than silently falling back to the CDN."""
    value = os.getenv("DATUS_TEST_CHATBOT_DIST")
    if not value:
        raise ValueError(
            "Set DATUS_TEST_CHATBOT_DIST to a directory containing @datus/web-chatbot dist assets and react.production.min.js before Web UI tests"
        )
    dist = Path(value).expanduser().resolve()
    missing = [
        name
        for name in ("datus-chatbot.css", "datus-chatbot.umd.js", "react.production.min.js")
        if not (dist / name).is_file()
    ]
    if missing:
        raise ValueError(f"Incomplete chatbot dist at {dist}: missing {', '.join(missing)}")
    return dist
