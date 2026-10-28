"""Small product fixtures: real runtime and storage, scripted remote responses."""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from datus.configuration.agent_config import AgentConfig


@contextmanager
def local_telemetry():
    """Keep tracing active without uploading test conversations."""
    from agents.tracing import get_trace_provider

    with patch("agents.tracing.processors.BackendSpanExporter.export"):
        try:
            yield
        finally:
            # The SDK batches traces on a background thread. Drain the queue
            # before restoring the exporter so later tests cannot upload it.
            get_trace_provider().force_flush()


@contextmanager
def isolated_connectors():
    """Give each temporary CLI project its own production connector cache."""
    from datus.tools.db_tools import db_manager

    # CLI caches are process-wide and keyed by datasource names. Each test
    # represents a fresh CLI invocation, even when its datasource is "sales".
    with patch.object(db_manager, "_cli_cache", {}), patch.object(db_manager, "_factory", None):
        try:
            yield
        finally:
            for manager in db_manager._cli_cache.values():
                manager.close()


def answer(text: str) -> dict:
    return {"role": "assistant", "content": text}


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


class ModelServer:
    """An OpenAI-compatible HTTP endpoint; it never replaces the SDK/Runner."""

    def __init__(self):
        self.replies = deque()
        self.requests = []
        self.errors = []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    with owner.lock:
                        owner.requests.append(copy.deepcopy(request))
                        if not owner.replies:
                            raise AssertionError("Unexpected model request: script exhausted")
                        reply = owner.replies.popleft()
                    message = reply(request) if callable(reply) else reply
                    response = {
                        "id": f"chatcmpl-{len(owner.requests)}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": request["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                    }
                    if request.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        delta = copy.deepcopy(message)
                        for index, call in enumerate(delta.get("tool_calls", [])):
                            call["index"] = index
                        for content, finish in ((delta, None), ({}, response["choices"][0]["finish_reason"])):
                            chunk = {
                                **response,
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": content, "finish_reason": finish}],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    else:
                        data = json.dumps(response).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    owner.errors.append(str(exc))
                    self.send_error(500, "Invalid test model request")

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.http.server_port}/v1"

    def enqueue(self, *replies):
        self.replies.extend(replies)

    @contextmanager
    def running(self):
        self.thread.start()
        try:
            with local_telemetry():
                yield self
            assert not self.errors, self.errors
            assert not self.replies, "Expected model requests did not execute"
        finally:
            self.http.shutdown()
            self.http.server_close()
            self.thread.join(timeout=5)


def product_config(root: Path, base_url: str, *, project: str = "core", live_model: dict | None = None) -> AgentConfig:
    """An independent business oracle: orders total 350, East 150, West 200."""
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "workspace"
    workspace.mkdir(exist_ok=True)
    database = root / "sales.sqlite"
    with sqlite3.connect(database) as db:
        db.executescript(
            "CREATE TABLE IF NOT EXISTS customers(id INTEGER PRIMARY KEY, region TEXT);"
            "CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER);"
            "DELETE FROM orders; DELETE FROM customers;"
            "INSERT INTO customers VALUES(1,'East'),(2,'West');"
            "INSERT INTO orders VALUES(1,1,100),(2,2,200),(3,1,50);"
        )
    return AgentConfig(
        nodes={},
        home=str(root / "home"),
        project_root=str(workspace),
        project_name=project,
        target="product",
        models={
            "product": live_model
            or {
                "type": "openai",
                "model": "core-test-model",
                "api_key": "test",
                "base_url": base_url,
                "context_length": 262144,
                "max_tokens": 2048,
            }
        },
        services={"datasources": {"sales": {"type": "sqlite", "uri": str(database), "default": True}}},
        agentic_nodes={
            "chat": {"system_prompt": "chat", "max_turns": 12},
            "gen_sql": {"system_prompt": "gen_sql", "tools": "db_tools.*", "max_turns": 12},
        },
    )
