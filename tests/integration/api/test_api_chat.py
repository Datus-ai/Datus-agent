"""
Integration tests for Chat API endpoints with real LLM interaction.

Exercises the full streaming chat lifecycle against the california_schools
SQLite database with a real LLM backend. Each test targets a distinct
user-facing scenario; prompts are designed to be short and deterministic
so that total LLM round-trips stay minimal.

  Basic stream      — SSE lifecycle (message/session/end), session list, history, delete
  Resume            — start via task_manager, reconnect via /resume
  Multi-turn        — two turns on the same session, context preserved
  Subagent routing  — route to chatbot custom sub-agent
  Stop mid-stream   — interrupt a running task via /stop
  Source proxy       — source="web" proxies fs tools, tool_result resolves channel
  ask_user e2e      — LLM calls ask_user, frontend submits via /user_interaction
  Invalid subagent  — 404 for non-existent subagent
  Error paths       — stop/resume/interaction on non-existent sessions
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

TESTS_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_DIR = TESTS_ROOT / "conf"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def parse_sse_body(body: str) -> list[dict]:
    """Parse raw SSE text into a list of {id, event, data} dicts."""
    events, current = [], {}
    for line in body.split("\n"):
        if line.startswith("id: "):
            current["id"] = int(line[4:])
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current["data"] = line[6:]
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def find_events(events: list[dict], event_type: str) -> list[dict]:
    return [e for e in events if e.get("event") == event_type]


def find_event(events: list[dict], event_type: str) -> Optional[dict]:
    matches = find_events(events, event_type)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _svc_mod():
    return sys.modules["datus.api.service"]


@pytest.fixture(scope="module")
def chat_agent_config(tmp_path_factory):
    """Load AgentConfig with bird_school namespace."""
    src = CONF_DIR / "agent.yml"
    tmp_cfg = tmp_path_factory.mktemp("chat_api_conf") / "agent.yml"
    shutil.copy2(src, tmp_cfg)
    from datus.configuration.agent_config_loader import load_agent_config

    return load_agent_config(config=str(tmp_cfg), namespace="bird_school", reload=True, force=True, yes=True)


@pytest.fixture(scope="module")
def chat_datus_service(chat_agent_config):
    from datus.api.services.datus_service import DatusService

    return DatusService(agent_config=chat_agent_config, project_id="chat_integration_test")


@pytest_asyncio.fixture(scope="module")
async def chat_client(chat_agent_config, chat_datus_service):
    """AsyncClient wired to the full FastAPI app with real DatusService."""
    import datus.api.deps as deps_mod
    from datus.api.auth import NoAuthProvider
    from datus.api.deps import init_deps
    from datus.api.service import DatusAPIService, create_app
    from datus.api.services.datus_service_cache import DatusServiceCache

    agent_args = argparse.Namespace(
        namespace="bird_school",
        config="tests/conf/agent.yml",
        max_steps=20,
        workflow="fixed",
        load_cp=None,
        debug=False,
    )
    app = create_app(agent_args)

    mod = _svc_mod()
    saved = mod.service
    saved_deps = (
        deps_mod._auth_provider,
        deps_mod._service_cache,
        deps_mod._namespace,
        deps_mod._default_source,
        deps_mod._default_interactive,
        deps_mod._stream_thinking,
    )
    cache = DatusServiceCache(max_size=4)

    async def _factory():
        return chat_datus_service

    # Enter try/finally BEFORE mutating globals so failures during setup
    # (DatusAPIService, init_deps, cache.get_or_create) still unwind the
    # patched module state and shut the cache down cleanly.
    try:
        mod.service = DatusAPIService(agent_args)
        init_deps(NoAuthProvider(), cache, namespace="bird_school")
        await cache.get_or_create("default", _factory)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=120.0) as c:
            yield c
    finally:
        mod.service = saved
        (
            deps_mod._auth_provider,
            deps_mod._service_cache,
            deps_mod._namespace,
            deps_mod._default_source,
            deps_mod._default_interactive,
            deps_mod._stream_thinking,
        ) = saved_deps
        # Shut the cache down last so if it raises the globals are already restored.
        await cache.shutdown()


# ---------------------------------------------------------------------------
# Helper to start a task via task_manager and wait for node init
# ---------------------------------------------------------------------------


async def _start_task(svc, message, session_id, *, source=None):
    """Start a background chat task and wait for node creation."""
    from datus.api.models.cli_models import StreamChatInput

    req = StreamChatInput(message=message, session_id=session_id)
    if source:
        req.source = source
    task = await svc.task_manager.start_chat(svc.agent_config, req)
    for _ in range(120):
        await asyncio.sleep(0.5)
        if task.node is not None or task.status != "running":
            break
    return task


async def _cleanup_task(svc, task):
    """Stop and cancel a task, swallowing errors."""
    await svc.task_manager.stop_task(task.session_id)
    if task.asyncio_task and not task.asyncio_task.done():
        task.asyncio_task.cancel()
        try:
            await asyncio.wait_for(task.asyncio_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


# ---------------------------------------------------------------------------
# Tests — N9 series
# ---------------------------------------------------------------------------


@pytest.mark.nightly
class TestAPIChatN9:
    """N9: Chat API integration tests with real LLM."""

    # ------ Basic stream + session lifecycle ------

    @pytest.mark.asyncio
    async def test_stream_and_session_lifecycle(self, chat_client):
        """stream → verify SSE events → list sessions → history → delete."""
        resp = await chat_client.post("/api/v1/chat/stream", json={"message": "How many schools are there in total?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_body(resp.text)
        assert len(events) >= 3

        session_ev = find_event(events, "session")
        assert session_ev is not None
        sid = session_ev["data"]["session_id"]

        end_ev = find_event(events, "end")
        assert end_ev is not None
        assert end_ev["data"]["action_count"] > 0

        # session list
        body = (await chat_client.get("/api/v1/chat/sessions")).json()
        assert sid in [s["session_id"] for s in body["data"].get("sessions", [])]

        # history
        body = (await chat_client.get("/api/v1/chat/history", params={"session_id": sid})).json()
        assert body["success"] is True
        roles = [m["role"] for m in body["data"].get("messages", [])]
        assert "user" in roles and "assistant" in roles

        # delete — API returns success; session file may already be cleaned
        # by the node itself, so we only assert the API call succeeds.
        del_body = (await chat_client.delete(f"/api/v1/chat/sessions/{sid}")).json()
        assert del_body["success"] is True

    # ------ Resume (reconnect) ------

    @pytest.mark.asyncio
    async def test_resume_from_cursor(self, chat_client, chat_datus_service):
        """start task → wait for events → resume from cursor 0."""
        task = await _start_task(chat_datus_service, "How many charter schools are there?", "resume_sess")
        try:
            resp = await chat_client.post(
                "/api/v1/chat/resume",
                json={"session_id": task.session_id, "from_event_id": 0},
            )
            assert resp.status_code == 200
            if "text/event-stream" in resp.headers.get("content-type", ""):
                events = parse_sse_body(resp.text)
                assert len(events) >= 1
                ids = [e["id"] for e in events if e.get("id", -1) >= 0]
                # Resume must replay numbered events as a contiguous sequence
                # starting at id=0. Empty list matches list(range(0)) == [],
                # so the check is unconditional — no hidden skip branch.
                assert ids == list(range(len(ids))), f"resume events must be a contiguous 0..N sequence, got {ids}"
            else:
                assert resp.json().get("errorCode") == "TASK_NOT_FOUND"
        finally:
            await _cleanup_task(chat_datus_service, task)

    # ------ Multi-turn context ------

    @pytest.mark.asyncio
    async def test_multi_turn_context(self, chat_client):
        """two turns on the same session, second references first."""
        r1 = await chat_client.post("/api/v1/chat/stream", json={"message": "What columns does the frpm table have?"})
        events1 = parse_sse_body(r1.text)
        sid = find_event(events1, "session")["data"]["session_id"]
        assert find_event(events1, "end") is not None

        r2 = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "Which of those columns stores the county name?", "session_id": sid},
        )
        events2 = parse_sse_body(r2.text)
        assert find_event(events2, "end") is not None
        assert "county" in json.dumps([e.get("data", {}) for e in events2]).lower()

        # history has both turns
        body = (await chat_client.get("/api/v1/chat/history", params={"session_id": sid})).json()
        assert len([m for m in body["data"].get("messages", []) if m["role"] == "user"]) >= 2
        await chat_client.delete(f"/api/v1/chat/sessions/{sid}")

    # ------ Subagent routing (chatbot) ------

    @pytest.mark.asyncio
    async def test_chatbot_subagent(self, chat_client):
        """route to chatbot custom sub-agent, verify completion."""
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "How many rows are in the schools table?", "subagent_id": "chatbot"},
        )
        assert resp.status_code == 200
        events = parse_sse_body(resp.text)
        session_ev = find_event(events, "session")
        assert session_ev is not None
        assert find_event(events, "end") is not None
        await chat_client.delete(f"/api/v1/chat/sessions/{session_ev['data']['session_id']}")

    # ------ Stop mid-stream ------

    @pytest.mark.asyncio
    async def test_stop_running_chat(self, chat_client, chat_datus_service):
        """start a chat, stop it, verify task terminates."""
        task = await _start_task(
            chat_datus_service,
            "List all distinct counties, their school counts, and average enrollment",
            "stop_sess",
        )
        try:
            # If the task is still running when we arrive, send /stop and wait
            # for it to terminate. If it already finished on its own (fast
            # LLM / small workload), we skip the stop call — the terminal-
            # state assert below still fires, so the test never silently passes.
            stop_success: bool | None = None
            if task.status == "running":
                body = (await chat_client.post("/api/v1/chat/stop", json={"session_id": "stop_sess"})).json()
                stop_success = body["success"]
                for _ in range(20):
                    await asyncio.sleep(0.5)
                    if task.status != "running":
                        break
            # Unconditional terminal-state check — applies whether we stopped
            # the task or it finished on its own.
            assert task.status in ("cancelled", "completed", "error"), (
                f"task should reach a terminal state, got {task.status}"
            )
            # Only assert stop-endpoint success if we actually called it.
            assert stop_success in (None, True), f"chat/stop must report success when invoked, got {stop_success}"
        finally:
            await _cleanup_task(chat_datus_service, task)

    # ------ Source proxy + tool_result ------

    @pytest.mark.asyncio
    async def test_source_proxy_tool_result(self, chat_client, chat_datus_service, tmp_path):
        """source='web' proxies fs tools; execute via FilesystemFuncTool, submit result via REST."""
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        fs_tool = FilesystemFuncTool(root_path=str(tmp_path))

        task = await _start_task(
            chat_datus_service,
            "Create a new directory called 'test_output_dir' for me.",
            "proxy_sess",
            source="vscode",
        )
        try:
            if task.node is None:
                pytest.skip("Node not created in time")

            # Poll task events for call-tool events and respond until task completes
            responded_ids = set()
            deadline = asyncio.get_event_loop().time() + 120
            while asyncio.get_event_loop().time() < deadline and task.status == "running":
                await asyncio.sleep(1)
                for ev in task.events:
                    if ev.event != "message" or not hasattr(ev.data, "payload"):
                        continue
                    p = ev.data.payload
                    if not p or not p.content:
                        continue
                    for c in p.content:
                        if c.type != "call-tool" or not c.payload:
                            continue
                        call_tool_id = c.payload.get("callToolId")
                        if not call_tool_id or call_tool_id in responded_ids:
                            continue
                        responded_ids.add(call_tool_id)

                        tool_name = c.payload.get("toolName", "")
                        tool_params = c.payload.get("toolParams", {})

                        # Execute the tool locally via FilesystemFuncTool
                        handler = getattr(fs_tool, tool_name, None)
                        if handler:
                            result = handler(**tool_params)
                            tool_result = result.model_dump()
                        else:
                            tool_result = {"success": 0, "error": f"Unknown tool: {tool_name}", "result": None}

                        # Submit result via REST API
                        resp = await chat_client.post(
                            "/api/v1/chat/tool_result",
                            json={
                                "session_id": "proxy_sess",
                                "call_tool_id": call_tool_id,
                                "tool_result": tool_result,
                            },
                        )
                        assert resp.json()["success"] is True

            assert len(responded_ids) >= 1, "Expected at least one proxied tool call"
            assert task.status in ("completed", "error")
        finally:
            await _cleanup_task(chat_datus_service, task)

    # ------ ask_user interaction e2e ------

    @pytest.mark.asyncio
    async def test_ask_user_interaction(self, chat_client, chat_datus_service):
        """prompt instructs LLM to call ask_user, submit via REST, agent completes."""
        task = await _start_task(
            chat_datus_service,
            "I want to know about schools in a specific county. "
            "Use the ask_user tool to ask me which county I'm interested in. "
            "Provide these options: Los Angeles, San Francisco, San Diego. "
            "After I answer, count the schools in that county.",
            "askuser_sess",
        )
        try:
            # Poll for user-interaction event
            interaction_key = None
            deadline = asyncio.get_event_loop().time() + 90
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(1)
                for ev in task.events:
                    if ev.event != "message" or not hasattr(ev.data, "payload"):
                        continue
                    p = ev.data.payload
                    if not p or not p.content:
                        continue
                    for c in p.content:
                        if c.type == "user-interaction":
                            interaction_key = c.payload.get("interactionKey")
                            break
                    if interaction_key:
                        break
                if interaction_key or task.status != "running":
                    break

            if interaction_key is None:
                pytest.skip(f"LLM did not call ask_user (status={task.status}, events={len(task.events)})")

            # Submit choice
            body = (
                await chat_client.post(
                    "/api/v1/chat/user_interaction",
                    json={"session_id": "askuser_sess", "interaction_key": interaction_key, "input": ["Los Angeles"]},
                )
            ).json()
            assert body["success"] is True
            assert body["data"]["submitted"] is True

            # Wait for completion
            for _ in range(120):
                await asyncio.sleep(1)
                if task.status != "running":
                    break
            assert task.status in ("completed", "error")

            # Response should mention Los Angeles
            dump = json.dumps(
                [ev.data.model_dump() if hasattr(ev.data, "model_dump") else str(ev.data) for ev in task.events]
            ).lower()
            assert "los angeles" in dump
        finally:
            await _cleanup_task(chat_datus_service, task)

    # ------ Invalid subagent → 404 ------

    @pytest.mark.asyncio
    async def test_invalid_subagent_404(self, chat_client):
        """streaming to a non-existent subagent returns 404."""
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "test", "subagent_id": "nonexistent_xyz"},
        )
        assert resp.status_code == 404

    # ------ Error paths (no LLM call needed) ------

    @pytest.mark.asyncio
    async def test_error_paths(self, chat_client):
        """stop / resume / interaction / tool_result on non-existent sessions."""
        # stop
        body = (await chat_client.post("/api/v1/chat/stop", json={"session_id": "no_such"})).json()
        assert body["success"] is False

        # resume
        body = (await chat_client.post("/api/v1/chat/resume", json={"session_id": "no_such"})).json()
        assert body["success"] is False and body["errorCode"] == "TASK_NOT_FOUND"

        # user_interaction
        body = (
            await chat_client.post(
                "/api/v1/chat/user_interaction",
                json={"session_id": "no_such", "interaction_key": "k", "input": ["x"]},
            )
        ).json()
        assert body["success"] is False

        # tool_result
        body = (
            await chat_client.post(
                "/api/v1/chat/tool_result",
                json={"session_id": "no_such", "call_tool_id": "t", "tool_result": {"success": 1, "result": "x"}},
            )
        ).json()
        assert body["success"] is False
