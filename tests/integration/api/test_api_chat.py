"""
Integration tests for Chat API endpoints with real LLM interaction.

These tests exercise the full streaming chat lifecycle against the
california_schools SQLite database with a real LLM backend (DeepSeek).

Scenario design:

  C1: Basic Chat       — send a simple factual question, parse SSE stream,
                         verify session/message/end events and SQL correctness
  C2: Resume (Reconnect) — start a chat, disconnect mid-stream, resume from
                           a cursor, verify no events are lost
  C3: Session Persistence — after a completed chat, list sessions, read history,
                            start a new turn on the same session
  C4: Subagent Routing   — route to a custom subagent (chatbot from agent.yml),
                           verify the conversation uses the correct node
  C5: Stop Mid-Stream    — start a long chat, stop before it finishes,
                           verify session is stopped gracefully
  C6: Edge Cases         — empty message, very short prompt, session lifecycle

Prerequisites (auto-skipped if missing):
  - DEEPSEEK_API_KEY env var
  - california_schools.sqlite at ~/.datus/benchmark/bird/dev_20240627/dev_databases/california_schools/
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

TESTS_ROOT = Path(__file__).resolve().parent.parent.parent
CONF_DIR = TESTS_ROOT / "conf"
SQLITE_DB = (
    Path.home()
    / ".datus"
    / "benchmark"
    / "bird"
    / "dev_20240627"
    / "dev_databases"
    / "california_schools"
    / "california_schools.sqlite"
)

# Skip entire module if LLM prerequisites are missing
pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
    pytest.mark.skipif(not SQLITE_DB.exists(), reason=f"california_schools.sqlite not found at {SQLITE_DB}"),
]


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


def parse_sse_body(body: str) -> list[dict]:
    """Parse raw SSE text into a list of {id, event, data} dicts."""
    events = []
    current = {}
    for line in body.split("\n"):
        if line.startswith("id: "):
            current["id"] = int(line[4:])
        elif line.startswith("event: "):
            current["event"] = line[7:]
        elif line.startswith("data: "):
            raw = line[6:]
            try:
                current["data"] = json.loads(raw)
            except json.JSONDecodeError:
                current["data"] = raw
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def find_events(events: list[dict], event_type: str) -> list[dict]:
    """Filter SSE events by event type."""
    return [e for e in events if e.get("event") == event_type]


def find_event(events: list[dict], event_type: str) -> Optional[dict]:
    """Find the first event of a given type."""
    matches = find_events(events, event_type)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _get_service_module():
    return sys.modules["datus.api.service"]


@pytest.fixture(scope="module")
def chat_agent_config(tmp_path_factory):
    """Load AgentConfig with bird_school namespace for chat tests."""
    src = CONF_DIR / "agent.yml"
    tmp_dir = tmp_path_factory.mktemp("chat_api_conf")
    tmp_cfg = tmp_dir / "agent.yml"
    shutil.copy2(src, tmp_cfg)

    from datus.configuration.agent_config_loader import load_agent_config

    config = load_agent_config(
        config=str(tmp_cfg),
        namespace="bird_school",
        reload=True,
        force=True,
        yes=True,
    )
    return config


@pytest.fixture(scope="module")
def chat_datus_service(chat_agent_config):
    """Create a real DatusService backed by california_schools."""
    from datus.api.services.datus_service import DatusService

    return DatusService(agent_config=chat_agent_config, project_id="chat_integration_test")


@pytest_asyncio.fixture(scope="module")
async def chat_client(chat_agent_config, chat_datus_service):
    """AsyncClient wired to the full FastAPI app with real DatusService."""
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

    svc_mod = _get_service_module()
    mock_service = DatusAPIService(agent_args)
    original_service = svc_mod.service
    svc_mod.service = mock_service

    namespace = "bird_school"
    auth_provider = NoAuthProvider(namespace=namespace)
    cache = DatusServiceCache(max_size=4)
    init_deps(auth_provider, cache, namespace=namespace)

    async def _factory():
        return chat_datus_service

    await cache.get_or_create(namespace, _factory)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            timeout=120.0,
        ) as client:
            yield client
    finally:
        svc_mod.service = original_service
        await cache.shutdown()


# ---------------------------------------------------------------------------
# C1: Basic Chat — factual question → SSE stream → SQL
# ---------------------------------------------------------------------------


class TestC1BasicChat:
    """C1: Send a simple factual question about california_schools,
    verify the full SSE lifecycle: message → session → thinking/tool → end.

    Prompt design: ask a concrete COUNT query that the LLM can answer
    in a single turn with a short SQL, minimizing token consumption.
    """

    @pytest.mark.asyncio
    async def test_c1_01_stream_count_query(self, chat_client):
        """C1-01: Ask 'How many schools are there?' — expect SQL with COUNT.

        Verifies:
        - SSE stream returns 200 with text/event-stream content-type
        - First event is a 'message' (processing notification)
        - A 'session' event provides session_id
        - Stream terminates with an 'end' event
        - At least one message event contains content
        """
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "How many schools are there in total?"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_body(resp.text)
        assert len(events) >= 3, f"Expected at least 3 SSE events, got {len(events)}"

        # Session event should provide session_id
        session_ev = find_event(events, "session")
        assert session_ev is not None, "Missing 'session' event"
        session_id = session_ev["data"]["session_id"]
        assert session_id, "session_id should not be empty"

        # End event should be present
        end_ev = find_event(events, "end")
        assert end_ev is not None, "Missing 'end' event — stream did not terminate properly"
        assert end_ev["data"]["action_count"] > 0, "At least one action should have been performed"
        assert end_ev["data"]["duration"] > 0, "Duration should be positive"

        # Message events should have content
        message_events = find_events(events, "message")
        assert len(message_events) >= 1, "At least one 'message' event expected"

        # Check that some event contains SQL-like content (tool call or markdown)
        all_data = json.dumps([e.get("data", {}) for e in events])
        has_sql_hint = any(kw in all_data.lower() for kw in ["select", "count", "schools"])
        assert has_sql_hint, "Response should contain SQL-related content about schools"

        # Store session_id for subsequent tests
        self.__class__._session_id = session_id

    @pytest.mark.asyncio
    async def test_c1_02_session_appears_in_list(self, chat_client):
        """C1-02: After a completed chat, the session should appear in session list."""
        session_id = getattr(self.__class__, "_session_id", None)
        if not session_id:
            pytest.skip("No session_id from previous test")

        resp = await chat_client.get("/api/v1/chat/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

        session_ids = [s["session_id"] for s in body["data"].get("sessions", [])]
        assert session_id in session_ids, f"Session {session_id} should appear in session list"

    @pytest.mark.asyncio
    async def test_c1_03_history_has_user_and_assistant(self, chat_client):
        """C1-03: Chat history should contain both user and assistant messages."""
        session_id = getattr(self.__class__, "_session_id", None)
        if not session_id:
            pytest.skip("No session_id from previous test")

        resp = await chat_client.get(
            "/api/v1/chat/history",
            params={"session_id": session_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

        messages = body["data"].get("messages", [])
        assert len(messages) >= 2, "History should have at least user + assistant messages"

        roles = [m["role"] for m in messages]
        assert "user" in roles, "History should contain user message"
        assert "assistant" in roles, "History should contain assistant response"

    @pytest.mark.asyncio
    async def test_c1_04_delete_session(self, chat_client):
        """C1-04: Delete the session and verify it no longer appears."""
        session_id = getattr(self.__class__, "_session_id", None)
        if not session_id:
            pytest.skip("No session_id from previous test")

        resp = await chat_client.delete(f"/api/v1/chat/sessions/{session_id}")
        assert resp.status_code == 200

        # Verify it's gone
        resp = await chat_client.get("/api/v1/chat/sessions")
        session_ids = [s["session_id"] for s in resp.json()["data"].get("sessions", [])]
        assert session_id not in session_ids, "Deleted session should not appear in list"


# ---------------------------------------------------------------------------
# C2: Resume (Reconnect) — disconnect mid-stream, resume from cursor
# ---------------------------------------------------------------------------


class TestC2ResumeChat:
    """C2: Simulate network interruption by consuming only a few events,
    then reconnect via /chat/resume with a cursor to get remaining events.

    Uses the DatusService.task_manager directly to start a task and then
    exercises the resume endpoint while the task is still running.
    """

    @pytest.mark.asyncio
    async def test_c2_01_resume_from_cursor(self, chat_client, chat_datus_service):
        """C2-01: Start chat via task_manager, resume via REST endpoint.

        Steps:
        1. Start a chat task directly through task_manager
        2. Wait briefly for the first few events to accumulate
        3. Call POST /chat/resume with from_event_id=0 to consume events
        4. Verify we receive session + message events

        Prompt: ultra-short factual query to keep execution fast.
        """
        from datus.api.models.cli_models import StreamChatInput

        task_manager = chat_datus_service.task_manager
        agent_config = chat_datus_service.agent_config

        request = StreamChatInput(
            message="What is the total number of charter schools?",
            session_id="resume_test_session",
        )

        # Start background task
        task = await task_manager.start_chat(agent_config, request)
        session_id = task.session_id

        # Wait a bit for events to accumulate
        for _ in range(60):
            await asyncio.sleep(1)
            if len(task.events) >= 2 or task.status != "running":
                break

        try:
            # Resume via REST endpoint
            resp = await chat_client.post(
                "/api/v1/chat/resume",
                json={"session_id": session_id, "from_event_id": 0},
            )
            assert resp.status_code == 200

            # If task completed, we get all events; if still running, we get partial
            if "text/event-stream" in resp.headers.get("content-type", ""):
                events = parse_sse_body(resp.text)
                assert len(events) >= 1, "Resume should return at least one event"

                # Verify continuity: event IDs should be sequential from cursor
                event_ids = [e["id"] for e in events if "id" in e and e["id"] >= 0]
                if len(event_ids) >= 2:
                    assert event_ids[0] == 0, "First event should have ID 0 when resuming from cursor 0"
            else:
                # Task already completed — resume returns JSON error
                body = resp.json()
                assert "TASK_NOT_FOUND" in body.get("errorCode", "")

        finally:
            # Ensure cleanup
            await task_manager.stop_task(session_id)
            # Wait for task to finish
            if task.asyncio_task and not task.asyncio_task.done():
                try:
                    await asyncio.wait_for(task.asyncio_task, timeout=30)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass


# ---------------------------------------------------------------------------
# C3: Session Persistence — multi-turn conversation on the same session
# ---------------------------------------------------------------------------


class TestC3MultiTurnSession:
    """C3: Send two consecutive messages to the same session, verifying
    the second turn has access to conversation context from the first.

    Turn 1: Ask a factual question about a specific table.
    Turn 2: Ask a follow-up that requires context from Turn 1.
    """

    @pytest.mark.asyncio
    async def test_c3_01_two_turn_conversation(self, chat_client):
        """C3-01: Two-turn conversation maintains context.

        Turn 1: 'What columns does the frpm table have?'
        Turn 2: 'Which of those columns stores the county name?'
        The second turn should reference CDSCode or county without
        re-explaining what 'frpm' is.
        """
        # --- Turn 1 ---
        resp1 = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "What columns does the frpm table have?"},
        )
        assert resp1.status_code == 200
        events1 = parse_sse_body(resp1.text)

        session_ev = find_event(events1, "session")
        assert session_ev is not None
        session_id = session_ev["data"]["session_id"]

        end_ev1 = find_event(events1, "end")
        assert end_ev1 is not None, "Turn 1 should complete"

        # --- Turn 2: same session_id ---
        resp2 = await chat_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Which of those columns stores the county name?",
                "session_id": session_id,
            },
        )
        assert resp2.status_code == 200
        events2 = parse_sse_body(resp2.text)

        end_ev2 = find_event(events2, "end")
        assert end_ev2 is not None, "Turn 2 should complete"

        # Turn 2 response should mention county-related column
        all_data2 = json.dumps([e.get("data", {}) for e in events2]).lower()
        county_mentioned = any(kw in all_data2 for kw in ["county", "county name", "county_name"])
        assert county_mentioned, "Turn 2 should mention county column from context"

        # --- Verify history has both turns ---
        resp_hist = await chat_client.get(
            "/api/v1/chat/history",
            params={"session_id": session_id},
        )
        assert resp_hist.status_code == 200
        messages = resp_hist.json()["data"].get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 2, "History should contain both user turns"

        # Cleanup
        self.__class__._session_id = session_id

    @pytest.mark.asyncio
    async def test_c3_02_cleanup(self, chat_client):
        """C3-02: Clean up test session."""
        session_id = getattr(self.__class__, "_session_id", None)
        if session_id:
            await chat_client.delete(f"/api/v1/chat/sessions/{session_id}")


# ---------------------------------------------------------------------------
# C4: Subagent Routing — chatbot node from agent.yml
# ---------------------------------------------------------------------------


class TestC4SubagentChat:
    """C4: Route conversation to the 'chatbot' subagent defined in agent.yml.

    The chatbot node has different tools and system prompt compared to
    the default chat node. We verify it initializes and responds.
    """

    @pytest.mark.asyncio
    async def test_c4_01_chatbot_subagent_responds(self, chat_client):
        """C4-01: Stream chat routed to chatbot subagent.

        The chatbot subagent is defined in agent.yml with gen_sql prompt
        and db_tools + context_search_tools. It should be able to answer
        a simple database question.
        """
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "How many rows are in the schools table?",
                "subagent_id": "chatbot",
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = parse_sse_body(resp.text)

        session_ev = find_event(events, "session")
        assert session_ev is not None, "Subagent chat should emit session event"

        end_ev = find_event(events, "end")
        assert end_ev is not None, "Subagent chat should complete"

        # Should have tool calls (db_tools)
        message_events = find_events(events, "message")
        assert len(message_events) >= 1

        session_id = session_ev["data"]["session_id"]

        # Cleanup
        await chat_client.delete(f"/api/v1/chat/sessions/{session_id}")

    @pytest.mark.asyncio
    async def test_c4_02_builtin_gen_sql_subagent(self, chat_client):
        """C4-02: Stream chat routed to builtin gen_sql subagent.

        gen_sql uses GenSQLAgenticNode which generates SQL directly
        without the full ChatAgenticNode orchestration.
        """
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Count schools by county, show top 3",
                "subagent_id": "gen_sql",
            },
        )
        assert resp.status_code == 200

        events = parse_sse_body(resp.text)

        # Check basic lifecycle events
        session_ev = find_event(events, "session")
        assert session_ev is not None

        end_or_error = find_event(events, "end") or find_event(events, "error")
        assert end_or_error is not None, "gen_sql subagent should complete or error"

        session_id = session_ev["data"]["session_id"]
        await chat_client.delete(f"/api/v1/chat/sessions/{session_id}")


# ---------------------------------------------------------------------------
# C5: Stop Mid-Stream — interrupt a running chat
# ---------------------------------------------------------------------------


class TestC5StopChat:
    """C5: Start a chat with a complex question that takes multiple turns,
    then stop it mid-stream and verify it terminates gracefully.
    """

    @pytest.mark.asyncio
    async def test_c5_01_stop_running_chat(self, chat_client, chat_datus_service):
        """C5-01: Start a slow chat and stop it before completion.

        Uses task_manager directly to start and then exercises
        POST /chat/stop while the task is running.
        """
        from datus.api.models.cli_models import StreamChatInput

        task_manager = chat_datus_service.task_manager
        agent_config = chat_datus_service.agent_config

        request = StreamChatInput(
            message="List all distinct counties, their school counts, and average enrollment for each county",
            session_id="stop_test_session",
        )

        task = await task_manager.start_chat(agent_config, request)

        # Wait for the task to start processing
        for _ in range(30):
            await asyncio.sleep(0.5)
            if len(task.events) >= 2 or task.status != "running":
                break

        if task.status == "running":
            # Stop via REST endpoint
            resp = await chat_client.post(
                "/api/v1/chat/stop",
                json={"session_id": "stop_test_session"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True, "Stop should succeed for a running task"

            # Wait for the task to actually stop
            for _ in range(20):
                await asyncio.sleep(0.5)
                if task.status != "running":
                    break

            assert task.status in ("cancelled", "completed", "error"), (
                f"Task should be stopped, got status: {task.status}"
            )
        else:
            # Task already finished — that's okay for a fast query
            pass

        # Ensure task is cleaned up
        if task.asyncio_task and not task.asyncio_task.done():
            task.asyncio_task.cancel()
            try:
                await asyncio.wait_for(task.asyncio_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass


# ---------------------------------------------------------------------------
# C6: Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestC6ChatEdgeCases:
    """C6: Boundary conditions for the chat API."""

    @pytest.mark.asyncio
    async def test_c6_01_explicit_session_id(self, chat_client):
        """C6-01: Client can provide their own session_id."""
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "How many schools are there?",
                "session_id": "custom_session_12345",
            },
        )
        assert resp.status_code == 200
        events = parse_sse_body(resp.text)

        session_ev = find_event(events, "session")
        assert session_ev is not None
        assert session_ev["data"]["session_id"] == "custom_session_12345"

        # Cleanup
        await chat_client.delete("/api/v1/chat/sessions/custom_session_12345")

    @pytest.mark.asyncio
    async def test_c6_02_resume_completed_task(self, chat_client):
        """C6-02: Resuming a completed (cleaned-up) task returns TASK_NOT_FOUND."""
        resp = await chat_client.post(
            "/api/v1/chat/resume",
            json={"session_id": "definitely_not_running_xyz"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["errorCode"] == "TASK_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_c6_03_stop_nonexistent_session(self, chat_client):
        """C6-03: Stopping a non-existent session returns clear error."""
        resp = await chat_client.post(
            "/api/v1/chat/stop",
            json={"session_id": "nonexistent_stop_xyz"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_c6_04_stream_invalid_subagent_returns_404(self, chat_client):
        """C6-04: Streaming to a non-existent subagent returns 404."""
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={
                "message": "test",
                "subagent_id": "totally_nonexistent_subagent_xyz",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_c6_05_interaction_no_active_task(self, chat_client):
        """C6-05: Submitting user interaction to non-existent task fails."""
        resp = await chat_client.post(
            "/api/v1/chat/user_interaction",
            json={
                "session_id": "nonexistent_session",
                "interaction_key": "key1",
                "input": ["option_a"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_c6_06_tool_result_no_active_task(self, chat_client):
        """C6-06: Submitting tool result to non-existent task fails."""
        resp = await chat_client.post(
            "/api/v1/chat/tool_result",
            json={
                "session_id": "nonexistent_session",
                "call_tool_id": "tool_1",
                "tool_result": {"success": 1, "result": "done"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_c6_07_sse_events_have_valid_structure(self, chat_client):
        """C6-07: Verify SSE events conform to the expected schema.

        Each event should have: id (int), event (str), data (json).
        The data.payload should contain message_id and role.
        """
        resp = await chat_client.post(
            "/api/v1/chat/stream",
            json={"message": "How many rows in the frpm table?"},
        )
        assert resp.status_code == 200

        events = parse_sse_body(resp.text)

        for ev in events:
            assert "id" in ev, f"Event missing 'id': {ev}"
            assert "event" in ev, f"Event missing 'event': {ev}"
            assert "data" in ev, f"Event missing 'data': {ev}"
            assert isinstance(ev["data"], dict), f"Event data should be dict: {ev['event']}"

        # Message events should have payload with message_id + role
        for ev in find_events(events, "message"):
            payload = ev["data"].get("payload", {})
            assert "message_id" in payload, "Message event payload should have message_id"
            assert "role" in payload, "Message event payload should have role"
            assert payload["role"] in ("user", "assistant"), f"Unexpected role: {payload['role']}"

        # Session event should have session_id
        session_ev = find_event(events, "session")
        if session_ev:
            assert "session_id" in session_ev["data"]

        # End event should have duration
        end_ev = find_event(events, "end")
        if end_ev:
            assert "duration" in end_ev["data"]
            assert "total_events" in end_ev["data"]

        # Cleanup
        if session_ev:
            await chat_client.delete(f"/api/v1/chat/sessions/{session_ev['data']['session_id']}")
