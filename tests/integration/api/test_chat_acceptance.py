"""Chat HTTP lifecycle with real service, tasks, sessions, SDK and local LLM HTTP."""

import asyncio
import json
import threading

import pytest

from tests.helpers.product_acceptance import ModelServer, answer, isolated_connectors, product_config, tool_call
from tests.integration.api.test_api_chat import _cleanup_task, _start_task, find_event, parse_sse_body
from tests.integration.api.test_api_chat import chat_client as chat_client  # noqa: F401
from tests.integration.api.test_api_chat import chat_datus_service as chat_datus_service  # noqa: F401

pytestmark = pytest.mark.acceptance


@pytest.fixture(scope="module")
def model_server():
    with ModelServer().running() as server:
        yield server


@pytest.fixture(scope="module")
def chat_agent_config(tmp_path_factory, model_server):
    from datus.utils.path_manager import reset_path_manager

    reset_path_manager()
    config = product_config(tmp_path_factory.mktemp("chat_acceptance"), model_server.url)
    config.current_datasource = "sales"
    with isolated_connectors():
        yield config
    reset_path_manager()


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_http_sql_multiturn_history_and_delete(chat_client, model_server):
    model_server.enqueue(
        tool_call("execute_sql", {"sql": "SELECT SUM(amount) total FROM orders"}), answer("Total: 350.")
    )
    response = await chat_client.post(
        "/api/v1/chat/stream", json={"message": "What is the total order amount?", "interactive": False}
    )
    assert response.status_code == 200
    events = parse_sse_body(response.text)
    assert find_event(events, "error") is None, response.text
    assert find_event(events, "end") is not None
    sid = find_event(events, "session")["data"]["session_id"]
    assert "350" in json.dumps(model_server.requests[-1]["messages"])

    model_server.enqueue(answer("The previous total is 350 USD."))
    followup = await chat_client.post(
        "/api/v1/chat/stream", json={"message": "What was the previous total?", "session_id": sid, "interactive": False}
    )
    assert find_event(parse_sse_body(followup.text), "end") is not None, followup.text
    assert "What is the total order amount?" in json.dumps(model_server.requests[-1]["messages"])
    history = (await chat_client.get("/api/v1/chat/history", params={"session_id": sid})).json()
    assert "350" in json.dumps(history), history
    sessions = (await chat_client.get("/api/v1/chat/sessions")).json()
    assert sid in json.dumps(sessions)
    deleted = await chat_client.delete(f"/api/v1/chat/sessions/{sid}")
    assert deleted.json()["success"], deleted.text
    assert sid not in json.dumps((await chat_client.get("/api/v1/chat/sessions")).json())


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_history_is_isolated_by_user(chat_client, model_server):
    model_server.enqueue(answer("A private answer for the owner."))
    response = await chat_client.post(
        "/api/v1/chat/stream",
        headers={"X-Datus-User-Id": "owner"},
        json={"message": "Remember my private question.", "interactive": False},
    )
    events = parse_sse_body(response.text)
    assert find_event(events, "error") is None, response.text
    sid = find_event(events, "session")["data"]["session_id"]
    peer = await chat_client.get(
        "/api/v1/chat/history", headers={"X-Datus-User-Id": "peer"}, params={"session_id": sid}
    )
    assert peer.status_code == 200, peer.text
    assert peer.json()["success"] is True
    assert peer.json()["data"]["messages"] == []
    assert "private answer" not in peer.text
    owner = await chat_client.get(
        "/api/v1/chat/history", headers={"X-Datus-User-Id": "owner"}, params={"session_id": sid}
    )
    assert owner.status_code == 200, owner.text
    assert owner.json()["success"] is True
    assert "private answer" in owner.text
    await chat_client.delete(f"/api/v1/chat/sessions/{sid}", headers={"X-Datus-User-Id": "owner"})


@pytest.mark.asyncio(loop_scope="module")
async def test_stop_releases_session_for_the_next_request(chat_client, chat_datus_service, model_server):
    entered, release = threading.Event(), threading.Event()

    def slow_response(request):
        entered.set()
        assert release.wait(15), "The test must release the remote response"
        return answer("Stopped response.")

    sid = "acceptance_stop"
    model_server.enqueue(slow_response)
    task = await _start_task(chat_datus_service, "Wait for a slow response.", sid)
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        stopped = await chat_client.post("/api/v1/chat/stop", json={"session_id": sid})
        assert stopped.json()["success"], stopped.text
        await _cleanup_task(chat_datus_service, task)
    finally:
        release.set()
        await _cleanup_task(chat_datus_service, task)
    model_server.enqueue(answer("The session is usable again."))
    response = await chat_client.post(
        "/api/v1/chat/stream", json={"message": "Continue after stopping.", "session_id": sid, "interactive": False}
    )
    events = parse_sse_body(response.text)
    assert find_event(events, "error") is None, response.text
    assert find_event(events, "end") is not None
    await chat_client.delete(f"/api/v1/chat/sessions/{sid}")
