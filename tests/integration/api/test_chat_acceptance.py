"""Chat HTTP lifecycle with real service, tasks, sessions, SDK and local LLM HTTP."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

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


@pytest_asyncio.fixture
async def resource_project(tmp_path, model_server, monkeypatch):
    """Use the normal auth/config/cache boundary with isolated real project storage."""
    from functools import partial
    from types import SimpleNamespace

    from datus.api.auth import HeaderContextProvider
    from datus.api.services.chat_task_manager import ChatTaskManager
    from datus.api.services.datus_service import DatusService
    from datus.api.services.datus_service_cache import DatusServiceCache
    from tests.integration.api.test_api_chat import chat_http_client

    class Projects(HeaderContextProvider):
        def __init__(self, config):
            super().__init__()
            self.configs = {"default": config}

        async def authenticate(self, request):
            context = await super().authenticate(request)
            context.project_id = request.headers.get("X-Test-Project", "default")
            context.config = self.configs[context.project_id]
            return context

    with isolated_connectors():
        config = product_config(tmp_path / "default", model_server.url, project="resource_default")
        config.current_datasource = "sales"
        factory = partial(ChatTaskManager, completed_task_ttl=1.0)
        service = DatusService(config, "default", task_manager_factory=factory)
        expired = asyncio.Event()
        expire = service.task_manager._expire_completed

        def observe_expiry():
            expire()
            expired.set()

        monkeypatch.setattr(service.task_manager, "_expire_completed", observe_expiry)

        async def wait_for_expiry(session_ids):
            # Observe actual timer callbacks without issuing lookups or purges.
            async with asyncio.timeout(5):
                while any(sid in service.task_manager._completed_tasks for sid in session_ids):
                    await expired.wait()
                    expired.clear()

        cache = DatusServiceCache(max_size=1)
        projects = Projects(config)
        async with chat_http_client(
            config, service, service_cache=cache, auth_provider=projects, task_manager_factory=factory
        ) as client:
            yield SimpleNamespace(
                client=client,
                cache=cache,
                service=service,
                projects=projects,
                root=tmp_path,
                wait_for_expiry=wait_for_expiry,
            )


@asynccontextmanager
async def _blocked_http_turn(project, model_server, outcome, first_reply, *remaining_replies):
    """Pause a real model call while the caller exercises an explicit terminal path."""
    import weakref
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()

    def reply(request):
        entered.set()
        assert release.wait(10), "The test must release the model HTTP response"
        return first_reply

    model_server.enqueue(reply, *remaining_replies)
    sid = f"resource_{outcome}"
    request = asyncio.create_task(
        project.client.post(
            "/api/v1/chat/stream", json={"message": "Compute the total.", "session_id": sid, "interactive": False}
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task = project.service.task_manager.get_task(sid)
        assert task.status == "running"
        yield SimpleNamespace(
            project=project,
            client=project.client,
            manager=project.service.task_manager,
            sid=sid,
            request=request,
            task=task,
            release=release,
            node=weakref.ref(task.node),
            llm_session_id=task.node.session_id,
            db_tool=task.node.db_func_tool,
            db_manager=task.node.db_func_tool._db_manager,
        )
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


async def _assert_terminal_turn(turn, expected_status):
    await asyncio.wait_for(turn.task.asyncio_task, timeout=10)
    assert turn.task.status == expected_status, turn.task.error
    assert turn.task.node is None
    assert not turn.db_manager._leases, "Finished real tools must return their managed borrows"
    assert turn.db_tool._primary_connector is None
    assert not turn.db_tool._connector_cache
    assert not turn.db_tool._borrowed_connections
    turn.db_tool = None


async def _assert_replay_expires(turn):
    from types import SimpleNamespace

    resumed = await turn.client.post("/api/v1/chat/resume", json={"session_id": turn.sid, "from_event_id": 0})
    assert resumed.headers["content-type"].startswith("text/event-stream"), resumed.text
    events = parse_sse_body(resumed.text)
    assert find_event(events, "session")["data"]["llm_session_id"] == turn.llm_session_id
    # No lookup or HTTP request occurs while waiting for automatic expiry.
    await turn.project.wait_for_expiry([turn.sid])
    assert turn.sid not in turn.manager._completed_tasks
    expired = await turn.client.post("/api/v1/chat/resume", json={"session_id": turn.sid})
    assert expired.json()["errorCode"] == "TASK_NOT_FOUND"
    history = await turn.client.get("/api/v1/chat/history", params={"session_id": turn.sid})
    assert history.json()["success"], history.text
    return SimpleNamespace(replay=resumed.text, events=events, history=history.text)


async def _assert_successful_node_collected(turn):
    import gc

    from agents.tracing import get_trace_provider

    # Failed/cancelled tasks can be retained by external HTTP timeout handles.
    # Those scenarios assert native cleanup above without assuming immediate GC.
    await asyncio.to_thread(get_trace_provider().force_flush)
    await asyncio.sleep(0)
    gc.collect()
    assert turn.node() is None


@pytest.mark.asyncio
async def test_http_turn_resource_cleanup_preserves_resume_until_expiry_completed(resource_project, model_server):
    first = tool_call("execute_sql", {"sql": "SELECT SUM(amount) total FROM orders"})
    async with _blocked_http_turn(resource_project, model_server, "completed", first, answer("Total: 350.")) as turn:
        turn.release.set()
        response = await turn.request
        assert response.status_code == 200, response.text
        await _assert_terminal_turn(turn, "completed")
        replay = await _assert_replay_expires(turn)
        assert "350" in replay.replay
        assert find_event(replay.events, "end")["event"] == "end"
        assert "350" in replay.history
        await _assert_successful_node_collected(turn)


@pytest.mark.asyncio
async def test_http_turn_resource_cleanup_preserves_resume_until_expiry_error(resource_project, model_server):
    from tests.helpers.product_acceptance import ModelError

    first = ModelError(400, "injected resource acceptance failure")
    async with _blocked_http_turn(resource_project, model_server, "error", first) as turn:
        turn.release.set()
        response = await turn.request
        assert response.status_code == 200, response.text
        await _assert_terminal_turn(turn, "error")
        replay = await _assert_replay_expires(turn)
        assert "injected resource acceptance failure" in replay.replay


@pytest.mark.asyncio
async def test_http_turn_resource_cleanup_preserves_resume_until_expiry_cancelled(resource_project, model_server):
    first = tool_call("execute_sql", {"sql": "SELECT SUM(amount) total FROM orders"})
    async with _blocked_http_turn(resource_project, model_server, "cancelled", first) as turn:
        stopped = await turn.client.post("/api/v1/chat/stop", json={"session_id": turn.sid})
        assert stopped.json()["success"], stopped.text
        turn.release.set()
        response = await turn.request
        assert response.status_code == 200, response.text
        await _assert_terminal_turn(turn, "cancelled")
        await _assert_replay_expires(turn)


@pytest.mark.asyncio
async def test_http_turn_resource_cleanup_preserves_resume_until_expiry_disconnected(resource_project, model_server):
    first = tool_call("execute_sql", {"sql": "SELECT SUM(amount) total FROM orders"})
    async with _blocked_http_turn(resource_project, model_server, "disconnected", first, answer("Total: 350.")) as turn:
        turn.request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn.request
        assert turn.task.status == "running", "Disconnect must not cancel the server-owned turn"
        turn.release.set()
        await _assert_terminal_turn(turn, "completed")
        replay = await _assert_replay_expires(turn)
        assert "350" in replay.replay
        assert find_event(replay.events, "end")["event"] == "end"
        assert "350" in replay.history
        await _assert_successful_node_collected(turn)


@pytest.mark.asyncio
async def test_http_project_eviction_closes_native_connections_and_rebuilds(resource_project, model_server):
    """Real CLI and datasource initialization is reclaimed by the real service cache."""
    import ast
    import sqlite3

    client, cache = resource_project.client, resource_project.cache
    query = {"sql_query": "SELECT SUM(amount) total FROM orders", "result_format": "json"}
    result = await client.post("/api/v1/sql/execute", json=query)
    assert result.json()["success"], result.text
    assert ast.literal_eval(result.json()["data"]["sql_return"]) == [{"total": 350}]
    detail = await client.get("/api/v1/table/detail", params={"table": "orders"})
    assert detail.json()["success"], detail.text
    original = cache._cache["default"]
    managers = [original.cli.db_manager, original.datasource.db_manager]
    native = [manager.first_conn("sales").connection for manager in managers]
    assert all(tuple(connection.execute("SELECT 1").fetchone()) == (1,) for connection in native)
    resource_project.projects.configs["other"] = product_config(
        resource_project.root / "other", model_server.url, project="resource_other"
    )
    resource_project.projects.configs["other"].current_datasource = "sales"
    other = await client.post("/api/v1/sql/execute", headers={"X-Test-Project": "other"}, json=query)
    assert other.json()["success"], other.text
    await cache.drain()
    assert "default" not in cache._cache
    for manager, connection in zip(managers, native):
        assert all(not group for group in manager._conn_dict.values())
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    rebuilt = await client.post("/api/v1/sql/execute", json=query)
    assert rebuilt.json()["success"], rebuilt.text
    assert ast.literal_eval(rebuilt.json()["data"]["sql_return"]) == [{"total": 350}]
    assert cache._cache["default"] is not original


@pytest.mark.asyncio
async def test_http_eviction_waits_for_cancelled_native_sql_worker(resource_project):
    """Evicting a cancelled HTTP request must wait for its actual SQLite query."""
    client, cache = resource_project.client, resource_project.cache
    warm = await client.post("/api/v1/sql/execute", json={"sql_query": "SELECT 1", "result_format": "json"})
    assert warm.json()["success"], warm.text
    service = cache._cache["default"]
    connector = service.cli.db_manager.first_conn("sales")
    native = connector.connection
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(10), "The test must release the native SQLite query"
        return 350

    native.create_function("resource_hold", 0, hold)
    request = asyncio.create_task(
        client.post("/api/v1/sql/execute", json={"sql_query": "SELECT resource_hold() total", "result_format": "json"})
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        await cache.evict("default")
        request.cancel()
        await asyncio.sleep(0)
        assert not request.done()
        assert connector.connection is native
    finally:
        release.set()
    response = await request
    assert response.json()["errorMessage"] == "SQL execution was cancelled"
    await asyncio.wait_for(cache.drain(), timeout=10)
    assert connector.connection is None
    assert service._cli is None


@asynccontextmanager
async def _blocked_metadata_worker(project, method, path, *, params=None, json_body=None, timeout_patch=None):
    """Hold actual SQLite metadata execution, rather than a mocked datasource method."""
    from types import SimpleNamespace

    service = project.service
    connector = service.datasource.db_manager.first_conn("sales")
    connector.execute_query("SELECT 1")
    native = connector.connection
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(10), "The test must release native metadata execution"
        return 0

    native.set_progress_handler(hold, 1)
    if timeout_patch is not None:
        from datus.api.routes import database_routes

        async def timeout_after_worker_starts(awaitable, timeout):
            worker = asyncio.create_task(awaitable)
            assert await asyncio.to_thread(entered.wait, 10)
            # Use the real wait_for cancellation protocol once native work is
            # known to be running; no wall-clock startup race is required.
            return await asyncio.wait_for(worker, timeout=0)

        timeout_patch.setattr(database_routes, "asyncio", SimpleNamespace(wait_for=timeout_after_worker_starts))
    request = asyncio.create_task(project.client.request(method, path, params=params, json=json_body))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        yield SimpleNamespace(
            service=service,
            connector=connector,
            native=native,
            release=release,
            request=request,
        )
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,params,json_body",
    [
        ("GET", "/api/v1/table/detail", {"table": "orders"}, None),
        ("POST", "/api/v1/table/columns", None, {"tables": ["orders"]}),
    ],
    ids=["table-detail", "table-columns"],
)
async def test_http_metadata_eviction_waits_for_cancelled_native_worker(
    resource_project, method, path, params, json_body
):
    async with _blocked_metadata_worker(resource_project, method, path, params=params, json_body=json_body) as turn:
        try:
            await asyncio.wait_for(resource_project.cache.evict("default"), timeout=2)
            turn.request.cancel()
            # The worker has entered SQLite and remains blocked throughout this
            # observation window. Cancellation must not release the request pin.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(turn.request), timeout=0.05)
            assert turn.service.has_active_tasks()
            assert turn.connector.connection is turn.native
        finally:
            turn.release.set()
        with pytest.raises(asyncio.CancelledError):
            await turn.request
        await asyncio.wait_for(resource_project.cache.drain(), timeout=10)
        assert turn.connector.connection is None
        assert turn.service._datasource is None


@pytest.mark.asyncio
async def test_http_catalog_cancellation_keeps_worker_pin_until_native_finishes(resource_project):
    async with _blocked_metadata_worker(resource_project, "GET", "/api/v1/catalog/list") as turn:
        try:
            turn.request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(turn.request, timeout=1)
            assert turn.service.has_active_tasks() is True
            await asyncio.wait_for(resource_project.cache.evict("default"), timeout=2)
            assert turn.connector.connection is turn.native
        finally:
            turn.release.set()
        await asyncio.wait_for(resource_project.cache.drain(), timeout=10)
        assert turn.connector.connection is None
        assert turn.service._datasource is None


@pytest.mark.asyncio
async def test_http_catalog_timeout_returns_before_native_worker_finishes(resource_project, monkeypatch):
    async with _blocked_metadata_worker(
        resource_project, "GET", "/api/v1/catalog/list", timeout_patch=monkeypatch
    ) as turn:
        try:
            # The route has hit its actual wait_for deadline, while SQLite
            # remains blocked. The response must finish before native release.
            response = await asyncio.wait_for(turn.request, timeout=1)
            assert response.json()["errorCode"] == "REQUEST_TIMEOUT"
            assert turn.release.is_set() is False
            assert turn.service.has_active_tasks() is True
            await asyncio.wait_for(resource_project.cache.evict("default"), timeout=2)
            assert turn.connector.connection is turn.native
        finally:
            turn.release.set()
        await asyncio.wait_for(resource_project.cache.drain(), timeout=10)
        assert turn.connector.connection is None
        assert turn.service._datasource is None


@pytest.mark.asyncio
async def test_http_repeated_turns_do_not_accumulate_nodes_or_replay(resource_project, model_server, record_property):
    """Repeated real requests settle to a fixed connection count and empty replay index."""
    import gc
    import weakref

    client, service = resource_project.client, resource_project.service
    manager = service.task_manager
    observations = []

    def reply(request):
        if any(message.get("role") == "tool" for message in request["messages"]):
            return answer("Total: 350.")
        return tool_call("execute_sql", {"sql": "SELECT SUM(amount) total FROM orders"})

    for round_index in range(3):
        ids = [f"resource_round_{round_index}_{index}" for index in range(3)]
        model_server.enqueue(*([reply] * (2 * len(ids))))
        responses = await asyncio.gather(
            *(
                client.post(
                    "/api/v1/chat/stream", json={"message": "Compute total.", "session_id": sid, "interactive": False}
                )
                for sid in ids
            )
        )
        await manager.wait_all_tasks()
        for response in responses:
            assert response.status_code == 200
            assert find_event(parse_sse_body(response.text), "end") is not None, response.text
            assert "350" in response.text
        tasks = [manager.get_task(sid) for sid in ids]
        assert [(task.session_id, task.node) for task in tasks] == [(sid, None) for sid in ids]
        references = [weakref.ref(task) for task in tasks]
        del tasks
        assert not manager.has_active_tasks()
        await resource_project.wait_for_expiry(ids)
        assert not manager._completed_tasks
        gc.collect()
        assert all(reference() is None for reference in references)
        from datus.tools.db_tools.db_manager import db_manager_instance

        owner = db_manager_instance(service.agent_config.datasource_configs)
        assert not owner._leases
        retained = sum(len(group) for group in owner._conn_dict.values())
        assert retained == 1
        observations.append(
            {"round": round_index, "retained_connections": retained, "completed_tasks": len(manager._completed_tasks)}
        )
    record_property("resource_rounds", json.dumps(observations))


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["service", "cache", "refreshed-cache"])
async def test_http_service_shutdown_stops_active_sse_before_draining_requests(resource_project, model_server, owner):
    async with _blocked_http_turn(resource_project, model_server, "shutdown", answer("Finished.")) as turn:
        if owner == "refreshed-cache":
            import copy
            from unittest.mock import AsyncMock

            await resource_project.cache.get_or_create(
                "default",
                AsyncMock(),
                expected_fingerprint="changed",
                agent_config=copy.deepcopy(resource_project.service.agent_config),
            )
        shutdown = resource_project.service.shutdown if owner == "service" else resource_project.cache.shutdown
        stopping = asyncio.create_task(shutdown())
        try:
            await asyncio.wait_for(asyncio.shield(stopping), timeout=3)
            response = await turn.request
            assert response.status_code == 200
            assert turn.task.status == "cancelled"
            assert turn.task.node is None
            assert resource_project.service.has_active_tasks() is False
        finally:
            turn.release.set()
            await asyncio.gather(stopping, return_exceptions=True)


@pytest.mark.asyncio
async def test_http_next_turn_waits_for_terminal_cleanup(resource_project, model_server, monkeypatch):
    async with _blocked_http_turn(resource_project, model_server, "slow_cleanup", answer("First reply.")) as turn:
        entered, release = asyncio.Event(), asyncio.Event()
        node_type = type(turn.task.node)
        close = node_type.aclose

        async def slow_close(node):
            entered.set()
            await release.wait()
            await close(node)

        monkeypatch.setattr(node_type, "aclose", slow_close)
        following = None
        try:
            turn.release.set()
            first_response = await asyncio.wait_for(turn.request, timeout=5)
            assert find_event(parse_sse_body(first_response.text), "end")["event"] == "end"
            await asyncio.wait_for(entered.wait(), timeout=5)
            model_server.enqueue(answer("Second reply."))
            following = asyncio.create_task(
                turn.client.post(
                    "/api/v1/chat/stream", json={"message": "Continue.", "session_id": turn.sid, "interactive": False}
                )
            )
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(following), timeout=0.05)
            assert turn.manager.get_task(turn.sid) is turn.task
            release.set()
            response = await asyncio.wait_for(following, timeout=10)
            assert response.status_code == 200
            assert "Second reply." in response.text
            assert find_event(parse_sse_body(response.text), "end")["event"] == "end"
            await turn.manager.wait_all_tasks()
            await _assert_terminal_turn(turn, "completed")
        finally:
            release.set()
            await asyncio.gather(*([following] if following is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_followup_waiter_does_not_cancel_previous_cleanup(resource_project, model_server, monkeypatch):
    async with _blocked_http_turn(resource_project, model_server, "cleanup_waiter", answer("First reply.")) as turn:
        entered, release, following_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        node_type = type(turn.task.node)
        close = node_type.aclose
        start = turn.manager.start_chat

        async def slow_close(node):
            entered.set()
            await release.wait()
            await close(node)

        async def observe_followup(*args, **kwargs):
            following_entered.set()
            return await start(*args, **kwargs)

        monkeypatch.setattr(node_type, "aclose", slow_close)
        following = None
        try:
            turn.release.set()
            await asyncio.wait_for(turn.request, timeout=5)
            await asyncio.wait_for(entered.wait(), timeout=5)
            monkeypatch.setattr(turn.manager, "start_chat", observe_followup)
            following = asyncio.create_task(
                turn.client.post(
                    "/api/v1/chat/stream", json={"message": "Continue.", "session_id": turn.sid, "interactive": False}
                )
            )
            await asyncio.wait_for(following_entered.wait(), timeout=5)
            following.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(following), timeout=1)
            assert turn.task.asyncio_task.done() is False
            assert turn.task.asyncio_task.cancelling() == 0
            release.set()
            await _assert_terminal_turn(turn, "completed")
        finally:
            release.set()
            await asyncio.gather(*([following] if following is not None else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_http_shutdown_during_node_construction_closes_the_completed_node(resource_project, monkeypatch):
    manager = resource_project.service.task_manager
    create_node = manager._create_node
    entered, release = threading.Event(), threading.Event()
    created = []

    def blocked_constructor(*args, **kwargs):
        node = create_node(*args, **kwargs)
        created.append(node)
        result = node.db_func_tool.read_query("SELECT SUM(amount) AS total FROM orders", compress=False)
        assert result.success == 1, result.error
        assert result.result == [{"total": 350}]
        entered.set()
        assert release.wait(10), "The test must release node construction"
        return node

    monkeypatch.setattr(manager, "_create_node", blocked_constructor)
    sid = "shutdown_during_construction"
    request = asyncio.create_task(
        resource_project.client.post(
            "/api/v1/chat/stream", json={"message": "Create a node.", "session_id": sid, "interactive": False}
        )
    )
    stopping = None
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task = manager.get_task(sid)
        db_tool = created[0].db_func_tool
        db_manager = db_tool._db_manager
        assert len(db_manager._leases) == 1
        stopping = asyncio.create_task(manager.shutdown())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(stopping), timeout=0.05)
        task.asyncio_task.cancel()  # Repeated cancellation must also retain construction.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(stopping), timeout=0.05)
        assert len(db_manager._leases) == 1
        release.set()
        await asyncio.wait_for(asyncio.shield(stopping), timeout=10)
        await asyncio.wait_for(request, timeout=10)
        assert task.status == "cancelled"
        assert task.node is None
        assert db_manager._leases == {}
        assert db_tool._primary_connector is None
        assert db_tool._borrowed_connections == {}
    finally:
        release.set()
        await asyncio.gather(request, *([stopping] if stopping is not None else []), return_exceptions=True)
        # Also clean the real node when verifying the regression on old code.
        for node in created:
            await node.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["constructor", "message_snapshot"])
async def test_http_early_cancellation_preserves_previous_structured_message(
    resource_project, model_server, monkeypatch, phase
):
    manager = resource_project.service.task_manager
    sid = "cancelled_repeated_text"
    original_parts = [
        {"type": "markdown", "payload": {"content": "Same question."}},
        {"type": "markdown", "payload": {"content": "Original content."}},
    ]
    cancelled_parts = [{"type": "markdown", "payload": {"content": "Same question.\n\nOriginal content."}}]
    model_server.enqueue(answer("Original answer."))
    first = await resource_project.client.post(
        "/api/v1/chat/stream", json={"messages": original_parts, "session_id": sid, "interactive": False}
    )
    assert find_event(parse_sse_body(first.text), "end")["event"] == "end"
    await manager.wait_all_tasks()
    before_response = await resource_project.client.get("/api/v1/chat/history", params={"session_id": sid})
    assert before_response.json()["success"] is True
    before_messages = before_response.json()["data"]["messages"]
    assert [message["content"] for message in before_messages if message["role"] == "user"] == [original_parts]
    before = [(message["role"], message["content"]) for message in before_messages]

    entered, release = threading.Event(), threading.Event()
    operation = {"constructor": "_create_node", "message_snapshot": "_max_message_id"}[phase]
    original = getattr(manager, operation)

    def blocked_operation(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10), "The test must release initialization"
        return result

    monkeypatch.setattr(manager, operation, blocked_operation)
    request = asyncio.create_task(
        resource_project.client.post(
            "/api/v1/chat/stream", json={"messages": cancelled_parts, "session_id": sid, "interactive": False}
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task = manager.get_task(sid)
        task.asyncio_task.cancel()
        release.set()
        await asyncio.wait_for(task.asyncio_task, timeout=10)
        await asyncio.wait_for(request, timeout=10)
        assert task.status == "cancelled"
        after = await resource_project.client.get("/api/v1/chat/history", params={"session_id": sid})
        assert after.json()["success"] is True
        assert [(message["role"], message["content"]) for message in after.json()["data"]["messages"]] == before
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)
