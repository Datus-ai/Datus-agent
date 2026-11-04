"""Structured subagent results must render consistently live and on replay."""

import json
from unittest.mock import MagicMock

import pytest

from datus.api.models.cli_models import SSEDataType, StreamChatInput
from datus.api.services.action_sse_converter import action_to_sse_event
from datus.api.services.chat_service import ChatService, _actions_to_payloads
from datus.api.services.chat_task_manager import ChatTask, ChatTaskManager
from datus.models.session_manager import SessionManager
from datus.prompts.prompt_manager import PromptManager
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

SUMMARY = "## Result\n\n- Completed and verified."
JOB_RESULT = {
    "mode": "cross-db",
    "source_database": "source",
    "target_database": "target",
    "tables": ["orders"],
    "rows_transferred": 10,
    "validation": {"total_checks": 1, "passed": 1, "failed": 0, "details": []},
    "explanation": SUMMARY,
}
PROTOCOLS = [
    pytest.param(
        "semantic_modeling",
        {"status": "generated", "blocker_code": None, "skip_reason": None, "output": SUMMARY},
        None,
        id="semantic-modeling",
    ),
    pytest.param("gen_sql", {"sql": "SELECT 1", "output": SUMMARY}, "SELECT 1", id="gen-sql"),
    pytest.param(
        "gen_sql_summary", {"sql_summary_file": "orders_abc.yaml", "output": SUMMARY}, None, id="gen-sql-summary"
    ),
    pytest.param("gen_table", {"table_name": "orders", "output": SUMMARY}, None, id="gen-table"),
    pytest.param("gen_job", {**JOB_RESULT, "sql": ""}, None, id="gen-job-cross-db"),
    pytest.param("gen_job", JOB_RESULT, None, id="gen-job-without-sql"),
    pytest.param(
        "gen_job",
        {**JOB_RESULT, "mode": "intra-db", "sql": "INSERT INTO orders SELECT 1"},
        "INSERT INTO orders SELECT 1",
        id="gen-job-intra-db",
    ),
    pytest.param(
        "gen_job",
        {**JOB_RESULT, "mode": "intra-db", "sql": r"INSERT INTO target SELECT $$\($$"},
        r"INSERT INTO target SELECT $$\($$",
        id="gen-job-dollar-literal",
    ),
]


def _action(action_id="response", *, depth=0, parent=None, action_type="response", output=None, node_class=None):
    return ActionHistory(
        action_id=action_id,
        role=ActionRole.ASSISTANT,
        action_type=action_type,
        messages="",
        input={"_node_class": node_class} if node_class else {},
        output=output or {},
        status=ActionStatus.SUCCESS,
        depth=depth,
        parent_action_id=parent,
    )


def _assert_summary(contents, sql=None):
    assert [(c.type, c.payload) for c in contents] == (
        ([("code", {"codeType": "sql", "content": sql})] if sql else []) + [("markdown", {"content": SUMMARY})]
    )


def test_user_content_restore_skips_node_response_records():
    task = ChatTask(session_id="mixed-contents", asyncio_task=MagicMock())
    first = [{"type": "markdown", "payload": {"content": "first"}}]
    second = [{"type": "markdown", "payload": {"content": "second"}}]
    task.register_message_contents("same user text", first)
    ChatTaskManager._record_node_response(task, _action(node_class="gen_sql"), ("sql", "raw SQL result"), [])
    task.register_message_contents("same user text", second)

    assert task.restore_message_contents("same user text", "user-1") == first
    assert task.restore_message_contents("same user text", "user-2") == second
    assert task.restore_message_contents("unregistered user text", "user-3") is None
    assert len(set(task.content_events.values())) == 2


@pytest.mark.parametrize("is_thinking", [False, True])
@pytest.mark.parametrize("depth", [0, 1])
def test_envelope_sql_normalization_preserves_quoted_literals(is_thinking, depth):
    sql = "SELECT COUNT\\(*)\nFROM orders WHERE path = 'C:\\(data\\)' AND note = 'a''\\(b\\)'"
    expected = "SELECT COUNT(*)\nFROM orders WHERE path = 'C:\\(data\\)' AND note = 'a''\\(b\\)'"
    raw = json.dumps({"sql": sql, "output": SUMMARY})
    action = _action(
        depth=depth,
        parent="sql-task" if depth else None,
        output={"raw_output": raw, "is_thinking": is_thinking},
        node_class="gen_sql",
    )
    event = action_to_sse_event(action, 1, action.action_id)
    _assert_summary(event.data.payload.content, expected)
    assert action.output["raw_output"] == raw


@pytest.mark.parametrize("version", ["0.9", "1.1"])
@pytest.mark.parametrize("is_thinking", [False, True])
def test_legacy_chat_completion_uses_host_prompt_identity(version, is_thinking):
    raw = json.dumps({"sql": "SELECT 1", "output": SUMMARY})
    action = _action(output={"raw_output": raw, "is_thinking": is_thinking}, node_class="chat")
    action.input.update(_system_prompt_template="chat_system", _prompt_version=version)
    event = action_to_sse_event(action, 1, action.action_id, stream_thinking=True, is_update=True)
    assert event.data.type == SSEDataType.UPDATE_MESSAGE
    _assert_summary(event.data.payload.content, "SELECT 1")
    assert action.input["_node_class"] == "chat"


@pytest.mark.asyncio
async def test_gen_sql_history_normalizes_sql_without_saved_display(real_agent_config):
    session_id = "gen_sql_session_escape_history"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    raw = json.dumps({"sql": r"SELECT COUNT\(*) FROM orders", "output": SUMMARY})
    await session.add_items(
        [
            {"role": "user", "content": "Count orders"},
            {"role": "assistant", "content": [{"type": "output_text", "text": raw}]},
        ]
    )
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == 1
    _assert_summary(answers[0].content, "SELECT COUNT(*) FROM orders")
    assert (await session.get_items())[-1]["content"][0]["text"] == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["0.9", "1.1"])
@pytest.mark.parametrize("include_sql", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("metadata_kind", ["correct", "wrong_version", "missing_template"])
async def test_legacy_chat_history_uses_prompt_provenance(
    real_agent_config, version, include_sql, delegated, metadata_kind
):
    parent_id = "chat_session_legacy_parent"
    child_id = "chat_session_legacy_child"
    call_id = "legacy-chat-task"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    parent = storage.get_session(parent_id)
    child_storage = SessionManager(session_dir=f"{real_agent_config.session_dir}/{parent_id}")
    owner = child_storage if delegated else storage
    session_id = child_id if delegated else parent_id
    session = owner.get_session(session_id)
    sql = "SELECT COUNT(*) FROM orders" if include_sql else None
    payload = {"output": SUMMARY, **({"sql": sql} if include_sql else {})}
    raw = json.dumps(payload)
    await session.add_items(
        [
            {"role": "user", "content": "Analyze orders"},
            {"role": "assistant", "content": [{"type": "output_text", "text": raw}]},
        ]
    )
    meta = {"node_name": "chat", "system_prompt_template": "chat_system", "prompt_version": version}
    if metadata_kind != "correct":
        meta["prompt_version"] = "1.2"
    if metadata_kind == "missing_template":
        meta.pop("system_prompt_template")
    owner.save_system_prompt_snapshot(
        session_id,
        PromptManager(agent_config=real_agent_config).render_template("chat_system", version=version),
        meta,
    )
    original_snapshot = owner.load_system_prompt_snapshot(session_id)
    if delegated:
        await parent.add_items(
            [
                {"role": "user", "content": "Delegate analysis"},
                {"type": "function_call", "call_id": call_id, "name": "task", "arguments": '{"type":"chat"}'},
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"success": True, "result": {"session_id": child_id}}),
                },
            ]
        )
    history = ChatService(real_agent_config).get_history(parent_id)
    assert history.success is True
    answers = [
        message for message in history.data.messages if message.role == "assistant" and message.depth == int(delegated)
    ]
    assert len(answers) == 1
    _assert_summary(answers[0].content, sql)
    assert answers[0].parent_action_id == (call_id if delegated else None)
    assert (await session.get_items())[-1]["content"][0]["text"] == raw
    assert owner.load_system_prompt_snapshot(session_id) == original_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("template,version", [("chat_system", "1.2"), ("chat_system", ""), ("custom_system", "0.9")])
async def test_chat_history_keeps_json_without_legacy_prompt_provenance(real_agent_config, template, version):
    session_id = "chat_session_freeform_json"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    raw = json.dumps({"sql": "SELECT 1", "output": "User data"})
    await session.add_items(
        [
            {"role": "user", "content": "Show JSON"},
            {"role": "assistant", "content": [{"type": "output_text", "text": raw}]},
        ]
    )
    storage.save_system_prompt_snapshot(
        session_id, "Chat prompt", {"node_name": "chat", "system_prompt_template": template, "prompt_version": version}
    )
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == 1
    assert [part.payload["content"] for part in answers[0].content] == [raw]


@pytest.mark.asyncio
@pytest.mark.parametrize("appended_version", ["0.9", "1.1"])
async def test_modern_chat_prompt_ignores_legacy_examples_appended_to_snapshot(real_agent_config, appended_version):
    session_id = "chat_session_modern_prompt"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    raw = json.dumps({"sql": "SELECT 1", "output": "User data"})
    await session.add_items(
        [
            {"role": "user", "content": "Show JSON"},
            {"role": "assistant", "content": [{"type": "output_text", "text": raw}]},
        ]
    )
    prompts = PromptManager(agent_config=real_agent_config)
    prompt = prompts.render_template("chat_system", version="1.2")
    prompt += "\n\n" + prompts.render_template("chat_system", version=appended_version)
    storage.save_system_prompt_snapshot(
        session_id, prompt, {"node_name": "chat", "system_prompt_template": "chat_system", "prompt_version": "0.9"}
    )
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == 1
    assert [part.payload["content"] for part in answers[0].content] == [raw]


@pytest.mark.parametrize("node,payload,sql", PROTOCOLS)
@pytest.mark.parametrize("is_thinking", [False, True])
def test_rejected_known_protocol_retains_all_fields(node, payload, sql, is_thinking):
    raw = json.dumps({**payload, "custom": True})
    action = _action(output={"raw_output": raw, "is_thinking": is_thinking}, node_class=node)
    event = action_to_sse_event(action, 1, action.action_id)
    assert [(part.type, part.payload["content"]) for part in event.data.payload.content] == [
        ("thinking" if is_thinking else "markdown", raw)
    ]
    assert action.output["raw_output"] == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_valid", [False, True])
async def test_rejected_protocol_is_not_replaced_by_lossy_wrapper_on_reload(real_agent_config, prior_valid):
    session_id = "gen_sql_session_rejected_protocol"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    raw = json.dumps({"sql": "SELECT 1", "output": SUMMARY, "custom": True})
    await session.add_items([{"role": "user", "content": "Generate SQL"}])

    class FakeNode:
        session_id = "gen_sql_session_rejected_protocol"

        async def execute_stream_with_interactions(self, action_history_manager):
            if prior_valid:
                earlier = json.dumps({"sql": "SELECT 0", "output": "Earlier result"})
                await session.add_items([{"role": "assistant", "content": [{"type": "output_text", "text": earlier}]}])
                yield _action("earlier", output={"raw_output": earlier, "is_thinking": False}, node_class="gen_sql")
            await session.add_items([{"role": "assistant", "content": [{"type": "output_text", "text": raw}]}])
            yield _action("raw", output={"raw_output": raw, "is_thinking": False}, node_class="gen_sql")
            yield _action("model-final", action_type="final_response", output={"raw_output": raw}, node_class="gen_sql")
            yield _action("parsed", action_type="gen_sql_response", output={"sql": "SELECT 1", "response": SUMMARY})

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: FakeNode()
    task = ChatTask(session_id=session_id, asyncio_task=MagicMock())
    await manager._run_loop(task, real_agent_config, StreamChatInput(message="Generate SQL", session_id=session_id))
    assert task.status == "completed"
    live = [event.data.payload for event in task.events if event.event == "message"]
    expected = (["SELECT 0", "Earlier result"] if prior_valid else []) + [raw]
    assert [part.payload["content"] for message in live for part in message.content] == expected
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == (2 if prior_valid else 1)
    assert [part.payload["content"] for message in answers for part in message.content] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("node,payload,sql", PROTOCOLS)
async def test_multipart_node_wrapper_preserves_rendered_result_on_reload(real_agent_config, node, payload, sql):
    session_id = "api-multipart-result"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    preface = "The requested work is complete."
    raw = json.dumps(payload)
    aggregate = f"{preface}\n{raw}"
    blocks = [{"type": "text", "text": preface}, {"type": "text", "text": raw}]
    await session.add_items([{"role": "user", "content": "Run the task"}])

    class FakeNode:
        session_id = "api-multipart-result"

        async def execute_stream_with_interactions(self, action_history_manager):
            await session.add_items([{"role": "assistant", "content": blocks}])
            yield _action("preface", output={"raw_output": preface, "is_thinking": False}, node_class=node)
            yield _action("result", output={"raw_output": raw, "is_thinking": False}, node_class=node)
            yield _action(
                "model-final", action_type="final_response", output={"raw_output": aggregate}, node_class=node
            )
            yield _action("node-final", action_type=f"{node}_response", output={"response": aggregate})

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: FakeNode()
    task = ChatTask(session_id=session_id, asyncio_task=MagicMock())
    await manager._run_loop(task, real_agent_config, StreamChatInput(message="Run the task", session_id=session_id))
    assert task.status == "completed"
    live = [event.data.payload for event in task.events if event.event == "message"]
    assert len(live) == 2
    _assert_summary(live[-1].content, sql)
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == 1
    _assert_summary(answers[0].content, sql)
    assert (await session.get_items())[-1]["content"] == blocks


@pytest.mark.parametrize("node,payload,sql", PROTOCOLS)
@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("is_thinking", [False, True])
def test_completed_response_extracts_known_protocol(node, payload, sql, depth, fenced, is_thinking):
    raw = json.dumps(payload)
    if fenced:
        raw = f"```json\n{raw}\n```"
    action = _action(
        depth=depth,
        parent="task" if depth else None,
        output={"raw_output": raw, "is_thinking": is_thinking},
        node_class=node,
    )
    event = action_to_sse_event(action, 1, action.action_id, stream_thinking=True, is_update=True)
    assert event.data.type == SSEDataType.UPDATE_MESSAGE
    assert event.data.payload.message_id == action.action_id
    assert event.data.payload.depth == depth
    assert event.data.payload.parent_action_id == action.parent_action_id
    _assert_summary(event.data.payload.content, sql)
    assert action.output["raw_output"] == raw  # Keep the internal result intact.


@pytest.mark.parametrize("node,payload,sql", PROTOCOLS)
def test_wrapper_and_history_use_the_same_protocol(node, payload, sql):
    raw = json.dumps(payload)
    response = _action(output={"raw_output": raw, "is_thinking": False}, node_class=node)
    # gen_table/gen_job wrappers still carry the original structured response.
    wrapper = _action("final", action_type=f"{node}_response", output={"response": raw, "sql": sql})
    event = action_to_sse_event(wrapper, 1, wrapper.action_id, include_final_response=True)
    _assert_summary(event.data.payload.content, sql)
    messages, _ = _actions_to_payloads([response, wrapper], 0)
    assert len(messages) == 1
    _assert_summary(messages[0].content, sql)


def test_wrapper_prefers_resolved_sql_over_enclosed_file_path():
    wrapper = _action(
        action_type="gen_sql_response",
        output={"sql": "SELECT 1", "response": json.dumps({"sql": "query.sql", "output": SUMMARY})},
    )
    event = action_to_sse_event(wrapper, 1, wrapper.action_id, include_final_response=True)
    assert [(part.type, part.payload) for part in event.data.payload.content] == [
        ("code", {"codeType": "sql", "content": "SELECT 1"}),
        ("markdown", {"content": SUMMARY}),
    ]


@pytest.mark.parametrize(
    "sql", [r"SELECT $$\($$", r"SELECT $pattern$\($pattern$", 'SELECT "column\\(name" FROM orders']
)
def test_resolved_sql_file_content_is_verbatim(sql):
    action = _action(
        action_type="gen_sql_response",
        output={"sql": sql, "sql_file_path": "queries/report.sql", "response": SUMMARY},
    )
    event = action_to_sse_event(action, 1, action.action_id, include_final_response=True)
    assert [(part.type, part.payload) for part in event.data.payload.content] == [
        ("code", {"codeType": "sql", "content": sql}),
        ("markdown", {"content": SUMMARY}),
    ]
    assert action.output["sql"] == sql


@pytest.mark.parametrize("sql", ["SELECT * FROM test.sql", "SELECT\t*\tFROM\ttest.sql", "SELECT *\nFROM test.sql"])
@pytest.mark.parametrize("action_type", ["response", "thinking", "gen_sql_response"])
def test_inline_sql_ending_in_qualified_sql_table_is_executable(sql, action_type):
    action = _action(
        action_type=action_type,
        output={"raw_output": json.dumps({"sql": sql, "output": SUMMARY}), "is_thinking": False},
        node_class="gen_sql",
    )
    event = action_to_sse_event(action, 1, action.action_id, include_final_response=True)
    assert [(part.type, part.payload) for part in event.data.payload.content] == [
        ("code", {"codeType": "sql", "content": sql}),
        ("markdown", {"content": SUMMARY}),
    ]


@pytest.mark.parametrize("status", ["skipped", "blocked"])
def test_semantic_outcomes_display_guidance_without_protocol_metadata(status):
    action = _action(
        output={"raw_output": {"status": status, "output": SUMMARY}, "is_thinking": False},
        node_class="semantic_modeling",
    )
    event = action_to_sse_event(action, 1, action.action_id)
    assert [(part.type, part.payload) for part in event.data.payload.content] == [("markdown", {"content": SUMMARY})]


@pytest.mark.parametrize(
    "text",
    [
        "## Ordinary Markdown\n\nAn answer.",
        '{"output": "user data", "items": [1, 2]}',
        '{"sql": "SELECT 1", "output": "user data", "custom": true}',
        'An example:\n```json\n{"sql":"SELECT 1","output":"example"}\n```',
        '{"table_name": "orders", "output": ["not markdown"]}',
        '{"table_name": "orders", "output": "user data"}',
        '{"sql": "SELECT 1", "output": "user data"}',
        '```json\n{"status":"generated","output":"user data"}\n```',
    ],
)
@pytest.mark.parametrize("is_thinking", [False, True])
def test_unrelated_content_is_preserved(text, is_thinking):
    action = _action(output={"raw_output": text, "is_thinking": is_thinking}, node_class="chat")
    event = action_to_sse_event(action, 1, action.action_id)
    assert event.data.payload.content[0].payload["content"] == text


@pytest.mark.parametrize("node,payload,sql", PROTOCOLS)
@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("fresh_completion_id", [False, True])
@pytest.mark.asyncio
async def test_stream_replaces_json_and_preserves_parent_answer(
    real_agent_config, node, payload, sql, depth, fresh_completion_id
):
    raw = json.dumps(payload)
    parent = "task" if depth else None
    delta = _action(
        "stream", depth=depth, parent=parent, action_type="thinking_delta", output={"delta": raw, "accumulated": raw}
    )
    delta.status = ActionStatus.PROCESSING
    response = _action(
        "completion" if fresh_completion_id else "stream",
        depth=depth,
        parent=parent,
        output={"raw_output": raw, "is_thinking": False},
        node_class=node,
    )
    wrapper = _action(
        "final", depth=depth, parent=parent, action_type=f"{node}_response", output={"response": SUMMARY, "sql": sql}
    )
    actions = [delta, response, wrapper]
    if depth:
        actions.append(_action("parent-final", action_type="chat_response", output={"response": SUMMARY}))

    class FakeNode:
        session_id = "structured-results"

        async def execute_stream_with_interactions(self, action_history_manager):
            for action in actions:
                yield action

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: FakeNode()
    task = ChatTask(session_id="structured-results", asyncio_task=MagicMock())
    await manager._run_loop(
        task, real_agent_config, StreamChatInput(message="run", session_id=task.session_id, stream_response=True)
    )
    assert task.status == "completed"
    messages = [e for e in task.events if e.event == "message"]
    assert len(messages) == (3 if depth else 2)
    assert messages[0].data.type == SSEDataType.CREATE_MESSAGE
    assert messages[1].data.type == SSEDataType.UPDATE_MESSAGE
    assert messages[0].data.payload.message_id == messages[1].data.payload.message_id
    _assert_summary(messages[1].data.payload.content, sql)
    assert messages[-1].data.payload.depth == 0
    _assert_summary(messages[-1].data.payload.content, None if depth else sql)


def test_replay_preserves_identical_answers_from_different_tasks_and_parent():
    actions = [
        _action("child-a", depth=1, parent="task-a", output={"raw_output": SUMMARY, "is_thinking": False}),
        _action("child-b", depth=1, parent="task-b", output={"raw_output": SUMMARY, "is_thinking": False}),
        _action("parent", action_type="chat_response", output={"response": SUMMARY}),
    ]
    messages, _ = _actions_to_payloads(actions, 0)
    assert [(m.depth, m.parent_action_id) for m in messages] == [(1, "task-a"), (1, "task-b"), (0, None)]


def test_live_dedupe_preserves_each_tasks_own_updates():
    from datus.api.services.chat_task_manager import (
        _remember_assistant_message,
        _should_skip_duplicate_assistant_message,
    )

    seen = {}
    first = _action("first", depth=1, parent="task-a", output={"raw_output": SUMMARY, "is_thinking": False})
    event = action_to_sse_event(first, 1, first.action_id)
    _remember_assistant_message(event, seen)
    for parent, depth in [("task-b", 1), (None, 0)]:
        action = _action("another", depth=depth, parent=parent, output={"raw_output": SUMMARY, "is_thinking": False})
        other = action_to_sse_event(action, 2, action.action_id, stream_thinking=True, is_update=True)
        assert not _should_skip_duplicate_assistant_message(action, other, seen)


def test_repeated_structured_summary_still_replaces_its_json_stream():
    """A retry's completion must not leave its raw JSON bubble on screen."""
    from datus.api.services.chat_task_manager import (
        _remember_assistant_message,
        _should_skip_duplicate_assistant_message,
    )

    raw = json.dumps({"status": "generated", "output": SUMMARY})
    first = _action("first", output={"raw_output": raw, "is_thinking": False}, node_class="semantic_modeling")
    event = action_to_sse_event(first, 1, first.action_id)
    seen = {}
    _remember_assistant_message(event, seen)

    retry = _action("retry", output={"raw_output": raw, "is_thinking": False}, node_class="semantic_modeling")
    update = action_to_sse_event(retry, 2, retry.action_id, stream_thinking=True, is_update=True)
    assert not _should_skip_duplicate_assistant_message(retry, update, seen)
    _assert_summary(update.data.payload.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("padding", [("", ""), ("", "\n"), ("\n \t", "\r\n \t")])
@pytest.mark.parametrize("multipart", [False, True])
async def test_sql_file_is_updated_with_resolved_sql_and_persisted(real_agent_config, stream, padding, multipart):
    """Use the actual live loop and SQLite replay, including a nonstandard session id."""
    session_id = "api-generated-session-id"
    raw = json.dumps({"sql": "queries/report.sql", "output": SUMMARY})
    sdk_text = f"{padding[0]}{raw}{padding[1]}"
    preface = "I verified the result."
    sdk_blocks = ([{"type": "text", "text": preface}] if multipart else []) + [
        {"type": "text" if multipart else "output_text", "text": sdk_text}
    ]
    sql = r"SELECT $$\($$ AS value FROM test.sql"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    session = storage.get_session(session_id)
    await session.add_items([{"role": "user", "content": "Generate SQL"}])

    class FakeNode:
        session_id = "api-generated-session-id"

        async def execute_stream_with_interactions(self, action_history_manager):
            if multipart:
                yield _action("preface", output={"raw_output": preface, "is_thinking": False}, node_class="gen_sql")
            delta = _action("stream", action_type="thinking_delta", output={"delta": sdk_text})
            delta.status = ActionStatus.PROCESSING
            yield delta
            await session.add_items([{"role": "assistant", "content": sdk_blocks}])
            yield _action("stream", output={"raw_output": raw, "is_thinking": False}, node_class="gen_sql")
            if multipart:
                yield _action(
                    "model-final",
                    action_type="final_response",
                    output={"raw_output": "\n".join(block["text"] for block in sdk_blocks)},
                    node_class="gen_sql",
                )
            yield _action(
                "final",
                action_type="custom_sql_agent_response",
                node_class="gen_sql",
                output={"response": SUMMARY, "sql": sql, "sql_file_path": "queries/report.sql"},
            )

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: FakeNode()
    task = ChatTask(session_id=session_id, asyncio_task=MagicMock())
    await manager._run_loop(
        task, real_agent_config, StreamChatInput(message="Generate SQL", session_id=session_id, stream_response=stream)
    )
    assert task.status == "completed"
    assert (await session.get_items())[-1]["content"] == sdk_blocks
    messages = [event for event in task.events if event.event == "message"]
    for event in messages:
        assert not any(
            item.type == "code" and item.payload["content"] == "queries/report.sql"
            for item in event.data.payload.content
        )
    assert messages[-1].data.type == SSEDataType.UPDATE_MESSAGE
    assert messages[-1].data.payload.message_id == "stream"
    _assert_summary(messages[-1].data.payload.content, sql)
    history = ChatService(real_agent_config).get_history(session_id)
    assert history.success is True
    answers = [message for message in history.data.messages if message.role == "assistant"]
    assert len(answers) == 1
    _assert_summary(answers[0].content, sql)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("padding", [("", ""), ("", "\n"), ("\n \t", "\r\n \t")])
@pytest.mark.parametrize("multipart", [False, True])
async def test_delegated_sql_file_view_survives_real_history_expansion(real_agent_config, stream, padding, multipart):
    parent_id = "chat_session_file_parent"
    child_id = "custom_sql_session_file_child"
    call_id = "sql-task"
    raw = json.dumps({"sql": "queries/report.sql", "output": SUMMARY})
    sdk_text = f"{padding[0]}{raw}{padding[1]}"
    preface = "I verified the delegated result."
    sdk_blocks = ([{"type": "text", "text": preface}] if multipart else []) + [
        {"type": "text" if multipart else "output_text", "text": sdk_text}
    ]
    sql = r"SELECT $pattern$\($pattern$ AS value FROM test.sql"
    storage = SessionManager(session_dir=real_agent_config.session_dir)
    parent = storage.get_session(parent_id)
    child_storage = SessionManager(session_dir=f"{real_agent_config.session_dir}/{parent_id}")
    child = child_storage.get_session(child_id)
    await parent.add_items([{"role": "user", "content": "Generate SQL"}])
    await child.add_items([{"role": "user", "content": "Generate SQL"}])
    result = {"success": True, "result": {"session_id": child_id, "response": SUMMARY}}

    class FakeNode:
        session_id = parent_id

        async def execute_stream_with_interactions(self, action_history_manager):
            await parent.add_items(
                [
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "task",
                        "arguments": json.dumps({"type": "custom_sql", "prompt": "Generate SQL"}),
                    }
                ]
            )
            start = _action(call_id, action_type="task")
            start.role, start.status = ActionRole.TOOL, ActionStatus.PROCESSING
            start.input = {"function_name": "task", "arguments": {"type": "custom_sql"}}
            yield start
            if multipart:
                yield _action(
                    "child-preface",
                    depth=1,
                    parent=call_id,
                    node_class="gen_sql",
                    output={"raw_output": preface, "is_thinking": False},
                )
            delta = _action(
                "child-stream", depth=1, parent=call_id, action_type="thinking_delta", output={"delta": sdk_text}
            )
            delta.status = ActionStatus.PROCESSING
            yield delta
            await child.add_items([{"role": "assistant", "content": sdk_blocks}])
            yield _action(
                "child-stream",
                depth=1,
                parent=call_id,
                node_class="gen_sql",
                output={"raw_output": raw, "is_thinking": False},
            )
            if multipart:
                yield _action(
                    "child-model-final",
                    depth=1,
                    parent=call_id,
                    node_class="gen_sql",
                    action_type="final_response",
                    output={"raw_output": "\n".join(block["text"] for block in sdk_blocks)},
                )
            yield _action(
                "child-final",
                depth=1,
                parent=call_id,
                node_class="gen_sql",
                action_type="custom_sql_response",
                output={"response": SUMMARY, "sql": sql, "sql_file_path": "queries/report.sql"},
            )
            await parent.add_items([{"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)}])
            done = _action(f"complete_{call_id}", action_type="task", output=result)
            done.role, done.input = ActionRole.TOOL, start.input
            yield done

        async def get_last_turn_usage(self):
            return None

    manager = ChatTaskManager()
    manager._create_node = lambda *args, **kwargs: FakeNode()
    task = ChatTask(session_id=parent_id, asyncio_task=MagicMock())
    await manager._run_loop(
        task, real_agent_config, StreamChatInput(message="Generate SQL", session_id=parent_id, stream_response=stream)
    )
    assert task.status == "completed"
    assert (await child.get_items())[-1]["content"] == sdk_blocks
    updates = [
        event for event in task.events if event.event == "message" and event.data.type == SSEDataType.UPDATE_MESSAGE
    ]
    _assert_summary(updates[-1].data.payload.content, sql)
    history = ChatService(real_agent_config).get_history(parent_id)
    assert history.success is True
    sql_answers = [message for message in history.data.messages if any(part.type == "code" for part in message.content)]
    assert len(sql_answers) == 1
    assert sql_answers[0].depth == 1
    assert sql_answers[0].parent_action_id == call_id
    _assert_summary(sql_answers[0].content, sql)
