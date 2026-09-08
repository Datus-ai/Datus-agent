# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for AgenticNode plan-mode persistence layer — CI tier."""

import json

import pytest


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """``cd`` into tmp_path so ``./.datus/plans/*.md`` lands in test scope."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _path_manager(node):
    from datus.utils.path_manager import get_path_manager

    return get_path_manager(agent_config=node.agent_config)


def _state_path(node, session_id):
    """Resolve ``agent_state_path`` the way production code does."""
    return _path_manager(node).agent_state_path(session_id)


def _write_legacy_plan_mode(path, **fields):
    """Write the pre-migration ``plan_mode`` section by hand.

    Plan-mode state now lives in the session database, so production no longer
    produces this layout. Sessions created before the move still carry it and
    must keep restoring, which is what these fixtures exercise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plan_mode": fields}), encoding="utf-8")


def _materialize_session(node):
    """Create the session database, the way the first message does."""
    node._get_or_create_session()


def _make_chat_node(real_agent_config, session_id=None):
    """Build a real ChatAgenticNode with the persistence-test config."""
    from datus.agent.node.chat_agentic_node import ChatAgenticNode
    from datus.configuration.node_type import NodeType

    return ChatAgenticNode(
        node_id="test_persist",
        description="Persistence node",
        node_type=NodeType.TYPE_CHAT,
        agent_config=real_agent_config,
        session_id=session_id,
    )


class TestPlanModeStatePersistence:
    """``activate_plan_mode`` / ``deactivate_plan_mode`` flush to the session db."""

    def test_activate_survives_a_rebuilt_node(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_aaaa")
        _materialize_session(node)

        node.activate_plan_mode()

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_aaaa")
        assert rebuilt.plan_mode_active is True
        assert rebuilt.plan_file_path == node.plan_file_path
        assert rebuilt.workflow_prompt_sent is False

    def test_deactivate_survives_a_rebuilt_node(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_bbbb")
        _materialize_session(node)
        node.activate_plan_mode()
        node.deactivate_plan_mode()

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_bbbb")
        # plan_mode_active flipped back to False; plan_file_path is preserved.
        assert rebuilt.plan_mode_active is False
        assert rebuilt.plan_file_path == node.plan_file_path
        assert rebuilt.workflow_prompt_sent is False

    def test_plan_mode_entered_before_the_first_message_is_still_persisted(self, chdir_tmp, real_agent_config):
        """Entering plan mode before typing finds no database to write to.

        ``session_id`` is allocated in ``__init__`` but the database is only
        created on the first message, and the write is skipped rather than
        materialising an empty file that would list as a session. Creating the
        session has to flush the pending flags, or a chat that entered plan
        mode before its first message would resume without it.
        """
        node = _make_chat_node(real_agent_config, session_id="chat_session_early")
        node.activate_plan_mode()

        _materialize_session(node)

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_early")
        assert rebuilt.plan_mode_active is True
        assert rebuilt.plan_file_path == node.plan_file_path

    def test_fresh_node_generates_session_id(self, chdir_tmp, real_agent_config):
        """When caller omits ``session_id``, ``__init__`` allocates one eagerly
        so persistence has a stable key from the very first turn."""
        node = _make_chat_node(real_agent_config)  # no session_id

        assert node.session_id  # always non-empty after construction
        assert node.session_id.startswith("chat_session_")

        _materialize_session(node)
        node.activate_plan_mode()

        rebuilt = _make_chat_node(real_agent_config, session_id=node.session_id)
        assert rebuilt.plan_mode_active is True


class TestSessionIdConstructorTriggersRestore:
    """Passing ``session_id`` to ``__init__`` rehydrates persisted plan-mode."""

    def test_constructor_session_id_restores(self, chdir_tmp, real_agent_config):
        anchor = _make_chat_node(real_agent_config, session_id="chat_session_dddd")
        _materialize_session(anchor)
        anchor.plan_file_path = "./.datus/plans/init.md"
        anchor.activate_plan_mode()

        node = _make_chat_node(real_agent_config, session_id="chat_session_dddd")

        assert node.plan_mode_active is True
        assert node.plan_file_path == "./.datus/plans/init.md"
        assert node._plan_just_confirmed is False  # one-shot flag never restored

    def test_a_state_file_from_before_the_move_still_restores(self, chdir_tmp, real_agent_config):
        """Sessions predating the database move keep their JSON section."""
        anchor = _make_chat_node(real_agent_config)
        _write_legacy_plan_mode(
            _state_path(anchor, "chat_session_legacyplan"),
            plan_mode_active=True,
            plan_file_path="./.datus/plans/init.md",
            workflow_prompt_sent=False,
        )

        node = _make_chat_node(real_agent_config, session_id="chat_session_legacyplan")

        assert node.plan_mode_active is True
        assert node.plan_file_path == "./.datus/plans/init.md"

    def test_the_database_wins_over_a_stale_state_file(self, chdir_tmp, real_agent_config):
        """A migrated session must not be dragged back by its leftover file."""
        anchor = _make_chat_node(real_agent_config, session_id="chat_session_bothsources")
        _materialize_session(anchor)
        anchor.activate_plan_mode()
        anchor.deactivate_plan_mode()
        _write_legacy_plan_mode(
            _state_path(anchor, "chat_session_bothsources"),
            plan_mode_active=True,
            plan_file_path="./.datus/plans/stale.md",
            workflow_prompt_sent=True,
        )

        node = _make_chat_node(real_agent_config, session_id="chat_session_bothsources")

        assert node.plan_mode_active is False

    def test_constructor_no_state_file_keeps_defaults(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_unknown")
        # No file present → defaults remain (False/None/False).
        assert node.plan_mode_active is False
        assert node.plan_file_path is None
        assert node.workflow_prompt_sent is False


class TestCompactStateNotPersisted:
    """``_compacted_until`` is an in-memory scan-start hint only — it must
    never be written to ``agent_state.json`` and a rebuilt node always
    starts at zero. Idempotency comes from the in-message
    ``[DATUS_ARCHIVED]`` marker, not from disk state.
    """

    def test_rebuilt_node_starts_at_zero_even_after_in_memory_advance(self, chdir_tmp, real_agent_config):
        """Anchor node advances the in-memory mark; a second node opened on
        the same session must NOT see that advance — it always starts fresh.
        """
        anchor = _make_chat_node(real_agent_config, session_id="chat_session_compact_r")
        _materialize_session(anchor)
        anchor._compacted_until = 10
        anchor.activate_plan_mode()  # forces a persisted plan-mode row

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_compact_r")
        # Fresh process → fresh scan; in-memory state is intentionally NOT
        # round-tripped because the archive marker covers correctness.
        assert rebuilt._compacted_until == 0

    def test_legacy_compact_section_is_ignored_on_load(self, chdir_tmp, real_agent_config):
        """Files written by older code carry a ``compact`` section. The loader
        must read past it without crashing and without rehydrating it into
        any node attribute.
        """
        anchor = _make_chat_node(real_agent_config, session_id="chat_session_compact_legacy")
        path = _state_path(anchor, "chat_session_compact_legacy")
        path.write_text(
            json.dumps(
                {
                    "plan_mode": {
                        "plan_mode_active": True,
                        "plan_file_path": "p.md",
                        "workflow_prompt_sent": False,
                    },
                    "compact": {"compacted_until": 14},
                }
            ),
            encoding="utf-8",
        )

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_compact_legacy")
        # Plan-mode side restored; compact stays at the in-memory default.
        assert rebuilt.plan_mode_active is True
        assert rebuilt._compacted_until == 0


class TestContextStatePersistence:
    """``persist_context_state`` flushes occupancy; a rebuilt node restores it."""

    def test_persist_writes_context_state_section(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_ctx1")

        node.persist_context_state(last_call_input_tokens=52_499, context_length=1_000_000)

        state_path = _state_path(node, "chat_session_ctx1")
        assert state_path.exists()
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data["context_state"] == {
            "last_call_input_tokens": 52_499,
            "context_length": 1_000_000,
            "valid": True,
        }
        # In-memory mirror updated so a same-process status-bar read is correct.
        assert node._restored_context_used == 52_499
        assert node._restored_context_length == 1_000_000

    def test_rebuilt_node_restores_context_state(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_ctx2")
        node.persist_context_state(last_call_input_tokens=12_004, context_length=200_000)

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_ctx2")
        assert rebuilt._restored_context_used == 12_004
        assert rebuilt._restored_context_length == 200_000

    def test_fresh_session_restores_zero(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_ctx_fresh")
        assert node._restored_context_used == 0
        assert node._restored_context_length == 0

    def test_persist_preserves_plan_mode_state(self, chdir_tmp, real_agent_config):
        """Writing occupancy must not disturb the session's plan-mode flags."""
        node = _make_chat_node(real_agent_config, session_id="chat_session_ctx3")
        _materialize_session(node)
        node.activate_plan_mode()
        node.persist_context_state(last_call_input_tokens=7, context_length=99)

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_ctx3")
        assert rebuilt.plan_mode_active is True  # survived the context-state write
        assert rebuilt._restored_context_used == 7
        assert rebuilt._restored_context_length == 99


class TestResetUsageCaches:
    """``_reset_usage_caches`` (invoked by clear/delete session) must zero the
    in-memory usage mirrors and drop the persisted ContextState so a status bar
    read after a reset no longer shows the previous turn's usage."""

    def test_reset_zeros_in_memory_and_removes_context_state(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_reset1")
        node.persist_context_state(last_call_input_tokens=52_499, context_length=1_000_000)
        node.running_turn_usage = object()  # stand-in for a TokenUsage snapshot

        node._reset_usage_caches()

        # In-memory mirrors zeroed.
        assert node.running_turn_usage is None
        assert node._restored_context_used == 0
        assert node._restored_context_length == 0
        # Persisted ContextState mirror removed — a rebuilt node restores zero.
        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_reset1")
        assert rebuilt._restored_context_used == 0
        assert rebuilt._restored_context_length == 0

    def test_reset_preserves_plan_mode_state(self, chdir_tmp, real_agent_config):
        """Only the usage mirror is dropped — plan mode is not usage state."""
        node = _make_chat_node(real_agent_config, session_id="chat_session_reset2")
        _materialize_session(node)
        node.activate_plan_mode()
        node.persist_context_state(last_call_input_tokens=7, context_length=99)

        node._reset_usage_caches()

        rebuilt = _make_chat_node(real_agent_config, session_id="chat_session_reset2")
        assert rebuilt.plan_mode_active is True
        assert rebuilt._restored_context_used == 0
