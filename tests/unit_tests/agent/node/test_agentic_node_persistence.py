# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for AgenticNode plan-mode persistence layer — CI tier."""

import json

import pytest

from datus.storage.session_state import PlanModeState


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
    """``activate_plan_mode`` / ``deactivate_plan_mode`` flush to disk."""

    def test_activate_writes_state_file(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config)
        node.session_id = "chat_session_aaaa"  # setter restore is a no-op (no file)

        node.activate_plan_mode()

        state_path = _state_path(node, "chat_session_aaaa")
        assert state_path.exists()
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert data["plan_mode_active"] is True
        assert data["plan_file_path"] == node.plan_file_path
        assert data["workflow_prompt_sent"] is False

    def test_deactivate_writes_state_file(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config)
        node.session_id = "chat_session_bbbb"
        node.activate_plan_mode()
        node.deactivate_plan_mode()

        state_path = _state_path(node, "chat_session_bbbb")
        data = json.loads(state_path.read_text(encoding="utf-8"))
        # plan_mode_active flipped back to False; plan_file_path is preserved.
        assert data["plan_mode_active"] is False
        assert data["plan_file_path"] == node.plan_file_path
        assert data["workflow_prompt_sent"] is False

    def test_activate_without_session_id_skips_disk(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config)  # no session_id
        assert node.session_id is None

        node.activate_plan_mode()
        # No state files anywhere under data/state — disk write was skipped.
        state_dir = _path_manager(node).project_data_dir / "state"
        assert not state_dir.exists() or not any(state_dir.iterdir())


class TestSessionIdSetterTriggersRestore:
    """Assigning ``session_id`` to an existing node rehydrates plan-mode."""

    def test_setter_restores_persisted_state(self, chdir_tmp, real_agent_config):
        # We need to know the state path before constructing the node, so
        # build a throwaway node just to resolve the project-scoped path.
        anchor = _make_chat_node(real_agent_config)
        plan_file = "./.datus/plans/feed.md"
        PlanModeState(
            plan_mode_active=True,
            plan_file_path=plan_file,
            workflow_prompt_sent=True,
        ).save(_state_path(anchor, "chat_session_cccc"))

        # Build a fresh node without session_id, then assign.
        node = _make_chat_node(real_agent_config)
        assert node.plan_mode_active is False
        assert node.plan_file_path is None

        node.session_id = "chat_session_cccc"

        assert node.plan_mode_active is True
        assert node.plan_file_path == plan_file
        assert node.workflow_prompt_sent is True
        assert node._plan_just_confirmed is False  # one-shot flag never restored

    def test_constructor_session_id_restores(self, chdir_tmp, real_agent_config):
        anchor = _make_chat_node(real_agent_config)
        PlanModeState(
            plan_mode_active=True,
            plan_file_path="./.datus/plans/init.md",
            workflow_prompt_sent=False,
        ).save(_state_path(anchor, "chat_session_dddd"))

        node = _make_chat_node(real_agent_config, session_id="chat_session_dddd")

        assert node.plan_mode_active is True
        assert node.plan_file_path == "./.datus/plans/init.md"

    def test_setter_no_state_file_keeps_defaults(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config)
        node.session_id = "chat_session_unknown"
        # No file present → defaults remain (False/None/False).
        assert node.plan_mode_active is False
        assert node.plan_file_path is None
        assert node.workflow_prompt_sent is False

    def test_setter_idempotent_for_same_value(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config)
        node.session_id = "chat_session_xxxx"
        node.activate_plan_mode()
        original_path = node.plan_file_path

        # Re-assigning the same value should NOT trigger a restore that
        # wipes the in-memory state we just set.
        node.session_id = "chat_session_xxxx"
        assert node.plan_mode_active is True
        assert node.plan_file_path == original_path

    def test_setter_to_none_does_not_restore(self, chdir_tmp, real_agent_config):
        node = _make_chat_node(real_agent_config, session_id="chat_session_yyyy")
        node.activate_plan_mode()
        # Clearing back to None must not crash and must not load any state.
        node.session_id = None
        assert node.session_id is None
