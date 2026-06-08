# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the read-only memory inheritance branch of ``_inject_memory_context``.

Built-in subagents (which own no memory of their own) launched via SubAgentTaskTool now
read their parent's MEMORY.md when ``inherited_memory(...)`` is active in the
contextvar. The injected block is read-only and must not contain the writable
"Save" instructions that the writable branch renders for ``chat`` / custom
subagents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datus.configuration.inherited_memory_overrides import inherited_memory
from datus.utils.memory_loader import has_memory


def _write_chat_memory(real_agent_config, content: str) -> Path:
    """Seed a parent (chat) MEMORY.md inside the fixture project root."""
    workspace_root = Path(real_agent_config.project_root)
    chat_dir = workspace_root / ".datus" / "memory" / "chat"
    chat_dir.mkdir(parents=True, exist_ok=True)
    memory_file = chat_dir / "MEMORY.md"
    memory_file.write_text(content, encoding="utf-8")
    return memory_file


def _new_gen_sql_node(real_agent_config):
    from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
    from datus.configuration.node_type import NodeType

    # ``is_subagent=True`` so the node follows the sub-agent path (read-only
    # inherited memory, no write tools) — the scenario this file exercises.
    return GenSQLAgenticNode(
        node_id="test_gen_sql_inherit",
        description="Test inherited memory for gen_sql",
        node_type=NodeType.TYPE_GEN_SQL,
        agent_config=real_agent_config,
        node_name="gen_sql",
        execution_mode="workflow",
        is_subagent=True,
    )


@pytest.mark.ci
class TestInheritedMemoryInjection:
    """Behavior of the new read-only inherited memory branch."""

    def test_builtin_with_inherited_memory_renders_readonly_block(self, real_agent_config, mock_llm_create):
        _write_chat_memory(
            real_agent_config,
            "## Profile\n- User prefers concise SQL comments.\n",
        )
        node = _new_gen_sql_node(real_agent_config)
        assert has_memory(node.get_node_name()) is False

        with inherited_memory("gen_sql", "chat"):
            prompt = node._inject_memory_context("BASE PROMPT")

        # Read-only header carries the originating agent name. The parent's
        # memory is inlined in full (single file, no topic-file path to render).
        assert "## Memory (read-only inheritance from chat)" in prompt
        # The seeded chat memory content shows up.
        assert "User prefers concise SQL comments." in prompt
        # Read-only branch must NOT include the writable "Save" instructions.
        assert "**Save**" not in prompt
        # Explicit read-only enforcement clause is present.
        assert "Read-only" in prompt

    def test_builtin_without_inherited_returns_base_prompt(self, real_agent_config, mock_llm_create):
        node = _new_gen_sql_node(real_agent_config)
        assert has_memory(node.get_node_name()) is False

        # No contextvar push — current behavior preserved.
        prompt = node._inject_memory_context("BASE PROMPT")
        assert prompt == "BASE PROMPT"
        assert "## Memory" not in prompt

    def test_inherited_does_not_affect_chat_node(self, real_agent_config, mock_llm_create):
        """Pushing inherited override for gen_sql must not change chat's prompt."""
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.configuration.node_type import NodeType

        _write_chat_memory(real_agent_config, "## Profile\n- chat-only fact\n")

        chat = ChatAgenticNode(
            node_id="test_chat_unaffected",
            description="chat",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        assert has_memory(chat.get_node_name()) is True

        with inherited_memory("gen_sql", "chat"):
            prompt = chat._inject_memory_context("BASE PROMPT")

        # Chat renders its OWN writable memory block (Save/Don't save), not the
        # read-only branch.
        assert "## Memory" in prompt
        assert "**Save**" in prompt
        assert "read-only inheritance" not in prompt

    def test_empty_parent_memory_short_circuits(self, real_agent_config, mock_llm_create):
        """If the parent has no MEMORY.md (or it is empty), do not render the read-only block."""
        node = _new_gen_sql_node(real_agent_config)

        # Do NOT seed chat memory.
        with inherited_memory("gen_sql", "chat"):
            prompt = node._inject_memory_context("BASE PROMPT")

        assert prompt == "BASE PROMPT"
        assert "read-only inheritance" not in prompt

    def test_explicit_override_node_name_takes_precedence(self, real_agent_config, mock_llm_create):
        """``override_node_name`` (feedback path) wins over inherited contextvar.

        Feedback intentionally writes the caller's memory, so it must keep the
        writable branch even when inherited_memory happens to be active.
        """
        _write_chat_memory(real_agent_config, "## Profile\n- shared fact\n")
        node = _new_gen_sql_node(real_agent_config)

        with inherited_memory("gen_sql", "chat"):
            prompt = node._inject_memory_context("BASE PROMPT", override_node_name="chat")

        # Writable branch path (Save instructions present, no read-only header).
        assert "## Memory" in prompt
        assert "**Save**" in prompt
        assert "read-only inheritance" not in prompt


@pytest.mark.ci
class TestMemoryToolMounting:
    """The new rule: main agents mount add_memory/edit_memory (built-in → shared
    'chat'); sub-agents never mount them."""

    def test_builtin_main_agent_mounts_memory_tool_bound_to_chat(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_gen_sql_main_mount",
            description="gen_sql as main agent",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=real_agent_config,
            node_name="gen_sql",
            execution_mode="workflow",
        )
        node._ensure_memory_tool_in_tools()

        assert node.memory_func_tool.memory_node == "chat"
        tool_names = {t.name for t in node.tools}
        assert {"add_memory", "edit_memory"}.issubset(tool_names)

    def test_subagent_does_not_mount_memory_tool(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_gen_sql_sub_mount",
            description="gen_sql as sub-agent",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=real_agent_config,
            node_name="gen_sql",
            execution_mode="workflow",
            is_subagent=True,
        )
        node._ensure_memory_tool_in_tools()

        assert node.memory_func_tool is None
        tool_names = {t.name for t in node.tools}
        assert "add_memory" not in tool_names
        assert "edit_memory" not in tool_names

    def test_custom_main_agent_mounts_memory_tool_bound_to_own_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_custom_main_mount",
            description="custom agent as main",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=real_agent_config,
            node_name="finance_agent",
            execution_mode="workflow",
        )
        node._ensure_memory_tool_in_tools()

        assert node.memory_func_tool.memory_node == "finance_agent"
