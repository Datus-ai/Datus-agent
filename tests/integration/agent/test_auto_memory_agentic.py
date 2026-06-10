# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Nightly coverage for the extension flow ``extension.auto_memory``.

Memory is a single 2000-byte-capped ``MEMORY.md`` per node, written exclusively
through the dedicated ``add_memory`` / ``edit_memory`` tools (see
``datus/tools/func_tool/memory_tools.py``); generic filesystem tools cannot
touch the ``.datus/memory/**`` subtree. This file covers:

1. Memory LOAD — a MEMORY.md written into a memory-enabled node's workspace
   memory dir (``{project_root}/.datus/memory/{node}/MEMORY.md``) is loaded and
   injected (writable branch) into that node's system prompt.
2. Memory inheritance — a built-in subagent that has no memory file of its own
   (``has_memory`` is False) inherits the parent's MEMORY.md in read-only
   mode when the ``inherited_memory`` contextvar is active (the path used when a
   custom subagent is launched via the ``task`` tool).
3. Workspace isolation — a sibling node without its own MEMORY.md does not pick
   up another node's memory.
4. Real-LLM smoke — a memory-enabled chat node whose MEMORY.md carries a
   distinctive instruction runs end-to-end and the loaded memory is present in
   the prompt it sends.

The add_memory/edit_memory write lifecycle (including the 2000-byte full →
prune → retry loop) is covered deterministically in
``tests/unit_tests/tools/func_tool/test_memory_tools.py`` and at the node level
in the chat / feedback acceptance tests.
"""

import os

import pytest

from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.configuration.inherited_memory_overrides import inherited_memory
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.chat_agentic_node_models import ChatNodeInput
from datus.utils.loggings import get_logger
from datus.utils.memory_loader import MEMORY_BASE_DIR, MEMORY_FILENAME, has_memory

logger = get_logger(__name__)

CHAT_MEMORY_MARKER = "Always prefer CTEs over nested subqueries in generated SQL."
CUSTOM_AGENT = "nightly_memory_agent"
CUSTOM_MEMORY_MARKER = "Revenue is computed from the net_amount column."


def _write_memory(workspace_root, node_name, content):
    """Seed a MEMORY.md for ``node_name`` under the workspace memory dir."""
    from pathlib import Path

    mem_dir = Path(workspace_root) / MEMORY_BASE_DIR / node_name
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_file = mem_dir / MEMORY_FILENAME
    mem_file.write_text(content, encoding="utf-8")
    return mem_file


def _isolated_config(nightly_agent_config, tmp_path):
    """Point the (function-scoped) config's project_root at an isolated tmp dir.

    Memory is resolved from ``project_root``; using tmp_path keeps each test's
    seeded MEMORY.md from colliding with the real project or other tests.
    ``project_root`` is a read-only property over ``_project_root``, so the
    backing field is set directly on this function-scoped config.
    """
    from pathlib import Path

    nightly_agent_config._project_root = Path(tmp_path).resolve()
    return nightly_agent_config


@pytest.mark.nightly
class TestAutoMemoryLoadAndInherit:
    """Deterministic memory load + inheritance behavior."""

    def test_chat_node_loads_own_memory_into_prompt(self, nightly_agent_config, tmp_path):
        config = _isolated_config(nightly_agent_config, tmp_path)
        _write_memory(str(tmp_path), "chat", f"# Memory\n\n## SQL style\n- {CHAT_MEMORY_MARKER}\n")

        node = ChatAgenticNode(
            node_id="nightly_mem_chat",
            description="chat memory load",
            node_type=NodeType.TYPE_CHAT,
            agent_config=config,
        )
        assert has_memory(node.get_node_name()) is True, "chat node must be memory-enabled"

        prompt = node._get_system_prompt()
        assert CHAT_MEMORY_MARKER in prompt, "Loaded chat MEMORY.md content must appear in the system prompt"
        # Writable branch renders Save instructions for memory-enabled nodes.
        assert "**Save**" in prompt, "Memory-enabled chat node must render the writable memory block"

    def test_builtin_subagent_inherits_parent_memory_readonly(self, nightly_agent_config, tmp_path):
        config = _isolated_config(nightly_agent_config, tmp_path)
        _write_memory(str(tmp_path), "chat", f"# Memory\n\n## Domain\n- {CUSTOM_MEMORY_MARKER}\n")

        # ``is_subagent=True`` so the node follows the sub-agent path; without it
        # the node counts as a main agent and resolves the shared 'chat' memory.
        node = GenSQLAgenticNode(
            node_id="nightly_mem_inherit",
            description="gen_sql inherits chat memory",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=config,
            node_name="gen_sql",
            execution_mode="workflow",
            is_subagent=True,
        )
        assert has_memory("gen_sql") is False, "built-in gen_sql must not own a memory file"

        # Without inheritance active, gen_sql renders no memory section.
        base_prompt = node._inject_memory_context("BASE PROMPT")
        assert base_prompt == "BASE PROMPT", "gen_sql must not load memory without an active inheritance override"

        # With inheritance active (the task-tool path), it reads chat's memory read-only.
        with inherited_memory("gen_sql", "chat"):
            inherited_prompt = node._inject_memory_context("BASE PROMPT")
        assert CUSTOM_MEMORY_MARKER in inherited_prompt, "Inherited parent memory content must appear read-only"
        assert "read-only inheritance from chat" in inherited_prompt, "Inherited block must be marked read-only"
        assert "**Save**" not in inherited_prompt, "Read-only inheritance must NOT render writable Save instructions"

    def test_custom_agent_memory_is_workspace_isolated(self, nightly_agent_config, tmp_path):
        config = _isolated_config(nightly_agent_config, tmp_path)
        # Seed memory only for the custom agent, not for chat.
        _write_memory(str(tmp_path), CUSTOM_AGENT, f"# Memory\n\n## Domain\n- {CUSTOM_MEMORY_MARKER}\n")

        custom_node = GenSQLAgenticNode(
            node_id="nightly_mem_custom",
            description="custom memory agent",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=config,
            node_name=CUSTOM_AGENT,
            execution_mode="workflow",
        )
        assert has_memory(custom_node.get_node_name()) is True, "custom (non-builtin) subagent must be memory-enabled"
        custom_prompt = custom_node._get_system_prompt()
        assert CUSTOM_MEMORY_MARKER in custom_prompt, "Custom agent must load its own MEMORY.md"

        chat_node = ChatAgenticNode(
            node_id="nightly_mem_chat_isolated",
            description="chat without own memory",
            node_type=NodeType.TYPE_CHAT,
            agent_config=config,
        )
        chat_prompt = chat_node._get_system_prompt()
        assert CUSTOM_MEMORY_MARKER not in chat_prompt, (
            "chat node must NOT see the custom agent's memory (per-node workspace isolation)"
        )


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestAutoMemoryRealLLM:
    """Real-LLM run of a memory-enabled node whose MEMORY.md is loaded."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_loaded_memory_present_in_real_run(self, nightly_agent_config, tmp_path):
        """N924-MEM: chat node runs end-to-end with its MEMORY.md loaded into the prompt."""
        config = _isolated_config(nightly_agent_config, tmp_path)
        _write_memory(str(tmp_path), "chat", f"# Memory\n\n## SQL style\n- {CHAT_MEMORY_MARKER}\n")

        node = ChatAgenticNode(
            node_id="nightly_mem_chat_llm",
            description="chat memory real run",
            node_type=NodeType.TYPE_CHAT,
            agent_config=config,
        )
        # The loaded memory must be wired into the system prompt the node sends.
        assert CHAT_MEMORY_MARKER in node._get_system_prompt(), "Memory must be loaded before the run"

        node.input = ChatNodeInput(
            user_message="How many schools are there in Alameda county?",
            database="california_schools",
            max_turns=15,
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2, f"Expected at least 2 actions, got {len(actions)}"
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Memory-enabled chat run should end SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
