# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for FeedbackAgenticNode.

Tests cover:
- Node initialization and tool setup
- Session copy mechanism
- Streaming execution with MockLLMModel
- Input validation
- Storage info extraction
- Node factory integration

Design principle: NO mock except LLM.
- Real AgentConfig (from conftest `real_agent_config`)
- Real Tools (FilesystemFuncTool)
- Real PromptManager (using built-in templates)
- The ONLY mock: LLMBaseModel.create_model -> MockLLMModel (via `mock_llm_create`)
"""

import json

import pytest

from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.feedback_agentic_node_models import FeedbackNodeInput, FeedbackNodeResult
from tests.unit_tests.mock_llm_model import build_simple_response

# ---------------------------------------------------------------------------
# Schema Model Tests
# ---------------------------------------------------------------------------


class TestFeedbackNodeModels:
    """Tests for FeedbackNodeInput and FeedbackNodeResult."""

    def test_input_minimal(self):
        inp = FeedbackNodeInput(user_message="analyze and archive")
        assert inp.user_message == "analyze and archive"
        assert inp.source_session_id is None
        assert inp.database is None

    def test_input_full(self):
        inp = FeedbackNodeInput(
            user_message="analyze",
            source_session_id="chat_session_abc123",
            database="test_db",
        )
        assert inp.source_session_id == "chat_session_abc123"
        assert inp.database == "test_db"

    def test_result_minimal(self):
        result = FeedbackNodeResult(success=True, response="Done")
        assert result.items_saved == 0
        assert result.storage_summary is None
        assert result.tokens_used == 0

    def test_result_full(self):
        result = FeedbackNodeResult(
            success=True,
            response="Archived 3 items",
            items_saved=3,
            storage_summary={"ext_knowledge": 2, "sql_summary": 1},
            tokens_used=1500,
        )
        assert result.items_saved == 3
        assert result.storage_summary["ext_knowledge"] == 2

    def test_result_error(self):
        result = FeedbackNodeResult(
            success=False,
            error="Template not found",
            response="Sorry, error occurred.",
        )
        assert result.success is False
        assert result.error == "Template not found"


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestFeedbackAgenticNodeInit:
    """Tests for FeedbackAgenticNode initialization."""

    def test_node_name(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.get_node_name() == "feedback"
        assert node.configured_node_name == "feedback"

    def test_inherits_agentic_node(self, real_agent_config, mock_llm_create):
        from datus.agent.node.agentic_node import AgenticNode
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert isinstance(node, AgenticNode)

    def test_node_id(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.id == "feedback_node"

    def test_node_type(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.type == NodeType.TYPE_FEEDBACK

    def test_setup_tools_includes_filesystem(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_setup_tools_includes_task(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "task" in tool_names

    def test_workflow_mode_no_ask_user(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" not in tool_names

    def test_interactive_mode_has_ask_user(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="interactive")
        tool_names = [tool.name for tool in node.tools]
        assert "ask_user" in tool_names

    def test_max_turns_default(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.max_turns == 30

    def test_execution_mode_stored(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.execution_mode == "workflow"


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.nightly
class TestFeedbackAgenticNodeExecution:
    """Tests for FeedbackAgenticNode streaming execution."""

    @pytest.mark.asyncio
    async def test_simple_response(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("I analyzed the conversation and found nothing worth archiving."),
            ]
        )

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze and archive this conversation")

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING
        assert actions[-1].status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_input_not_set_raises(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.utils.exceptions import DatusException, ErrorCode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = None

        action_manager = ActionHistoryManager()
        with pytest.raises(DatusException) as excinfo:
            async for _ in node.execute_stream(action_manager):
                pass
        assert excinfo.value.code == ErrorCode.COMMON_FIELD_REQUIRED

    @pytest.mark.asyncio
    async def test_execution_interrupted_propagates(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.cli.execution_state import ExecutionInterrupted

        async def _raise_interrupted(*args, **kwargs):
            raise ExecutionInterrupted("User pressed ESC")
            yield  # noqa: makes this an async generator

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise_interrupted

        action_manager = ActionHistoryManager()
        with pytest.raises(ExecutionInterrupted):
            async for _ in node.execute_stream(action_manager):
                pass

    @pytest.mark.asyncio
    async def test_execution_error_yields_error_action(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        async def _raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")
            yield  # noqa

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise_error

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)

        assert len(actions) >= 2
        last = actions[-1]
        assert last.status == ActionStatus.FAILED
        assert last.action_type == "error"

    @pytest.mark.asyncio
    async def test_result_set_on_success(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        mock_llm_create.reset(
            responses=[
                build_simple_response("Feedback analysis complete."),
            ]
        )

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert node.result is not None
        assert isinstance(node.result, FeedbackNodeResult)
        assert node.result.success is True

    @pytest.mark.asyncio
    async def test_result_set_on_error(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")
            yield  # noqa

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _raise

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert node.result is not None
        assert node.result.success is False
        assert "boom" in node.result.error


# ---------------------------------------------------------------------------
# Storage Info Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractStorageInfo:
    """Tests for _extract_storage_info method."""

    def test_no_actions_returns_zero(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        items_saved, summary = node._extract_storage_info([])
        assert items_saved == 0
        assert summary is None

    def test_counts_successful_task_actions(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        actions = [
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="task",
                messages="Tool call: task",
                input_data={"arguments": json.dumps({"type": "gen_ext_knowledge", "prompt": "test"})},
                status=ActionStatus.SUCCESS,
            ),
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="task",
                messages="Tool call: task",
                input_data={"arguments": json.dumps({"type": "gen_sql_summary", "prompt": "test"})},
                status=ActionStatus.SUCCESS,
            ),
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="read_file",
                messages="Tool call: read_file",
                input_data={},
                status=ActionStatus.SUCCESS,
            ),
        ]

        items_saved, summary = node._extract_storage_info(actions)
        assert items_saved == 2
        assert summary == {"ext_knowledge": 1, "sql_summary": 1}

    def test_ignores_stale_instance_actions(self, real_agent_config, mock_llm_create):
        """_extract_storage_info must count from the passed-in list, not self.actions,
        so a reused node instance doesn't report stale counts from previous runs."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.action_history import ActionHistory

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.actions = [
            ActionHistory.create_action(
                role=ActionRole.TOOL,
                action_type="task",
                messages="Previous run task",
                input_data={"arguments": json.dumps({"type": "gen_ext_knowledge", "prompt": "prev"})},
                status=ActionStatus.SUCCESS,
            ),
        ]
        items_saved, summary = node._extract_storage_info([])
        assert items_saved == 0
        assert summary is None

    @pytest.mark.asyncio
    async def test_stream_populates_items_saved(self, real_agent_config, mock_llm_create):
        """Regression: the current stream's task() tool calls must be counted.

        Previously _extract_storage_info ran before self.actions was populated
        from action_history_manager, so items_saved was always 0 in practice —
        even though the per-method unit test passed because it pre-seeded
        self.actions manually.
        """
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.schemas.action_history import ActionHistory

        async def _stream_with_task_actions(*args, **kwargs):
            ahm = kwargs.get("action_history_manager")
            for sub_type in ("gen_ext_knowledge", "gen_sql_summary"):
                act = ActionHistory.create_action(
                    role=ActionRole.TOOL,
                    action_type="task",
                    messages="Tool call: task",
                    input_data={
                        "function_name": "task",
                        "arguments": json.dumps({"type": sub_type, "prompt": "x"}),
                    },
                    output_data={"response": "ok"},
                    status=ActionStatus.SUCCESS,
                )
                if ahm is not None:
                    ahm.add_action(act)
                yield act

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(user_message="Analyze")
        mock_llm_create.generate_with_tools_stream = _stream_with_task_actions

        action_manager = ActionHistoryManager()
        async for _ in node.execute_stream(action_manager):
            pass

        assert node.result is not None
        assert node.result.success is True
        assert node.result.items_saved == 2
        assert node.result.storage_summary == {"ext_knowledge": 1, "sql_summary": 1}


# ---------------------------------------------------------------------------
# Memory Enablement Tests
# ---------------------------------------------------------------------------


class TestMemoryEnabled:
    """Verify the ``memory_enabled`` attribute on AgenticNode gates Auto Memory injection."""

    def test_chat_node_defaults_to_enabled(self, real_agent_config, mock_llm_create):
        from datus.agent.node.chat_agentic_node import ChatAgenticNode
        from datus.configuration.node_type import NodeType

        node = ChatAgenticNode(
            node_id="test_chat_mem",
            description="Test chat memory",
            node_type=NodeType.TYPE_CHAT,
            agent_config=real_agent_config,
        )
        assert node.memory_enabled is True

    def test_feedback_node_defaults_to_disabled(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        assert node.memory_enabled is False

    def test_builtin_subagent_skips_memory_injection(self, real_agent_config, mock_llm_create):
        """Built-in subagents (memory_enabled=False) must NOT get the Auto Memory
        section in their system prompt — only chat and custom agents do."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_gensql_mem",
            description="Test gensql memory",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gen_sql",
            execution_mode="workflow",
        )
        assert node.memory_enabled is False

        prompt = node._inject_memory_context("BASE PROMPT")
        assert prompt == "BASE PROMPT"
        assert "## Auto Memory" not in prompt

    def test_explicit_override_forces_injection(self, real_agent_config, mock_llm_create):
        """Passing override_node_name bypasses self.memory_enabled (feedback path)."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_gensql_override",
            description="Test gensql override",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gen_sql",
            execution_mode="workflow",
        )
        assert node.memory_enabled is False

        prompt = node._inject_memory_context("BASE PROMPT", override_node_name="chat")
        assert "## Auto Memory" in prompt
        assert ".datus/memory/chat" in prompt

    def test_explicit_memory_enabled_override(self, real_agent_config, mock_llm_create):
        """Passing memory_enabled=True to __init__ overrides has_memory() default."""
        from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
        from datus.configuration.node_type import NodeType

        node = GenSQLAgenticNode(
            node_id="test_gensql_mem_opt_in",
            description="Test gensql opt-in memory",
            node_type=NodeType.TYPE_GENSQL,
            agent_config=real_agent_config,
            node_name="gen_sql",
            execution_mode="workflow",
        )
        # Default resolved via has_memory("gen_sql") → False
        assert node.memory_enabled is False

        # Flipping the attribute post-init should re-enable injection.
        node.memory_enabled = True
        prompt = node._inject_memory_context("BASE PROMPT")
        assert "## Auto Memory" in prompt
        assert ".datus/memory/gen_sql" in prompt


# ---------------------------------------------------------------------------
# NodeType and Node Factory Tests
# ---------------------------------------------------------------------------


class TestFeedbackNodeType:
    """Tests for NodeType integration with feedback."""

    def test_type_input_feedback(self):
        from datus.configuration.node_type import NodeType

        inp = NodeType.type_input(
            NodeType.TYPE_FEEDBACK,
            {"user_message": "analyze conversation"},
        )
        assert isinstance(inp, FeedbackNodeInput)
        assert inp.user_message == "analyze conversation"

    def test_feedback_in_action_types(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_FEEDBACK in NodeType.ACTION_TYPES

    def test_feedback_in_descriptions(self):
        from datus.configuration.node_type import NodeType

        assert NodeType.TYPE_FEEDBACK in NodeType.NODE_TYPE_DESCRIPTIONS
        desc = NodeType.get_description(NodeType.TYPE_FEEDBACK)
        assert "feedback" in desc.lower() or "archival" in desc.lower()

    def test_node_factory_creates_feedback(self, real_agent_config, mock_llm_create):
        from datus.agent.node import Node
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        node = Node.new_instance(
            node_id="test_feedback",
            description="Test feedback factory",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=None,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, FeedbackAgenticNode)
        assert node.execution_mode == "workflow"

    def test_node_factory_with_input_data(self, real_agent_config, mock_llm_create):
        from datus.agent.node import Node
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.configuration.node_type import NodeType

        input_data = FeedbackNodeInput(user_message="test input")
        node = Node.new_instance(
            node_id="test_feedback",
            description="Test feedback factory",
            node_type=NodeType.TYPE_FEEDBACK,
            input_data=input_data,
            agent_config=real_agent_config,
            tools=[],
        )
        assert isinstance(node, FeedbackAgenticNode)
        assert node.input is not None
        assert node.input.user_message == "test input"


# ---------------------------------------------------------------------------
# Constants and Memory Tests
# ---------------------------------------------------------------------------


class TestFeedbackConstants:
    """Tests for feedback registration in constants and memory."""

    def test_feedback_in_sys_sub_agents(self):
        """feedback is a reserved system name (in SYS_SUB_AGENTS) even though
        it is a top-level node and not a task()-delegatable subagent."""
        from datus.utils.constants import SYS_SUB_AGENTS

        assert "feedback" in SYS_SUB_AGENTS

    def test_feedback_has_no_own_memory(self):
        """The feedback node updates the caller's memory, not its own."""
        from datus.utils.memory_loader import has_memory

        assert has_memory("feedback") is False

    def test_feedback_in_init_all(self):
        from datus.agent.node import __all__

        assert "FeedbackAgenticNode" in __all__


# ---------------------------------------------------------------------------
# System Prompt / Memory Injection Tests
# ---------------------------------------------------------------------------


class TestFeedbackSystemPrompt:
    """Tests verifying that feedback reuses the standard _inject_memory_context
    pipeline to attach the CALLER's memory (not its own)."""

    def test_feedback_system_template_has_no_caller_variables(self):
        """feedback_system_1.0.j2 must not reference caller_node_name/caller_memory_dir.

        Memory path/conventions are injected via the shared memory_context template
        through _finalize_system_prompt; duplicating them in feedback_system would
        re-introduce the two-path problem this refactor eliminated.
        """
        from pathlib import Path

        import datus

        template_path = Path(datus.__file__).parent / "prompts" / "prompt_templates" / "feedback_system_1.0.j2"
        content = template_path.read_text(encoding="utf-8")
        assert "caller_node_name" not in content
        assert "caller_memory_dir" not in content

    def test_feedback_system_prompt_injects_caller_memory(self, real_agent_config, mock_llm_create):
        """When feedback renders its system prompt, the caller's memory is injected
        via the standard memory_context template under the Auto Memory section."""
        from pathlib import Path

        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.caller_node_name = "chat"
        node.input = FeedbackNodeInput(user_message="Analyze and archive")

        workspace_root = Path(node._resolve_workspace_root())
        caller_memory_dir = workspace_root / ".datus" / "memory" / "chat"
        caller_memory_dir.mkdir(parents=True, exist_ok=True)
        memory_content = "# Chat memory\n- user prefers DuckDB over SQLite for local analytics"
        (caller_memory_dir / "MEMORY.md").write_text(memory_content, encoding="utf-8")

        prompt = node._get_system_prompt()

        assert "## Auto Memory" in prompt
        # get_memory_dir returns a relative path; it should appear verbatim.
        assert ".datus/memory/chat" in prompt
        # The truncated memory content itself is embedded in <memory>…</memory>.
        assert "user prefers DuckDB over SQLite for local analytics" in prompt
        # When real memory content is present, the empty placeholder must NOT appear.
        assert "empty — no memories saved yet" not in prompt

    def test_feedback_system_prompt_injects_memory_for_caller_without_default_memory(
        self, real_agent_config, mock_llm_create
    ):
        """Memory context is injected unconditionally: even when the caller node
        (e.g. ``gen_sql``) does NOT opt into memory by default (``has_memory`` is
        False for it), feedback still renders the Auto Memory section pointing at
        that caller's memory directory so feedback can create/update it."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode
        from datus.utils.memory_loader import has_memory

        # Precondition: gen_sql is a built-in subagent without default memory.
        assert has_memory("gen_sql") is False

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.caller_node_name = "gen_sql"
        node.input = FeedbackNodeInput(user_message="Analyze and archive")

        prompt = node._get_system_prompt()

        assert "## Auto Memory" in prompt
        assert ".datus/memory/gen_sql" in prompt

    def test_feedback_system_prompt_renders_empty_placeholder_when_caller_file_missing(
        self, real_agent_config, mock_llm_create
    ):
        """No MEMORY.md for the caller → Auto Memory section still renders a
        <memory> block with an explicit empty placeholder, so the model knows
        memory was checked and is currently empty (rather than silently
        omitting the block and leaving the model to guess)."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.caller_node_name = "chat"
        node.input = FeedbackNodeInput(user_message="Analyze and archive")

        prompt = node._get_system_prompt()

        assert "## Auto Memory" in prompt
        assert ".datus/memory/chat" in prompt
        # The <memory> block is always rendered, with an explicit empty marker
        # when no MEMORY.md has been created yet.
        assert "<memory>" in prompt
        assert "</memory>" in prompt
        assert "empty — no memories saved yet" in prompt

    def test_feedback_system_prompt_renders_empty_placeholder_when_caller_file_is_empty(
        self, real_agent_config, mock_llm_create
    ):
        """Caller MEMORY.md exists but is empty → same empty placeholder as missing.
        From the model's perspective there is no actionable difference between
        'file absent' and 'file present but empty', so both share one branch."""
        from pathlib import Path

        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.caller_node_name = "chat"
        node.input = FeedbackNodeInput(user_message="Analyze and archive")

        workspace_root = Path(node._resolve_workspace_root())
        caller_memory_dir = workspace_root / ".datus" / "memory" / "chat"
        caller_memory_dir.mkdir(parents=True, exist_ok=True)
        (caller_memory_dir / "MEMORY.md").write_text("", encoding="utf-8")

        prompt = node._get_system_prompt()

        assert "## Auto Memory" in prompt
        assert "<memory>" in prompt
        assert "empty — no memories saved yet" in prompt


# ---------------------------------------------------------------------------
# Caller Resolution Tests
# ---------------------------------------------------------------------------


class TestResolveCallerNodeName:
    """Tests for FeedbackAgenticNode._resolve_caller_node_name().

    The resolver reads the explicit ``caller_node_name`` attribute (set by the
    CLI on node switch) and defaults to ``"chat"`` when no caller was set. It
    no longer parses the ``source_session_id`` prefix — that coupling to the
    session-id encoding has been removed.
    """

    def test_uses_caller_node_name_attribute(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.caller_node_name = "gen_sql"

        assert node._resolve_caller_node_name() == "gen_sql"

    def test_defaults_to_chat_when_unset(self, real_agent_config, mock_llm_create):
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")

        assert node.caller_node_name is None
        assert node._resolve_caller_node_name() == "chat"

    def test_ignores_source_session_id(self, real_agent_config, mock_llm_create):
        """Even when ``input.source_session_id`` looks like ``gen_sql_session_*``,
        the resolver must return the default (``"chat"``) unless the explicit
        ``caller_node_name`` attribute is set."""
        from datus.agent.node.feedback_agentic_node import FeedbackAgenticNode

        node = FeedbackAgenticNode(agent_config=real_agent_config, execution_mode="workflow")
        node.input = FeedbackNodeInput(
            user_message="Analyze",
            source_session_id="gen_sql_session_deadbeef",
        )

        # source_session_id is now advisory and used only for session-copy in
        # execute_stream; it no longer influences caller resolution.
        assert node._resolve_caller_node_name() == "chat"
