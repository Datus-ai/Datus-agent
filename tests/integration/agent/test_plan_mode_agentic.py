# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Integration tests for Plan Mode (CLI ``cli/plan_mode.md`` core flow).

Plan Mode is a documented core CLI behavior with only unit coverage before this
suite. These tests drive the real plan-mode lifecycle end-to-end with a real
LLM: the node authors a step-by-step plan, confirms it, and (with
``auto_execute_plan``) executes it without blocking on interactive confirmation.

The driver mirrors what ``PrintModeRunner`` does for the ``--print --plan-mode``
flag (create node → ``create_node_input(plan_mode=True)`` → set
``auto_execute_plan`` → ``execute_stream_with_interactions``), so this also
guards the headless plan-mode entry point.
"""

import pytest

from datus.agent.node.node_factory import create_interactive_node, create_node_input
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Tools that prove the plan was authored and/or confirmed during the run.
PLAN_TOOL_ACTION_TYPES = {"todo_write", "todo_update", "confirm_plan"}


@pytest.mark.nightly
@pytest.mark.product_e2e
class TestPlanModeAgentic:
    """Plan Mode end-to-end with a real LLM."""

    def test_plan_mode_tools_registered(self, nightly_agent_config):
        """The chat node exposes the plan-mode tool surface (confirm_plan + todos)."""
        node = create_interactive_node(
            None,
            nightly_agent_config,
            node_id_suffix="_plan_tools",
            execution_mode="workflow",
        )

        tool_names = {tool.name for tool in node.tools}
        assert "confirm_plan" in tool_names, f"Missing confirm_plan tool, got: {sorted(tool_names)}"
        assert "todo_write" in tool_names, f"Missing todo_write tool, got: {sorted(tool_names)}"

    @pytest.mark.asyncio
    async def test_plan_mode_generates_and_executes_plan(self, nightly_agent_config):
        """Plan mode authors a plan, confirms it, and auto-executes to a SUCCESS.

        Assertions are behavior-based (plan file allocated, a plan-authoring or
        confirm tool fired, terminal SUCCESS) rather than tied to the LLM's
        exact wording, so the test is not brittle across models.
        """
        node = create_interactive_node(
            None,
            nightly_agent_config,
            node_id_suffix="_plan_exec",
            execution_mode="workflow",
        )

        node_input = create_node_input(
            user_message=(
                "Work in plan mode. First write a short step-by-step plan for answering: "
                "'Which county has the most schools in the california_schools database?'. "
                "Then execute the plan and give the answer."
            ),
            node=node,
            plan_mode=True,
        )
        # Headless run: auto-approve the plan so execution does not block on an
        # interactive confirmation prompt (parity with print mode).
        assert hasattr(node_input, "auto_execute_plan"), "chat node input should support auto_execute_plan"
        node_input.auto_execute_plan = True
        node.input = node_input

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream_with_interactions(action_manager):
            actions.append(action)
            logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        # Plan mode must have been activated for this turn: a plan file is
        # allocated on activation and never cleared, even after confirm_plan
        # deactivates the flag at the end of the turn.
        assert node.plan_file_path, "Plan mode should allocate a plan file when input.plan_mode is set"

        # The agent must have actually exercised the plan-mode workflow (authored
        # the plan and/or confirmed it), not just answered directly.
        action_types = [a.action_type for a in actions]
        assert any(a.action_type in PLAN_TOOL_ACTION_TYPES and a.status == ActionStatus.SUCCESS for a in actions), (
            f"Plan-mode run should successfully author/confirm a plan via {sorted(PLAN_TOOL_ACTION_TYPES)}, "
            f"got action types: {action_types}"
        )

        # First action is the USER request entering the loop.
        assert actions[0].role == ActionRole.USER
        assert actions[0].status == ActionStatus.PROCESSING

        # Terminal action is a successful completion.
        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
