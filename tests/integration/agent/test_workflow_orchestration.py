# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Nightly coverage for the extension flow ``extension.workflow_orchestration``.

Drives a real multi-node workflow end-to-end through ``WorkflowRunner`` against
a real LLM and the bundled california_schools SQLite database. The
``gen_sql_agentic`` builtin plan is a genuine multi-node graph
(``gen_sql -> execute_sql -> output``) whose agentic ``gen_sql`` node performs
its own schema discovery via ``db_tools.*`` — so it exercises cross-node
orchestration without depending on a prebuilt embedding/KB store (which the
``schema_linking``-first plans require and which is not provisioned in this
nightly environment).

Assertions:
- The deterministic test confirms the runner assembles the expected multi-node
  graph from the plan (no LLM needed).
- The real-LLM streaming test asserts the runner emits its lifecycle actions
  (init -> per-node execution -> completion), advances through more than one
  node, and finishes with the completion action marked SUCCESS.
"""

import argparse
import os

import pytest

from datus.agent.workflow_runner import WorkflowRunner
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.node_models import SqlTask
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _make_args(max_steps: int = 20, workflow: str = "gen_sql_agentic") -> argparse.Namespace:
    return argparse.Namespace(
        max_steps=max_steps,
        workflow=workflow,
        load_cp=None,
        debug=False,
        # Must be a concrete bool: the agentic gen_sql node (first node of this
        # plan) builds GenSQLNodeInput(plan_mode=...) which rejects None.
        plan_mode=False,
    )


def _make_sql_task() -> SqlTask:
    return SqlTask(
        id="nightly_wf_orchestration",
        datasource="bird_school",
        database_type="sqlite",
        task="How many schools are there in Alameda county?",
        database_name="california_schools",
    )


@pytest.mark.nightly
class TestWorkflowGraphAssembly:
    """Deterministic: the runner builds the expected multi-node gen_sql_agentic graph."""

    def test_gen_sql_agentic_plan_is_multi_node(self, nightly_agent_config) -> None:
        runner = WorkflowRunner(
            args=_make_args(),
            agent_config=nightly_agent_config,
            run_id="nightly-wf-assembly",
        )
        runner.initialize_workflow(_make_sql_task())

        from datus.agent.workflow import Workflow

        assert isinstance(runner.workflow, Workflow), (
            f"Runner must initialize a Workflow, got {type(runner.workflow).__name__}"
        )
        node_types = [n.type for n in runner.workflow.nodes.values()]
        assert len(node_types) >= 3, f"gen_sql_agentic plan should assemble a multi-node graph, got {node_types}"
        # The plan contains a gen_sql node, executes SQL, and ends in an output node.
        type_strs = [str(t) for t in node_types]
        assert any("gen_sql" in t for t in type_strs), (
            f"gen_sql_agentic plan should contain a gen_sql node, got {type_strs}"
        )
        assert any("execute_sql" in t for t in type_strs), (
            f"gen_sql_agentic plan should contain an execute_sql node, got {type_strs}"
        )
        assert any("output" in t for t in type_strs), (
            f"gen_sql_agentic plan should contain an output node, got {type_strs}"
        )


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestWorkflowOrchestrationRealLLM:
    """Real-LLM multi-node workflow execution via WorkflowRunner."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_gen_sql_agentic_workflow_completes(self, nightly_agent_config) -> None:
        """N924-WF: run a multi-node gen_sql_agentic workflow end-to-end with a real LLM."""
        runner = WorkflowRunner(
            args=_make_args(max_steps=20),
            agent_config=nightly_agent_config,
            run_id="nightly-wf-real",
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in runner.run_stream(
            sql_task=_make_sql_task(),
            check_storage=False,
            action_history_manager=action_manager,
        ):
            actions.append(action)

        action_ids = [a.action_id for a in actions]
        assert "workflow_initialization" in action_ids, f"Missing init action, got: {action_ids}"
        assert "workflow_completion" in action_ids, f"Missing completion action, got: {action_ids}"

        # The runner emits one node_execution_* action per node it drives; a
        # multi-node graph must produce more than one.
        node_exec_actions = [a for a in actions if a.action_type == "node_execution"]
        assert len(node_exec_actions) >= 2, (
            f"Multi-node workflow should execute >=2 nodes, got {len(node_exec_actions)}: "
            f"{[a.action_id for a in node_exec_actions]}"
        )

        # The final completion action must report success.
        completion_actions = [a for a in actions if a.action_id == "workflow_completion"]
        final_completion = completion_actions[-1]
        assert final_completion.status == ActionStatus.SUCCESS, (
            f"Workflow completion should be SUCCESS, got {final_completion.status}: {final_completion.output}"
        )
        assert final_completion.output.get("workflow_saved") is True, (
            f"Completed workflow should be persisted, output: {final_completion.output}"
        )
