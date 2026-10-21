# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""OntologyAnalyst end to end with a real LLM on the Dosi aviation fixture.

The questions come from the Dosi ontology benchmark and are asked the way a
business user would: none names a metric, field or concept path. Each probes
something the ontology settles: a two-hop role plus closed value sets, and the
same metric through two different airport roles. Expected values trace to
``tests/data/semantic_models/aviation/seed.sql``.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from datus.agent.node.ontology_analyst_agentic_node import OntologyAnalystAgenticNode
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.ontology_analyst_agentic_node_models import OntologyAnalystNodeInput
from datus.tools.semantic_tools.dosi.engine import binding_supports_ontology
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

FIXTURE_ROOT = Path(__file__).parents[2] / "data" / "semantic_models"
MODEL_DIR = FIXTURE_ROOT / "aviation"
ONTOLOGY_DIR = FIXTURE_ROOT / "ontology" / "aviation"

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(
        not binding_supports_ontology(),
        reason="installed dosi-engine binding has no Engine.describe/paths",
    ),
]


@pytest.fixture
def ontology_agent_config(tmp_path, monkeypatch):
    """The acceptance config (real LLM) with its datasource pointed at the aviation model, ontology and seed.

    Built on a DuckDB datasource rather than ``nightly_agent_config``, which
    needs the BIRD benchmark checkout this suite does not use.
    """
    from tests.conftest import load_acceptance_config

    nightly_agent_config = load_acceptance_config(datasource="local_duckdb")
    nightly_agent_config.agentic_nodes = copy.deepcopy(nightly_agent_config.agentic_nodes)
    database = tmp_path / "aviation.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute((MODEL_DIR / "seed.sql").read_text(encoding="utf-8"))

    path_manager = nightly_agent_config.path_manager
    monkeypatch.setattr(path_manager, "semantic_model_path", lambda datasource: MODEL_DIR)
    monkeypatch.setattr(path_manager, "ontology_path", lambda datasource: ONTOLOGY_DIR)
    duckdb_config = SimpleNamespace(to_dict=lambda: {"type": "duckdb", "uri": str(database)})
    monkeypatch.setattr(nightly_agent_config, "current_db_config", lambda datasource=None: duckdb_config)
    return nightly_agent_config


def _build_node(agent_config) -> OntologyAnalystAgenticNode:
    node = OntologyAnalystAgenticNode(
        node_id="ontology_analyst_itest",
        description="OntologyAnalyst integration test",
        node_type=NodeType.TYPE_ONTOLOGY_ANALYST,
        agent_config=agent_config,
        node_name="ontology_analyst",
        execution_mode="workflow",
    )
    assert node.startup_error is None, f"OntologyAnalyst failed to start: {node.startup_error}"
    return node


async def _ask(node, question: str) -> tuple[str, list]:
    node.input = OntologyAnalystNodeInput(user_message=question)
    actions = []
    async for action in node.execute_stream(ActionHistoryManager()):
        actions.append(action)
        logger.info("Action: role=%s status=%s type=%s", action.role, action.status, action.action_type)
    assert actions[-1].status == ActionStatus.SUCCESS, f"Last action failed: {actions[-1].output}"
    # Only the final answer: the node result also carries action_history, whose
    # tool outputs would satisfy `expect` (and trip `forbid`) on their own.
    final_output = actions[-1].output or {}
    answer = final_output.get("response") or final_output.get("markdown_report") or ""
    assert answer, f"No final answer in {final_output}"
    return answer.replace("*", "").replace("`", ""), actions


def test_startup_loads_the_concept_catalog(ontology_agent_config):
    node = _build_node(ontology_agent_config)
    assert node.concept_catalog_mode == "full"
    assert '"FlightAlert"' in node._get_system_prompt()


QUESTIONS = [
    pytest.param(
        "Among flights to JFK, which P1 alerts are still open? Give the alert id, type, flight number and airport name.",
        [r"A007", r"A011"],
        [],
        id="open_p1_alerts_into_jfk",
    ),
    pytest.param(
        "Which destination airport has the most delayed flights? "
        "List the delayed flight count per destination airport.",
        [r"LAX[^\d\n]*3\b", r"SFO[^\d\n]*2\b", r"JFK[^\d\n]*2\b", r"SEA[^\d\n]*1\b"],
        [],
        id="delayed_flights_by_destination",
    ),
    pytest.param(
        "By origin airport, which airport has the most delayed flights? "
        "List the delayed flight count per origin airport.",
        [r"SFO[^\d\n]*6\b", r"JFK[^\d\n]*1\b", r"LAS[^\d\n]*1\b"],
        # The destination role's numbers mean the two roles were confused.
        [r"LAX[^\d\n]*3\b"],
        id="delayed_flights_by_origin",
    ),
    pytest.param(
        "How many alerts are still unresolved? Break it down by priority.",
        [r"P0[^\d\n]*1\b", r"P1[^\d\n]*3\b", r"P2[^\d\n]*4\b"],
        [],
        id="unresolved_alerts_by_priority",
    ),
    pytest.param(
        "Looking only at flights scheduled to depart on 2026-08-14 and 2026-08-15, "
        "how many delayed flights does each destination airport have?",
        [r"LAX[^\d\n]*1\b", r"SFO[^\d\n]*1\b", r"JFK[^\d\n]*1\b"],
        # SEA's only delayed flight departs outside the window.
        [r"SEA[^\d\n]*[1-9]\b"],
        id="delayed_flights_by_destination_in_window",
    ),
    pytest.param(
        "Which delayed flights also have alerts? List the flight number, delay minutes and alert count.",
        [
            r"SK202[^\n]*95[^\n]*2",
            r"SK103[^\n]*60[^\n]*2",
            r"SK201[^\n]*55[^\n]*1",
            r"SK102[^\n]*42[^\n]*2",
            r"UA303[^\n]*30[^\n]*1",
            r"UA401[^\n]*29[^\n]*1",
            r"UA302[^\n]*25[^\n]*1",
        ],
        # Joining delays and alerts at record level splits SK202's 95 minutes into 65 and 30.
        [r"SK202[^\n]*\b65\b"],
        id="delayed_flights_with_alerts",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("question, expect, forbid", QUESTIONS)
async def test_answers_ontology_question(ontology_agent_config, question, expect, forbid):
    node = _build_node(ontology_agent_config)
    answer, actions = await _ask(node, question)

    used = {a.action_type for a in actions if a.role == ActionRole.TOOL and a.status == ActionStatus.SUCCESS}
    assert used & {"query_metrics", "select_records"}, f"No result-producing tool succeeded; used {sorted(used)}"
    missing = [pattern for pattern in expect if not re.search(pattern, answer)]
    assert missing == [], f"Answer lacks {missing}:\n{answer}"
    present = [pattern for pattern in forbid if re.search(pattern, answer)]
    assert present == [], f"Answer contains {present}:\n{answer}"
