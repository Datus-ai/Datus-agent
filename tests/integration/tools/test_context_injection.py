# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Nightly coverage for the data_access.context_injection flow (issue #922).

This module proves each context source can be RETRIEVED and INJECTED into the
SQL-generation path:

* Retrieval layer (no LLM): self-seed catalog/schema-metadata (semantic objects),
  subject-tree, historical/reference SQL, metrics, and external knowledge via the
  RAG ``upsert_batch`` / ``after_init`` APIs, then assert ``ContextSearchTools``
  surfaces them through ``search_knowledge`` / ``get_knowledge`` /
  ``list_subject_tree`` and fuzzy (semantic + FTS) search.
* Injection layer (real LLM): construct ``GenSQLAgenticNode`` from the
  ``nightly_agent_config`` fixture, assert the context-search tools are wired into
  ``node.tools``, then run the node end-to-end against the seeded reference SQL and
  assert a tool action occurred and the run succeeded.

Modelled after tests/integration/tools/test_context_search.py (retrieval) and
tests/integration/agent/test_gen_metrics_agentic.py (execute_stream).
"""

import os

import pytest
from pandas import Timestamp

from datus.agent.node.gen_sql_agentic_node import GenSQLAgenticNode
from datus.configuration.agent_config import AgentConfig
from datus.configuration.node_type import NodeType
from datus.schemas.action_history import ActionHistoryManager, ActionRole, ActionStatus
from datus.schemas.gen_sql_agentic_node_models import GenSQLNodeInput
from datus.storage.ext_knowledge.store import ExtKnowledgeRAG
from datus.storage.metric.store import MetricRAG, build_metric_id
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.tools.func_tool.context_search import ContextSearchTools
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Shared subject hierarchy so every seeded context source lands under the same
# subject-tree branch and ``list_subject_tree`` can nest them together.
SUBJECT_PATH = ["california_schools", "context_injection"]


# ── Seed payload builders ─────────────────────────────────────────────────


def _metric_items():
    """A metric whose business name is 'school_count' (used for fuzzy search)."""
    name = "school_count"
    return [
        {
            "id": build_metric_id(SUBJECT_PATH, name),
            "subject_path": SUBJECT_PATH,
            "semantic_model_name": "nightly_injection_school",
            "name": name,
            "description": (
                "Total number of schools / campuses in the California schools dataset, "
                "supporting county-level campus counting and enrollment analysis."
            ),
            "metric_type": "simple",
            "measure_expr": "COUNT(*)",
            "base_measures": ["school_count"],
            "dimensions": ["county", "charter"],
            "entities": [],
            "catalog_name": "",
            "database_name": "",
            "schema_name": "",
            "sql": "SELECT COUNT(*) AS school_count FROM schools",
            "yaml_path": "tests/data/metrics/nightly_injection_school_count.yml",
        }
    ]


def _reference_sql_items():
    """Reference SQL counting Fresno charter schools (drives the injection run)."""
    return [
        {
            "id": "nightly_injection_ref_sql_fresno_charter",
            "subject_path": SUBJECT_PATH,
            "name": "fresno_charter_school_count",
            "sql": ("SELECT COUNT(*) AS school_count FROM schools WHERE County = 'Fresno' AND Charter = 1"),
            "comment": "Deterministic reference SQL seed for context injection nightly tests.",
            "summary": "Count charter schools located in Fresno county from the California schools dataset.",
            "search_text": "school schools Fresno charter California county reference SQL count",
            "filepath": "tests/data/reference_sql/nightly_injection_ref_sql_fresno_charter.sql",
            "tags": "nightly,school,reference_sql,injection",
        }
    ]


def _knowledge_items():
    """External-knowledge entry describing the Charter flag semantics."""
    return [
        {
            "subject_path": SUBJECT_PATH,
            "name": "charter_flag_definition",
            "search_text": "charter school definition flag Charter column meaning",
            "explanation": (
                "In the California schools dataset the Charter column equals 1 for charter "
                "schools and 0 otherwise. Filter Charter = 1 to count charter campuses."
            ),
        }
    ]


def _semantic_objects():
    """A table-kind semantic object so search_semantic_objects has catalog metadata."""
    table_name = "schools"
    return [
        {
            "id": f"table:nightly_injection.{table_name}",
            "kind": "table",
            "name": table_name,
            "fq_name": f"california_schools.public.{table_name}",
            "semantic_model_name": "nightly_injection_school",
            "catalog_name": "",
            "database_name": "california_schools",
            "schema_name": "public",
            "table_name": table_name,
            "description": (
                "California schools catalog table holding one row per school / campus, "
                "including County and Charter attributes."
            ),
            "is_dimension": False,
            "is_measure": False,
            "is_entity_key": False,
            "is_deprecated": False,
            "expr": "",
            "column_type": "",
            "agg": "",
            "create_metric": False,
            "agg_time_dimension": "",
            "is_partition": False,
            "time_granularity": "",
            "entity": "",
            "yaml_path": "",
            "updated_at": Timestamp.now().floor("ms"),
        }
    ]


def _seed_all_context_sources(config: AgentConfig):
    """Seed every context source against ``config`` and build indices.

    Returns the constructed RAG stores so callers can introspect if needed.
    """
    metric_store = MetricRAG(config)
    metric_store.upsert_batch(_metric_items())
    metric_store.after_init()

    reference_sql_store = ReferenceSqlRAG(config)
    reference_sql_store.upsert_batch(_reference_sql_items())
    reference_sql_store.after_init()

    knowledge_store = ExtKnowledgeRAG(config)
    knowledge_store.batch_upsert_knowledge(_knowledge_items())
    knowledge_store.store.after_init()

    semantic_store = SemanticModelRAG(config)
    semantic_store.upsert_batch(_semantic_objects())
    semantic_store.create_indices()

    return metric_store, reference_sql_store, knowledge_store, semantic_store


@pytest.fixture(scope="module")
def seeded_context_data(agent_config: AgentConfig):
    """Seed all context sources once for the retrieval-layer tests."""
    return _seed_all_context_sources(agent_config)


@pytest.mark.nightly
class TestContextRetrievalInjection:
    """Retrieval-layer coverage: each context source is searchable / fetchable."""

    @pytest.fixture
    def ctx_tools(self, agent_config: AgentConfig, seeded_context_data) -> ContextSearchTools:
        return ContextSearchTools(agent_config)

    def test_search_knowledge_returns_seeded_entry(self, ctx_tools):
        """External knowledge: search_knowledge surfaces the seeded charter entry."""
        assert ctx_tools.has_knowledge is True, "Seeded external knowledge should be detected"

        result = ctx_tools.search_knowledge("what does the charter flag mean")

        assert result.success == 1, f"search_knowledge should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) >= 1, "Should surface at least the seeded knowledge entry"

        names = [entry.get("name") for entry in result.result]
        assert "charter_flag_definition" in names, f"Seeded knowledge name missing from results: {names}"

    def test_get_knowledge_by_path(self, ctx_tools):
        """External knowledge: get_knowledge fetches by full subject path + name."""
        assert ctx_tools.has_knowledge is True, "Seeded external knowledge should be detected"

        full_path = SUBJECT_PATH + ["charter_flag_definition"]
        result = ctx_tools.get_knowledge(paths=[full_path])

        assert result.success == 1, f"get_knowledge should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) == 1, f"Should fetch exactly one entry for path {full_path}, got {result.result}"

        entry = result.result[0]
        assert entry.get("name") == "charter_flag_definition", f"Wrong entry returned: {entry}"
        assert "Charter column equals 1" in entry.get("explanation", ""), (
            f"Explanation should match seeded content, got: {entry.get('explanation')}"
        )

    def test_list_subject_tree_nests_all_sources(self, ctx_tools):
        """Subject-tree: list_subject_tree returns the seeded entries nested by path."""
        result = ctx_tools.list_subject_tree()

        assert result.success == 1, f"list_subject_tree should succeed, got error: {result.error}"
        assert isinstance(result.result, dict), f"Result should be a dict, got {type(result.result)}"

        root, leaf_key = SUBJECT_PATH[0], SUBJECT_PATH[1]
        assert root in result.result, f"Root subject '{root}' missing from tree: {list(result.result.keys())}"
        leaf = result.result[root].get(leaf_key)
        assert isinstance(leaf, dict), f"Leaf subject '{leaf_key}' missing under '{root}': {result.result[root]}"

        assert "school_count" in leaf.get("metrics", []), f"Seeded metric not nested in subject tree: {leaf}"
        assert "fresno_charter_school_count" in leaf.get("reference_sql", []), (
            f"Seeded reference SQL not nested in subject tree: {leaf}"
        )
        assert "charter_flag_definition" in leaf.get("knowledge", []), (
            f"Seeded knowledge not nested in subject tree: {leaf}"
        )

    def test_fuzzy_metric_search_surfaces_school_count(self, ctx_tools):
        """Fuzzy search: a semantically-related query ('campus count') finds 'school_count'."""
        assert ctx_tools.has_metrics is True, "Seeded metric should be detected"

        result = ctx_tools.search_metrics("campus count")

        assert result.success == 1, f"search_metrics should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) >= 1, "Fuzzy query 'campus count' should still surface a seeded metric"

        names = [m.get("name") for m in result.result]
        assert "school_count" in names, f"Fuzzy 'campus count' query should surface 'school_count', got: {names}"

    def test_fuzzy_reference_sql_search_surfaces_seed(self, ctx_tools):
        """Fuzzy search: a related query surfaces the seeded Fresno charter reference SQL."""
        assert ctx_tools.has_reference_sql is True, "Seeded reference SQL should be detected"

        result = ctx_tools.search_reference_sql("charter schools in Fresno county")

        assert result.success == 1, f"search_reference_sql should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) >= 1, "Related query should surface a seeded reference SQL"

        names = [r.get("name") for r in result.result]
        assert "fresno_charter_school_count" in names, (
            f"Related query should surface the seeded reference SQL, got: {names}"
        )

    def test_search_semantic_objects_surfaces_catalog_table(self, ctx_tools):
        """Catalog/schema metadata: search_semantic_objects finds the seeded table object."""
        assert ctx_tools.has_semantic_objects is True, "Seeded semantic object should be detected"

        result = ctx_tools.search_semantic_objects("schools campus table")

        assert result.success == 1, f"search_semantic_objects should succeed, got error: {result.error}"
        assert isinstance(result.result, list), f"Result should be a list, got {type(result.result)}"
        assert len(result.result) >= 1, "Should surface at least the seeded table object"

        names = [obj.get("name") for obj in result.result]
        assert "schools" in names, f"Seeded table 'schools' should be surfaced, got: {names}"


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
class TestContextInjectionRealLLM:
    """Injection-layer coverage: context tools are wired into and invoked by gen_sql."""

    def _build_node(self, config: AgentConfig) -> GenSQLAgenticNode:
        return GenSQLAgenticNode(
            node_id="nightly_context_injection",
            description="Nightly context injection node",
            node_type=NodeType.TYPE_GEN_SQL,
            agent_config=config,
            node_name="gen_sql",
            execution_mode="workflow",
        )

    def test_context_tools_wired_into_node(self, nightly_agent_config):
        """Context-search tools are present in the gen_sql node's tool list."""
        # Seed first so ContextSearchTools detects data and exposes the tools.
        _seed_all_context_sources(nightly_agent_config)

        node = self._build_node(nightly_agent_config)
        tool_names = [t.name for t in node.tools]

        assert "search_reference_sql" in tool_names, f"search_reference_sql missing from gen_sql tools: {tool_names}"
        assert "search_metrics" in tool_names, f"search_metrics missing from gen_sql tools: {tool_names}"
        assert "list_subject_tree" in tool_names, f"list_subject_tree missing from gen_sql tools: {tool_names}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_execute_stream_invokes_context_tools(self, nightly_agent_config):
        """End-to-end: the node runs to SUCCESS and invokes a context-search tool.

        Proving context injection means proving a context-search tool was actually
        called during generation, not merely that some (e.g. db schema) tool ran.
        """
        _seed_all_context_sources(nightly_agent_config)

        node = self._build_node(nightly_agent_config)

        node.input = GenSQLNodeInput(
            user_message=(
                "How many charter schools are there in Fresno county? "
                "Before writing SQL, call list_subject_tree and search_reference_sql to look up "
                "any existing reference SQL for Fresno charter schools, and reuse it."
            ),
        )

        action_manager = ActionHistoryManager()
        actions = []
        async for action in node.execute_stream(action_manager):
            actions.append(action)
            logger.info(f"Action: role={action.role}, status={action.status}, type={action.action_type}")

        assert len(actions) >= 2, f"Should have at least 2 actions, got {len(actions)}"

        # A TOOL action's action_type is the tool name, so we can assert that a
        # context-search tool (not just any tool) was invoked during generation.
        context_tool_names = {
            "search_reference_sql",
            "get_reference_sql",
            "search_metrics",
            "get_metrics",
            "search_knowledge",
            "get_knowledge",
            "search_semantic_objects",
            "list_subject_tree",
        }
        invoked_tools = {a.action_type for a in actions if a.role == ActionRole.TOOL}
        assert invoked_tools & context_tool_names, (
            "Expected at least one context-search tool to be invoked during SQL generation, "
            f"but only these tools were called: {sorted(invoked_tools)}"
        )

        assert actions[-1].status == ActionStatus.SUCCESS, (
            f"Last action should be SUCCESS, got {actions[-1].status}: {actions[-1].output}"
        )
