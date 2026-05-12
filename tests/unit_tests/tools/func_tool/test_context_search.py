# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Extended unit tests for ContextSearchTools covering uncovered lines."""

from unittest.mock import Mock, patch

import pytest

from datus.configuration.agent_config import AgentConfig
from datus.tools.func_tool import ContextSearchTools
from datus.tools.func_tool.base import FuncToolResult, normalize_null
from datus.tools.func_tool.context_search import _fill_subject_tree, _normalize_subject_tree


class TestNormalizeNull:
    def test_none_returns_none(self):
        assert normalize_null(None) is None

    def test_string_null_returns_none(self):
        assert normalize_null("null") is None

    def test_string_none_returns_none(self):
        assert normalize_null("None") is None

    def test_valid_string_passes_through(self):
        assert normalize_null("value") == "value"

    def test_list_passes_through(self):
        lst = ["a", "b"]
        assert normalize_null(lst) is lst

    def test_zero_passes_through(self):
        assert normalize_null(0) == 0


class TestFillSubjectTree:
    def test_fills_tree_with_metrics(self):
        tree = {}
        entries = [{"subject_path": ["Finance", "Revenue"], "name": "revenue_total"}]
        _fill_subject_tree(tree, entries, "metrics")
        assert tree["Finance"]["Revenue"]["metrics"] == {"revenue_total"}

    def test_fills_tree_multiple_entries(self):
        tree = {}
        entries = [
            {"subject_path": ["Sales"], "name": "orders"},
            {"subject_path": ["Sales"], "name": "revenue"},
        ]
        _fill_subject_tree(tree, entries, "metrics")
        assert tree["Sales"]["metrics"] == {"orders", "revenue"}

    def test_skips_entries_without_subject_path(self):
        tree = {}
        entries = [{"name": "orphan_metric"}]  # no subject_path
        _fill_subject_tree(tree, entries, "metrics")
        assert tree == {}

    def test_mixed_entry_types(self):
        tree = {}
        metric_entries = [{"subject_path": ["A"], "name": "m1"}]
        sql_entries = [{"subject_path": ["A"], "name": "sql1"}]
        _fill_subject_tree(tree, metric_entries, "metrics")
        _fill_subject_tree(tree, sql_entries, "reference_sql")
        assert "m1" in tree["A"]["metrics"]
        assert "sql1" in tree["A"]["reference_sql"]


class TestNormalizeSubjectTree:
    def test_converts_sets_to_sorted_lists(self):
        tree = {"Sales": {"metrics": {"b_metric", "a_metric"}}}
        _normalize_subject_tree(tree)
        assert tree["Sales"]["metrics"] == ["a_metric", "b_metric"]

    def test_nested_normalization(self):
        tree = {"A": {"B": {"reference_sql": {"z_sql", "a_sql"}}}}
        _normalize_subject_tree(tree)
        assert tree["A"]["B"]["reference_sql"] == ["a_sql", "z_sql"]

    def test_non_set_leaves_unchanged(self):
        tree = {"Sales": {"metrics": ["already_list"]}}
        _normalize_subject_tree(tree)
        assert tree["Sales"]["metrics"] == ["already_list"]


@pytest.fixture
def mock_agent_config():
    config = Mock()
    config.sub_agent_config.return_value = None
    return config


@pytest.fixture
def build_tools(mock_agent_config):
    def _builder(
        metric_cfg=None,
        sql_cfg=None,
        knowledge_cfg=None,
        semantic_cfg=None,
        template_cfg=None,
        sub_agent_name=None,
        sub_agent_tools=None,
    ):
        metric_cfg = metric_cfg or {}
        sql_cfg = sql_cfg or {}
        knowledge_cfg = knowledge_cfg or {}
        semantic_cfg = semantic_cfg or {}
        template_cfg = template_cfg or {}

        metric_rag = Mock()
        metric_entries = metric_cfg.get("entries", [])
        metric_rag.search_all_metrics.return_value = metric_entries
        metric_rag.search_metrics.return_value = metric_cfg.get("search_return", [])
        metric_rag.get_metrics_size.return_value = metric_cfg.get("size", len(metric_entries))
        metric_rag.get_metrics_detail.return_value = metric_cfg.get("detail_return", [])
        if "search_metrics_side_effect" in metric_cfg:
            metric_rag.search_metrics.side_effect = metric_cfg["search_metrics_side_effect"]
        if "detail_side_effect" in metric_cfg:
            metric_rag.get_metrics_detail.side_effect = metric_cfg["detail_side_effect"]

        sql_rag = Mock()
        sql_entries = sql_cfg.get("entries", [])
        sql_rag.search_all_reference_sql.return_value = sql_entries
        sql_rag.search_reference_sql.return_value = sql_cfg.get("search_return", [])
        sql_rag.get_reference_sql_size.return_value = sql_cfg.get("size", len(sql_entries))
        sql_rag.get_reference_sql_detail.return_value = sql_cfg.get("detail_return", [])
        if "detail_side_effect" in sql_cfg:
            sql_rag.get_reference_sql_detail.side_effect = sql_cfg["detail_side_effect"]
        if "search_sql_side_effect" in sql_cfg:
            sql_rag.search_reference_sql.side_effect = sql_cfg["search_sql_side_effect"]

        semantic_rag = Mock()
        semantic_rag.get_size.return_value = semantic_cfg.get("size", 0)
        mock_storage = Mock()
        mock_storage.search_objects.return_value = semantic_cfg.get("search_return", [])
        if "search_side_effect" in semantic_cfg:
            mock_storage.search_objects.side_effect = semantic_cfg["search_side_effect"]
        semantic_rag.storage = mock_storage

        ext_knowledge_rag = Mock()
        knowledge_entries = knowledge_cfg.get("entries", [])
        ext_knowledge_rag.get_knowledge_size.return_value = knowledge_cfg.get("size", len(knowledge_entries))
        ext_knowledge_rag.query_knowledge.return_value = knowledge_cfg.get("search_return", [])
        ext_knowledge_rag.get_knowledge_batch.return_value = knowledge_cfg.get("get_return", [])
        ext_knowledge_rag.store = Mock()
        ext_knowledge_rag.store.search_all_knowledge.return_value = knowledge_entries
        if "query_side_effect" in knowledge_cfg:
            ext_knowledge_rag.query_knowledge.side_effect = knowledge_cfg["query_side_effect"]

        reference_template_rag = Mock()
        template_entries = template_cfg.get("entries", [])
        reference_template_rag.search_all_reference_templates.return_value = template_entries
        reference_template_rag.get_reference_template_size.return_value = template_cfg.get(
            "size", len(template_entries)
        )

        if sub_agent_name is not None:
            mock_agent_config.sub_agent_config.return_value = {"tools": sub_agent_tools or ""}

        with (
            patch("datus.tools.func_tool.context_search.MetricRAG", return_value=metric_rag),
            patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=semantic_rag),
            patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=sql_rag),
            patch("datus.tools.func_tool.context_search.ExtKnowledgeRAG", return_value=ext_knowledge_rag),
            patch("datus.tools.func_tool.context_search.ReferenceTemplateRAG", return_value=reference_template_rag),
            patch(
                "datus.tools.func_tool.context_search.MetricRAG.storage.subject_tree",
                return_value=Mock(),
            ),
        ):
            from datus.tools.func_tool.context_search import ContextSearchTools

            tools = ContextSearchTools(mock_agent_config, sub_agent_name=sub_agent_name)
        return tools, metric_rag, sql_rag, ext_knowledge_rag, semantic_rag

    return _builder


class TestGetMetrics:
    def test_success(self, build_tools):
        metric_detail = {"name": "revenue", "description": "Total revenue", "sql_query": "SELECT SUM(amount)"}
        tools, metric_rag, _, _, _ = build_tools(
            metric_cfg={
                "entries": [{"subject_path": ["Finance"], "name": "revenue"}],
                "detail_return": [metric_detail],
            }
        )

        result = tools.get_metrics(subject_path=["Finance"], name="revenue")

        assert result.success == 1
        assert result.result == metric_detail

    def test_not_found(self, build_tools):
        tools, _, _, _, _ = build_tools(
            metric_cfg={
                "entries": [{"subject_path": ["Finance"], "name": "revenue"}],
                "detail_return": [],
            }
        )

        result = tools.get_metrics(subject_path=["Finance"], name="unknown")

        assert result.success == 0
        assert "No matched result" in result.error

    def test_exception_returns_failure(self, build_tools):
        tools, metric_rag, _, _, _ = build_tools(
            metric_cfg={
                "entries": [{"subject_path": ["Finance"], "name": "revenue"}],
                "detail_side_effect": Exception("db error"),
            }
        )

        result = tools.get_metrics(subject_path=["Finance"], name="revenue")

        assert result.success == 0
        assert "db error" in result.error

    def test_null_name_normalized(self, build_tools):
        tools, metric_rag, _, _, _ = build_tools(
            metric_cfg={
                "entries": [{"subject_path": ["Finance"], "name": "revenue"}],
                "detail_return": [],
            }
        )

        tools.get_metrics(subject_path=["Finance"], name="null")

        metric_rag.get_metrics_detail.assert_called_once_with(
            subject_path=["Finance"],
            name="",
        )


class TestGetReferenceSQL:
    def test_success(self, build_tools):
        sql_detail = {"name": "sales_query", "sql": "SELECT * FROM sales", "summary": "Sales report"}
        tools, _, sql_rag, _, _ = build_tools(
            sql_cfg={
                "entries": [{"subject_path": ["Sales"], "name": "sales_query"}],
                "detail_return": [sql_detail],
            }
        )

        result = tools.get_reference_sql(subject_path=["Sales"], name="sales_query")

        assert result.success == 1
        assert result.result == sql_detail

    def test_not_found_returns_error(self, build_tools):
        tools, _, sql_rag, _, _ = build_tools(
            sql_cfg={
                "entries": [{"subject_path": ["Sales"], "name": "sales_query"}],
                "detail_return": [],
            }
        )

        result = tools.get_reference_sql(subject_path=["Sales"], name="unknown")

        assert result.success == 0
        assert "No matched result" in result.error

    def test_exception_returns_failure(self, build_tools):
        tools, _, sql_rag, _, _ = build_tools(
            sql_cfg={
                "entries": [{"subject_path": ["Sales"], "name": "sales_query"}],
                "detail_side_effect": Exception("sql error"),
            }
        )

        result = tools.get_reference_sql(subject_path=["Sales"], name="query")

        assert result.success == 0
        assert "sql error" in result.error

    def test_null_name_normalized(self, build_tools):
        tools, _, sql_rag, _, _ = build_tools(
            sql_cfg={
                "entries": [{"subject_path": ["Sales"], "name": "sales_query"}],
                "detail_return": [],
            }
        )

        tools.get_reference_sql(subject_path=["Sales"], name="None")

        sql_rag.get_reference_sql_detail.assert_called_once_with(
            subject_path=["Sales"],
            name="",
            selected_fields=["name", "sql", "summary", "tags"],
        )


class TestSearchSemanticObjects:
    def test_success(self, build_tools):
        tools, _, _, _, semantic_rag = build_tools(
            semantic_cfg={
                "size": 2,
                "search_return": [{"kind": "table", "name": "orders", "description": "orders table"}],
            }
        )

        result = tools.search_semantic_objects("orders table")

        assert result.success == 1
        assert len(result.result) == 1

    def test_with_kinds_filter(self, build_tools):
        tools, _, _, _, semantic_rag = build_tools(
            semantic_cfg={
                "size": 2,
                "search_return": [{"kind": "column", "name": "amount"}],
            }
        )

        result = tools.search_semantic_objects("amount column", kinds=["column"])

        assert result.success == 1
        semantic_rag.storage.search_objects.assert_called_once_with(
            query_text="amount column",
            kinds=["column"],
            top_n=5,
        )

    def test_null_kinds_normalized(self, build_tools):
        tools, _, _, _, semantic_rag = build_tools(semantic_cfg={"size": 1, "search_return": []})

        tools.search_semantic_objects("test", kinds="null")

        semantic_rag.storage.search_objects.assert_called_once_with(
            query_text="test",
            kinds=None,
            top_n=5,
        )

    def test_exception_returns_failure(self, build_tools):
        tools, _, _, _, semantic_rag = build_tools(
            semantic_cfg={
                "size": 1,
                "search_side_effect": Exception("vector search error"),
            }
        )

        result = tools.search_semantic_objects("test")

        assert result.success == 0
        assert "vector search error" in result.error


class TestSearchKnowledge:
    def test_success(self, build_tools):
        knowledge_entries = [{"subject_path": ["Business"], "name": "GMV"}]
        search_result = [{"search_text": "GMV", "explanation": "Gross Merchandise Value"}]
        tools, _, _, ext_knowledge_rag, _ = build_tools(
            knowledge_cfg={"entries": knowledge_entries, "search_return": search_result}
        )

        result = tools.search_knowledge("GMV definition")

        assert result.success == 1
        assert result.result == search_result

    def test_null_subject_path_normalized(self, build_tools):
        knowledge_entries = [{"subject_path": ["Business"], "name": "GMV"}]
        tools, _, _, ext_knowledge_rag, _ = build_tools(
            knowledge_cfg={"entries": knowledge_entries, "search_return": []}
        )

        tools.search_knowledge("test", subject_path="null")

        ext_knowledge_rag.query_knowledge.assert_called_once_with(
            query_text="test",
            subject_path=None,
            top_n=5,
        )

    def test_exception_returns_failure(self, build_tools):
        knowledge_entries = [{"subject_path": ["Business"], "name": "GMV"}]
        tools, _, _, ext_knowledge_rag, _ = build_tools(
            knowledge_cfg={"entries": knowledge_entries, "query_side_effect": Exception("knowledge error")}
        )

        result = tools.search_knowledge("GMV")

        assert result.success == 0
        assert "knowledge error" in result.error


class TestGetKnowledgeEmptyPaths:
    def test_empty_paths_returns_error(self, build_tools):
        knowledge_entries = [{"subject_path": ["Business"], "name": "GMV"}]
        tools, _, _, _, _ = build_tools(knowledge_cfg={"entries": knowledge_entries})

        result = tools.get_knowledge(paths=[])

        assert result.success == 0
        assert "No paths provided" in result.error


class TestListSubjectTreeWithKnowledge:
    def test_knowledge_entries_included(self, build_tools):
        knowledge_entries = [{"subject_path": ["Business", "Terms"], "name": "GMV"}]
        tools, _, _, _, _ = build_tools(knowledge_cfg={"entries": knowledge_entries})

        result = tools.list_subject_tree()
        assert result.success == 1
        assert "Business" in result.result
        assert "GMV" in result.result["Business"]["Terms"]["knowledge"]

    def test_empty_stores_return_empty_tree(self, build_tools):
        tools, _, _, _, _ = build_tools()
        result = tools.list_subject_tree()
        assert result.success == 1
        assert result.result == {}


class TestListSubjectTreeSubAgentScoping:
    """Regression: collectors must run on raw `has_*` flags, not literal `_show_*`
    membership in `tool_list`. Wildcard tool patterns ('context_search_tools.*')
    are stored verbatim by SubAgentConfig.tool_list, so a literal `in` check
    used to short-circuit every collector and return {}. Scope filtering still
    happens at the storage layer via _sub_agent_conditions().
    """

    def test_wildcard_tool_list_returns_populated_tree(self, build_tools):
        metric_entries = [{"subject_path": ["Finance"], "name": "revenue"}]
        sql_entries = [{"subject_path": ["Finance"], "name": "rev_sql"}]
        knowledge_entries = [{"subject_path": ["Business"], "name": "GMV"}]
        template_entries = [{"subject_path": ["Ops"], "name": "tpl_a"}]
        tools, _, _, _, _ = build_tools(
            metric_cfg={"entries": metric_entries},
            sql_cfg={"entries": sql_entries},
            knowledge_cfg={"entries": knowledge_entries},
            template_cfg={"entries": template_entries},
            sub_agent_name="agent_a",
            sub_agent_tools="context_search_tools.*,date_parsing_tools.*",
        )

        result = tools.list_subject_tree()

        assert result.success == 1
        assert result.result["Finance"]["metrics"] == ["revenue"]
        assert result.result["Finance"]["reference_sql"] == ["rev_sql"]
        assert result.result["Business"]["knowledge"] == ["GMV"]
        assert result.result["Ops"]["reference_template"] == ["tpl_a"]

    def test_literal_tool_list_returns_populated_tree(self, build_tools):
        metric_entries = [{"subject_path": ["Finance"], "name": "revenue"}]
        sql_entries = [{"subject_path": ["Finance"], "name": "rev_sql"}]
        tools, _, _, _, _ = build_tools(
            metric_cfg={"entries": metric_entries},
            sql_cfg={"entries": sql_entries},
            sub_agent_name="agent_b",
            sub_agent_tools=("context_search_tools.list_subject_tree,context_search_tools.search_metrics"),
        )

        result = tools.list_subject_tree()

        assert result.success == 1
        assert result.result["Finance"]["metrics"] == ["revenue"]
        assert result.result["Finance"]["reference_sql"] == ["rev_sql"]

    def test_empty_stores_with_wildcard_returns_empty_tree(self, build_tools):
        tools, _, _, _, _ = build_tools(
            sub_agent_name="agent_c",
            sub_agent_tools="context_search_tools.*",
        )

        result = tools.list_subject_tree()

        assert result.success == 1
        assert result.result == {}

    def test_empty_tool_list_still_returns_data(self, build_tools):
        # With sub_agent_config present but empty tools, the storage-layer
        # _sub_agent_conditions() owns scope filtering. The collectors should
        # still surface populated subject paths rather than silently returning
        # {} based on the legacy literal `_show_*` membership check.
        metric_entries = [{"subject_path": ["Finance"], "name": "revenue"}]
        tools, _, _, _, _ = build_tools(
            metric_cfg={"entries": metric_entries},
            sub_agent_name="agent_d",
            sub_agent_tools="",
        )

        result = tools.list_subject_tree()

        assert result.success == 1
        assert result.result["Finance"]["metrics"] == ["revenue"]


class TestAllToolsName:
    def test_returns_list_of_strings(self):
        from datus.tools.func_tool.context_search import ContextSearchTools

        names = ContextSearchTools.all_tools_name()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "list_subject_tree" in names
        assert "search_metrics" in names


def _make_full_rag_mocks():
    """Create fully configured RAG mocks with proper integer return values."""
    mock_metric = Mock()
    mock_metric.get_metrics_size.return_value = 0
    mock_metric.storage.subject_tree = Mock()

    mock_semantic = Mock()
    mock_semantic.get_size.return_value = 0

    mock_sql = Mock()
    mock_sql.get_reference_sql_size.return_value = 0

    mock_knowledge = Mock()
    mock_knowledge.get_knowledge_size.return_value = 0

    mock_reference_template = Mock()
    mock_reference_template.get_reference_template_size.return_value = 0

    return mock_metric, mock_semantic, mock_sql, mock_knowledge, mock_reference_template


class TestCreateFactoryMethods:
    def test_create_dynamic(self, mock_agent_config):
        mock_metric, mock_semantic, mock_sql, mock_knowledge, mock_ref_tpl = _make_full_rag_mocks()
        with (
            patch("datus.tools.func_tool.context_search.MetricRAG", return_value=mock_metric),
            patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=mock_semantic),
            patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=mock_sql),
            patch("datus.tools.func_tool.context_search.ExtKnowledgeRAG", return_value=mock_knowledge),
            patch("datus.tools.func_tool.context_search.ReferenceTemplateRAG", return_value=mock_ref_tpl),
        ):
            from datus.tools.func_tool.context_search import ContextSearchTools

            tool = ContextSearchTools.create_dynamic(mock_agent_config)
        assert isinstance(tool, ContextSearchTools)

    def test_create_static(self, mock_agent_config):
        mock_metric, mock_semantic, mock_sql, mock_knowledge, mock_ref_tpl = _make_full_rag_mocks()
        with (
            patch("datus.tools.func_tool.context_search.MetricRAG", return_value=mock_metric),
            patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=mock_semantic),
            patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=mock_sql),
            patch("datus.tools.func_tool.context_search.ExtKnowledgeRAG", return_value=mock_knowledge),
            patch("datus.tools.func_tool.context_search.ReferenceTemplateRAG", return_value=mock_ref_tpl),
        ):
            from datus.tools.func_tool.context_search import ContextSearchTools

            tool = ContextSearchTools.create_static(mock_agent_config, database_name="testdb")
        assert isinstance(tool, ContextSearchTools)


# ---------------------------------------------------------------------------
# Available tools and search behavior cases merged from test_context_search_tools.py
# ---------------------------------------------------------------------------


METRIC_ENTRIES = [
    {"subject_path": ["Sales", "Revenue", "Monthly"], "name": "monthly_sales"},
    {"subject_path": ["Sales", "Revenue", "Quarterly"], "name": "quarterly_sales"},
]

SQL_ENTRIES = [
    {"subject_path": ["Sales", "Revenue", "Monthly"], "name": "sales_query"},
    {"subject_path": ["Support", "Tickets", "Escalations"], "name": "support_query"},
]


@pytest.fixture
def context_search_agent_config() -> AgentConfig:
    config = Mock(spec=AgentConfig)
    config.rag_storage_path.return_value = "/tmp/test_rag_storage"
    config.sub_agent_config.return_value = None
    return config


def _build_tree_structure(entries: list) -> dict:
    """Build a tree structure from subject path entries."""
    tree = {}
    for entry in entries:
        current = tree
        for part in entry["subject_path"]:
            if part not in current:
                current[part] = {"node_id": None, "children": {}}
            current = current[part]["children"]
    return tree


@pytest.fixture
def build_context_search_tools(context_search_agent_config):
    def _builder(metric_cfg=None, sql_cfg=None, knowledge_cfg=None):
        metric_cfg = metric_cfg or {}
        sql_cfg = sql_cfg or {}
        knowledge_cfg = knowledge_cfg or {}

        mock_subject_tree = Mock()
        mock_subject_tree.find_or_create_path = Mock()

        metric_rag = Mock()
        metric_entries = metric_cfg.get("entries", [])
        metric_rag.search_all_metrics.return_value = metric_entries
        metric_rag.search_metrics.return_value = metric_cfg.get("search_return", [])
        metric_rag.get_metrics_size.return_value = metric_cfg.get("size", len(metric_entries))
        if "search_all_side_effect" in metric_cfg:
            metric_rag.search_all_metrics.side_effect = metric_cfg["search_all_side_effect"]
        if "search_metrics_side_effect" in metric_cfg:
            metric_rag.search_metrics.side_effect = metric_cfg["search_metrics_side_effect"]

        sql_rag = Mock()
        sql_entries = sql_cfg.get("entries", [])
        sql_rag.search_all_reference_sql.return_value = sql_entries
        sql_rag.search_reference_sql.return_value = sql_cfg.get("search_return", [])
        sql_rag.get_reference_sql_size.return_value = sql_cfg.get("size", len(sql_entries))
        if "search_all_side_effect" in sql_cfg:
            sql_rag.search_all_reference_sql.side_effect = sql_cfg["search_all_side_effect"]
        if "search_sql_side_effect" in sql_cfg:
            sql_rag.search_reference_sql.side_effect = sql_cfg["search_sql_side_effect"]

        semantic_rag = Mock()
        semantic_rag.get_size.return_value = 0

        ext_knowledge_rag = Mock()
        knowledge_entries = knowledge_cfg.get("entries", [])
        ext_knowledge_rag.get_knowledge_size.return_value = knowledge_cfg.get("size", len(knowledge_entries))
        ext_knowledge_rag.query_knowledge.return_value = knowledge_cfg.get("search_return", [])
        ext_knowledge_rag.get_knowledge_batch.return_value = knowledge_cfg.get("get_return", [])
        ext_knowledge_rag.store = Mock()
        ext_knowledge_rag.store.search_all_knowledge.return_value = knowledge_entries
        if "get_knowledge_side_effect" in knowledge_cfg:
            ext_knowledge_rag.get_knowledge_batch.side_effect = knowledge_cfg["get_knowledge_side_effect"]

        all_entries = metric_entries + sql_entries + knowledge_entries
        mock_subject_tree.get_tree_structure.return_value = _build_tree_structure(all_entries)

        reference_template_rag = Mock()
        reference_template_rag.get_reference_template_size.return_value = 0

        with (
            patch("datus.tools.func_tool.context_search.MetricRAG", return_value=metric_rag),
            patch("datus.tools.func_tool.context_search.SemanticModelRAG", return_value=semantic_rag),
            patch("datus.tools.func_tool.context_search.ReferenceSqlRAG", return_value=sql_rag),
            patch("datus.tools.func_tool.context_search.ExtKnowledgeRAG", return_value=ext_knowledge_rag),
            patch("datus.tools.func_tool.context_search.ReferenceTemplateRAG", return_value=reference_template_rag),
            patch(
                "datus.tools.func_tool.context_search.MetricRAG.storage.subject_tree",
                return_value=mock_subject_tree,
            ),
        ):
            tools = ContextSearchTools(context_search_agent_config)
        return tools, metric_rag, sql_rag, mock_subject_tree, ext_knowledge_rag

    return _builder


def test_available_tools_with_metrics_and_sql(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {
        "list_subject_tree",
        "search_metrics",
        "get_metrics",
        "search_reference_sql",
        "get_reference_sql",
    }


def test_available_tools_metrics_only(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": METRIC_ENTRIES, "search_return": [{"name": "monthly_sales"}]},
        sql_cfg={"entries": [], "size": 0},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_metrics", "get_metrics"}


def test_available_tools_sql_only(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": [], "size": 0},
        sql_cfg={"entries": SQL_ENTRIES, "search_return": [{"name": "sales_query"}]},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_reference_sql", "get_reference_sql"}


def test_list_domain_layers_tree_combined(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={"entries": SQL_ENTRIES},
    )

    result = tools.list_subject_tree()
    assert isinstance(result, FuncToolResult)
    assert result.success == 1
    assert result.result == {
        "Sales": {
            "Revenue": {
                "Monthly": {
                    "metrics": ["monthly_sales"],
                    "reference_sql": ["sales_query"],
                },
                "Quarterly": {
                    "metrics": ["quarterly_sales"],
                },
            }
        },
        "Support": {
            "Tickets": {
                "Escalations": {
                    "reference_sql": ["support_query"],
                }
            }
        },
    }


def test_collect_metrics_entries_handles_exception(build_context_search_tools):
    tools, metric_rag, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": [], "size": 1, "search_all_side_effect": RuntimeError("metrics offline")}
    )

    entries = tools._collect_metrics_entries()
    assert entries == []
    metric_rag.search_all_metrics.assert_called_once()


def test_collect_sql_entries_handles_exception(build_context_search_tools):
    tools, _, sql_rag, _, _ = build_context_search_tools(
        sql_cfg={"entries": [], "size": 1, "search_all_side_effect": RuntimeError("sql offline")}
    )

    entries = tools._collect_sql_entries()
    assert entries == []
    sql_rag.search_all_reference_sql.assert_called_once()


def test_search_metrics_passes_filters(build_context_search_tools):
    tools, metric_rag, _, _, _ = build_context_search_tools(
        metric_cfg={
            "entries": METRIC_ENTRIES,
            "search_return": [{"name": "monthly_sales"}],
        }
    )

    result = tools.search_metrics(
        query_text="revenue",
        subject_path=["Sales", "Revenue", "Monthly"],
        top_n=3,
    )

    assert result.success == 1
    metric_rag.search_metrics.assert_called_once_with(
        query_text="revenue",
        subject_path=["Sales", "Revenue", "Monthly"],
        top_n=3,
    )


def test_search_metrics_handles_failure(build_context_search_tools):
    tools, metric_rag, _, _, _ = build_context_search_tools(
        metric_cfg={
            "entries": METRIC_ENTRIES,
            "search_metrics_side_effect": Exception("metric search failed"),
        }
    )

    result = tools.search_metrics("revenue")
    assert result.success == 0
    assert "metric search failed" in (result.error or "")
    metric_rag.search_metrics.assert_called_once()


def test_search_historical_sql(build_context_search_tools):
    tools, _, sql_rag, _, _ = build_context_search_tools(
        metric_cfg={"entries": METRIC_ENTRIES},
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_return": [{"name": "sales_query", "sql": "SELECT * FROM sales"}],
        },
    )

    result = tools.search_reference_sql("sales report", subject_path=["Sales", "Revenue"], top_n=2)
    assert result.success == 1
    sql_rag.search_reference_sql.assert_called_once_with(
        query_text="sales report",
        subject_path=["Sales", "Revenue"],
        top_n=2,
        selected_fields=["name", "sql", "summary", "tags"],
    )


def test_search_historical_sql_handles_failure(build_context_search_tools):
    tools, _, sql_rag, _, _ = build_context_search_tools(
        sql_cfg={
            "entries": SQL_ENTRIES,
            "search_sql_side_effect": Exception("sql search failed"),
        }
    )

    result = tools.search_reference_sql("sales report")
    assert result.success == 0
    assert "sql search failed" in (result.error or "")
    sql_rag.search_reference_sql.assert_called_once()


KNOWLEDGE_ENTRIES = [
    {"subject_path": ["Business", "Terms"], "name": "GMV"},
    {"subject_path": ["Business", "Terms"], "name": "ARR"},
]


def test_available_tools_with_knowledge(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        metric_cfg={"entries": [], "size": 0},
        sql_cfg={"entries": [], "size": 0},
        knowledge_cfg={"entries": KNOWLEDGE_ENTRIES},
    )

    tool_names = {tool.name for tool in tools.available_tools()}
    assert tool_names == {"list_subject_tree", "search_knowledge", "get_knowledge"}


def test_get_knowledge_success(build_context_search_tools):
    knowledge_detail = {
        "search_text": "GMV",
        "explanation": "Gross Merchandise Value is the total sales value",
    }
    tools, _, _, _, ext_knowledge_rag = build_context_search_tools(
        knowledge_cfg={
            "entries": KNOWLEDGE_ENTRIES,
            "get_return": [knowledge_detail],
        }
    )

    result = tools.get_knowledge(paths=[["Business", "Terms", "GMV"]])
    assert result.success == 1
    assert result.result == [knowledge_detail]
    ext_knowledge_rag.get_knowledge_batch.assert_called_once_with(
        paths=[["Business", "Terms", "GMV"]],
    )


def test_get_knowledge_not_found(build_context_search_tools):
    tools, _, _, _, _ = build_context_search_tools(
        knowledge_cfg={
            "entries": KNOWLEDGE_ENTRIES,
            "get_return": [],
        }
    )

    result = tools.get_knowledge(paths=[["Business", "Terms", "Unknown"]])
    assert result.success == 0
    assert result.error == "No matched result"


def test_get_knowledge_handles_failure(build_context_search_tools):
    tools, _, _, _, ext_knowledge_rag = build_context_search_tools(
        knowledge_cfg={
            "entries": KNOWLEDGE_ENTRIES,
            "get_knowledge_side_effect": Exception("knowledge retrieval failed"),
        }
    )

    result = tools.get_knowledge(paths=[["Business", "Terms", "GMV"]])
    assert result.success == 0
    assert "knowledge retrieval failed" in (result.error or "")
    ext_knowledge_rag.get_knowledge_batch.assert_called_once()
