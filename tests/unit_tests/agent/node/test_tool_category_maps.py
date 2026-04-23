# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``_tool_category_map`` coverage for write-capable subagent nodes.

These methods are pure routing — no setup needed beyond hand-stubbed
attributes. Covering them at this tier keeps the regression surface
for ``db_tools.*`` / ``filesystem_tools.*`` / ``scheduler_tools.*``
profile rules under unit tests that finish in milliseconds, instead of
only through ``@pytest.mark.nightly`` flows that spin up real DB
connectors and drive the OpenAI Agents SDK tool loop.

``trans_to_function_tool`` (the OpenAI Agents SDK wrapper) introspects
``func.__name__`` / the docstring, so any attribute wrapped via it must
be a *real* callable — a bare ``MagicMock`` raises ``AttributeError``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _stub(*names):
    """Stub tool registry returning named FunctionTool-like instances."""
    bucket = MagicMock()
    bucket.available_tools = MagicMock(return_value=[_named(n) for n in names])
    return bucket


def _named(name):
    t = MagicMock()
    t.name = name
    return t


def _fn(name):
    """Real callable with ``__name__`` set — required by ``trans_to_function_tool``.

    Parameter names must be public: Pydantic treats leading-underscore
    names as private attributes and rejects them when ``function_schema``
    builds its input model from the signature.
    """

    def _impl() -> None:
        return None

    _impl.__name__ = name
    _impl.__doc__ = f"{name} stub"
    return _impl


class TestGenTableToolCategoryMap:
    def test_db_and_filesystem_buckets_with_ddl_helper(self):
        from datus.agent.node.gen_table_agentic_node import GenTableAgenticNode

        node = GenTableAgenticNode.__new__(GenTableAgenticNode)
        node.skill_func_tool = None
        node.ask_user_tool = _stub("ask_user")
        node.filesystem_func_tool = _stub("read_file", "write_file")

        # ``db_func_tool`` needs ``execute_ddl`` as a real callable — the node
        # wraps it via ``trans_to_function_tool`` which introspects ``__name__``.
        node.db_func_tool = SimpleNamespace(
            available_tools=lambda: [_named("read_query")],
            execute_ddl=_fn("execute_ddl"),
        )

        mapping = node._tool_category_map()
        assert [t.name for t in mapping["db_tools"] if hasattr(t, "name")][0] == "read_query"
        assert len(mapping["db_tools"]) == 2
        assert [t.name for t in mapping["filesystem_tools"]] == ["read_file", "write_file"]
        assert mapping["tools"][0].name == "ask_user"


class TestSchedulerToolCategoryMap:
    def test_scheduler_bucket_plus_ask_user(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode.__new__(SchedulerAgenticNode)
        node.skill_func_tool = None
        node.scheduler_tools = _stub("list_jobs", "delete_job")
        node.ask_user_tool = _stub("ask_user")

        mapping = node._tool_category_map()
        assert [t.name for t in mapping["scheduler_tools"]] == ["list_jobs", "delete_job"]
        assert mapping["tools"][0].name == "ask_user"

    def test_missing_scheduler_tools_omits_bucket(self):
        from datus.agent.node.scheduler_agentic_node import SchedulerAgenticNode

        node = SchedulerAgenticNode.__new__(SchedulerAgenticNode)
        node.skill_func_tool = None
        node.scheduler_tools = None
        node.ask_user_tool = None

        assert node._tool_category_map() == {}


class TestSqlSummaryToolCategoryMap:
    def test_filesystem_and_semantic_helpers(self):
        from datus.agent.node.sql_summary_agentic_node import SqlSummaryAgenticNode

        node = SqlSummaryAgenticNode.__new__(SqlSummaryAgenticNode)
        node.skill_func_tool = None
        node.filesystem_func_tool = _stub("read_file")
        # ``generate_sql_summary_id`` gets wrapped — needs ``__name__``.
        node.generation_tools = SimpleNamespace(
            generate_sql_summary_id=_fn("generate_sql_summary_id"),
        )

        mapping = node._tool_category_map()
        assert mapping["filesystem_tools"][0].name == "read_file"
        assert len(mapping["semantic_tools"]) == 1


class TestGenJobToolCategoryMap:
    def test_db_bucket_includes_write_helpers(self):
        from datus.agent.node.gen_job_agentic_node import GenJobAgenticNode

        node = GenJobAgenticNode.__new__(GenJobAgenticNode)
        node.skill_func_tool = None
        node.ask_user_tool = _stub("ask_user")
        node.filesystem_func_tool = _stub("read_file")

        # All wrapped helpers must have real ``__name__``.
        node.db_func_tool = SimpleNamespace(
            available_tools=lambda: [_named("read_query")],
            execute_ddl=_fn("execute_ddl"),
            execute_write=_fn("execute_write"),
            transfer_query_result=_fn("transfer_query_result"),
            get_migration_capabilities=_fn("get_migration_capabilities"),
            suggest_table_layout=_fn("suggest_table_layout"),
            validate_ddl=_fn("validate_ddl"),
        )

        mapping = node._tool_category_map()
        # 1 available + 6 wrapped helpers = 7.
        assert len(mapping["db_tools"]) == 7
        assert mapping["filesystem_tools"][0].name == "read_file"
        assert mapping["tools"][0].name == "ask_user"


class TestGenMetricsToolCategoryMap:
    def test_semantic_bucket_combines_adapters_and_generation_helpers(self):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
        node.skill_func_tool = None
        node.db_func_tool = _stub("read_query")
        node.semantic_tools = _stub("query_metrics")
        node.generation_tools = SimpleNamespace(
            check_semantic_object_exists=_fn("check_semantic_object_exists"),
            end_metric_generation=_fn("end_metric_generation"),
            end_semantic_model_generation=_fn("end_semantic_model_generation"),
        )
        node.gen_semantic_model_tools = _stub("build_measure")
        node.filesystem_func_tool = _stub("read_file")
        node.ask_user_tool = _stub("ask_user")

        mapping = node._tool_category_map()
        # 1 (query_metrics) + 3 (generation helpers) + 1 (build_measure) = 5.
        assert len(mapping["semantic_tools"]) == 5
        assert mapping["db_tools"][0].name == "read_query"
        assert mapping["filesystem_tools"][0].name == "read_file"
        assert mapping["tools"][0].name == "ask_user"

    def test_semantic_bucket_omitted_when_empty(self):
        from datus.agent.node.gen_metrics_agentic_node import GenMetricsAgenticNode

        node = GenMetricsAgenticNode.__new__(GenMetricsAgenticNode)
        node.skill_func_tool = None
        node.db_func_tool = None
        node.semantic_tools = None
        node.generation_tools = None
        node.gen_semantic_model_tools = None
        node.filesystem_func_tool = None
        node.ask_user_tool = None

        assert node._tool_category_map() == {}


class TestSkillCreatorToolCategoryMap:
    def test_registers_filesystem_db_skill_loader_and_catchall(self):
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode.__new__(SkillCreatorAgenticNode)
        node.skill_func_tool = None
        node.filesystem_func_tool = _stub("read_file")
        node.db_func_tool = _stub("read_query")
        node.skill_func_tool_instance = _stub("load_skill")
        node.ask_user_tool = _stub("ask_user")
        node.skill_validate_tool = _stub("validate_skill")
        node._session_search_tool = _stub("search_sessions")

        mapping = node._tool_category_map()
        assert mapping["filesystem_tools"][0].name == "read_file"
        assert mapping["db_tools"][0].name == "read_query"
        assert mapping["skills"][0].name == "load_skill"
        assert {t.name for t in mapping["tools"]} == {"ask_user", "validate_skill", "search_sessions"}

    def test_catchall_omitted_when_nothing_to_add(self):
        from datus.agent.node.gen_skill_agentic_node import SkillCreatorAgenticNode

        node = SkillCreatorAgenticNode.__new__(SkillCreatorAgenticNode)
        node.skill_func_tool = None
        node.filesystem_func_tool = None
        node.db_func_tool = None
        node.skill_func_tool_instance = None
        node.ask_user_tool = None
        node.skill_validate_tool = None
        node._session_search_tool = None

        assert node._tool_category_map() == {}
