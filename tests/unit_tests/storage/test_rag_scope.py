# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.rag_scope."""

from unittest.mock import MagicMock

import pytest
from datus_storage_base.conditions import build_where

from datus.storage.rag_scope import _build_sub_agent_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent_config(sub_agent_configs=None, db_type=""):
    """Create a mock AgentConfig with optional sub-agent configs."""
    config = MagicMock()
    config.db_type = db_type
    config.sub_agent_config = MagicMock(side_effect=lambda name: (sub_agent_configs or {}).get(name, {}))
    return config


def _mock_storage(has_subject_tree=False):
    """Create a mock storage, optionally with a subject_tree."""
    storage = MagicMock()
    if has_subject_tree:
        storage.subject_tree = MagicMock()
    else:
        storage.subject_tree = None
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSubAgentFilter:
    """Tests for _build_sub_agent_filter."""

    @pytest.mark.parametrize(
        "sub_agent_name,sub_agent_configs,expected_description",
        [
            (None, None, "no sub_agent_name"),
            ("unknown_agent", None, "sub-agent name with no matching config"),
            ("team_a", {"team_a": {"system_prompt": "team_a"}}, "config without scoped_context"),
        ],
    )
    def test_returns_none_when_no_scope(self, sub_agent_name, sub_agent_configs, expected_description):
        """_build_sub_agent_filter returns None when there is no effective scope."""
        config = _mock_agent_config(sub_agent_configs=sub_agent_configs)
        result = _build_sub_agent_filter(config, sub_agent_name, _mock_storage(), "tables")
        assert result is None, f"Expected None for case: {expected_description}"

    def test_table_scope_builds_filter(self):
        """Sub-agent with tables scoped context -> table filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": "public.users"}}},
        )
        storage = _mock_storage()
        result = _build_sub_agent_filter(config, "team_a", storage, "tables")
        assert result is not None
        clause = build_where(result)
        assert "users" in clause

    def test_subject_scope_without_tree_raises(self):
        """Subject-based scope without subject_tree on storage -> raises DatusException."""
        import pytest

        from datus.utils.exceptions import DatusException

        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=False)
        with pytest.raises(DatusException, match="subject_tree"):
            _build_sub_agent_filter(config, "team_a", storage, "metrics")

    def test_subject_scope_with_tree_builds_filter(self):
        """Subject-based scope with subject_tree -> builds subject filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=True)
        storage.subject_tree.get_matched_children_id.return_value = [1, 2]
        result = _build_sub_agent_filter(config, "team_a", storage, "metrics")
        assert result is not None

    def test_empty_scope_value_returns_none(self):
        """Empty scope value -> no filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": ""}}},
        )
        result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
        assert result is None


class TestRuntimeOverrideIntegration:
    """Verify ``AgentConfig.sub_agent_config`` consults the ContextVar override.

    These tests exercise the real ``AgentConfig.sub_agent_config`` method (not the
    test mock above) so the runtime override path is covered end-to-end.
    """

    def _real_config(self, agentic_nodes):
        # Bypass __init__ to avoid loading agent.yml; only the method we test is needed.
        from datus.configuration.agent_config import AgentConfig

        config = AgentConfig.__new__(AgentConfig)
        config.agentic_nodes = agentic_nodes
        return config

    def test_yaml_path_when_no_override(self):
        config = self._real_config({"team_a": {"scoped_context": {"tables": "public.users"}}})
        result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
        assert result is not None
        assert "users" in build_where(result)

    def test_runtime_override_takes_precedence_over_yaml(self):
        from datus.configuration.scoped_context_overrides import effective_subagent
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        config = self._real_config({"team_a": {"scoped_context": {"tables": "public.users"}}})
        override_cfg = SubAgentConfig(
            system_prompt="x",
            scoped_context=ScopedContext(tables="public.orders"),
        )
        with effective_subagent("team_a", override_cfg):
            result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
        assert result is not None
        clause = build_where(result)
        assert "orders" in clause
        assert "users" not in clause

    def test_override_supplies_scope_when_yaml_has_none(self):
        from datus.configuration.scoped_context_overrides import effective_subagent
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        # Builtin subagent: no yaml entry; parent inheritance should still produce a filter.
        config = self._real_config({})
        override_cfg = SubAgentConfig(
            system_prompt="x",
            scoped_context=ScopedContext(tables="public.users"),
        )
        with effective_subagent("gen_metrics", override_cfg):
            result = _build_sub_agent_filter(config, "gen_metrics", _mock_storage(), "tables")
        assert result is not None
        assert "users" in build_where(result)

    @pytest.mark.asyncio
    async def test_override_isolated_per_asyncio_task(self):
        import asyncio

        from datus.configuration.scoped_context_overrides import effective_subagent
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        config = self._real_config({"team_a": {"scoped_context": {"tables": "public.users"}}})

        async def run(table: str) -> str:
            cfg = SubAgentConfig(system_prompt="x", scoped_context=ScopedContext(tables=table))
            with effective_subagent("team_a", cfg):
                await asyncio.sleep(0.01)
                result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
            return build_where(result)

        c1, c2 = await asyncio.gather(run("table_one"), run("table_two"))
        assert "table_one" in c1 and "table_two" not in c1
        assert "table_two" in c2 and "table_one" not in c2

    def test_override_preserves_non_subagentconfig_yaml_keys(self):
        """Override must layer on top of YAML so keys outside SubAgentConfig
        (model, max_turns, permissions, ...) survive when scoped_context is overridden.
        """
        from datus.configuration.scoped_context_overrides import effective_subagent
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        config = self._real_config(
            {
                "team_a": {
                    "model": "gpt-4.1",
                    "max_turns": 7,
                    "permissions": {"read": True},
                    "scoped_context": {"tables": "public.users"},
                }
            }
        )
        override_cfg = SubAgentConfig(
            system_prompt="x",
            scoped_context=ScopedContext(tables="public.orders"),
        )
        with effective_subagent("team_a", override_cfg):
            merged = config.sub_agent_config("team_a")
        # Override fields win.
        assert merged["scoped_context"]["tables"] == "public.orders"
        # YAML-only fields preserved.
        assert merged["model"] == "gpt-4.1"
        assert merged["max_turns"] == 7
        assert merged["permissions"] == {"read": True}

    def test_override_only_overwrites_explicitly_set_fields(self):
        """Fields not explicitly set on the override SubAgentConfig must not
        clobber the YAML entry with their pydantic defaults.
        """
        from datus.configuration.scoped_context_overrides import effective_subagent
        from datus.schemas.agent_models import ScopedContext, SubAgentConfig

        config = self._real_config(
            {
                "team_a": {
                    "system_prompt": "yaml_prompt",
                    "tools": "list_tables,describe_table",
                    "scoped_context": {"tables": "public.users"},
                }
            }
        )
        # Only scoped_context is explicitly set on the override.
        override_cfg = SubAgentConfig(scoped_context=ScopedContext(tables="public.orders"))
        with effective_subagent("team_a", override_cfg):
            merged = config.sub_agent_config("team_a")
        assert merged["scoped_context"]["tables"] == "public.orders"
        # YAML system_prompt and tools survive because override didn't set them.
        assert merged["system_prompt"] == "yaml_prompt"
        assert merged["tools"] == "list_tables,describe_table"
