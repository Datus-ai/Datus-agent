# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.schemas.agent_models.SubAgentConfig``.

Primary focus: the ``subagents`` field normalization validator.
"""

import pytest

from datus.schemas.agent_models import ScopedContext, SubAgentConfig, merge_scoped_contexts


@pytest.mark.ci
class TestSubAgentsFieldNormalization:
    """Tests for the ``subagents`` field validator / normalizer."""

    def test_none_stays_none(self):
        cfg = SubAgentConfig(subagents=None)
        assert cfg.subagents is None
        assert cfg.subagent_list == []

    def test_empty_string_collapses_to_none(self):
        cfg = SubAgentConfig(subagents="")
        assert cfg.subagents is None
        assert cfg.subagent_list == []

    def test_whitespace_only_collapses_to_none(self):
        cfg = SubAgentConfig(subagents="   ")
        assert cfg.subagents is None

    def test_wildcard_alone(self):
        cfg = SubAgentConfig(subagents="*")
        assert cfg.subagents == "*"
        assert cfg.subagent_list == ["*"]

    def test_wildcard_mixed_collapses_to_wildcard(self):
        """``*, foo, bar`` is ambiguous -> collapse to the canonical ``*``."""
        cfg = SubAgentConfig(subagents="*, gen_sql, explore")
        assert cfg.subagents == "*"
        assert cfg.subagent_list == ["*"]

    def test_wildcard_at_end_mixed_still_collapses(self):
        cfg = SubAgentConfig(subagents="gen_sql, *")
        assert cfg.subagents == "*"

    def test_explicit_list(self):
        cfg = SubAgentConfig(subagents="gen_sql, explore")
        assert cfg.subagents == "gen_sql,explore"
        assert cfg.subagent_list == ["gen_sql", "explore"]

    def test_duplicates_removed(self):
        cfg = SubAgentConfig(subagents="gen_sql, gen_sql, explore, gen_sql")
        assert cfg.subagent_list == ["gen_sql", "explore"]

    def test_stray_whitespace_and_empty_tokens_stripped(self):
        cfg = SubAgentConfig(subagents="  gen_sql , , explore ,")
        assert cfg.subagent_list == ["gen_sql", "explore"]

    def test_single_entry(self):
        cfg = SubAgentConfig(subagents="explore")
        assert cfg.subagents == "explore"
        assert cfg.subagent_list == ["explore"]

    def test_non_string_rejected(self):
        """Non-string values must fail validation — ``subagents`` is a string field."""
        from pydantic import ValidationError

        with pytest.raises((ValidationError, TypeError)):
            SubAgentConfig(subagents=["gen_sql"])

    def test_as_payload_omits_subagents_when_none(self):
        cfg = SubAgentConfig(system_prompt="x", subagents=None)
        payload = cfg.as_payload()
        assert "subagents" not in payload

    def test_as_payload_includes_subagents_when_set(self):
        cfg = SubAgentConfig(system_prompt="x", subagents="gen_sql, explore")
        payload = cfg.as_payload()
        assert payload["subagents"] == "gen_sql,explore"


@pytest.mark.ci
class TestScopedContextMerge:
    """Whole-segment override semantics for parent → child scoped_context inheritance."""

    PARENT = ScopedContext(
        datasource="db_p",
        tables="public.users,public.orders",
        metrics="rev",
        sqls="q1",
    )

    def test_child_none_inherits_parent(self):
        merged = merge_scoped_contexts(self.PARENT, None)
        assert merged.tables == "public.users,public.orders"
        assert merged.metrics == "rev"
        assert merged.sqls == "q1"
        assert merged.datasource == "db_p"

    def test_child_empty_inherits_parent(self):
        empty_child = ScopedContext()
        merged = merge_scoped_contexts(self.PARENT, empty_child)
        assert merged.tables == self.PARENT.tables
        assert merged.metrics == self.PARENT.metrics
        assert merged.datasource == "db_p"

    def test_child_only_datasource_still_inherits_parent_kb(self):
        # is_empty ignores datasource → child considered empty → inherit parent
        ds_only = ScopedContext(datasource="db_c")
        merged = merge_scoped_contexts(self.PARENT, ds_only)
        assert merged.tables == self.PARENT.tables
        assert merged.metrics == self.PARENT.metrics

    def test_child_with_tables_replaces_whole_segment(self):
        child = ScopedContext(tables="public.products")
        merged = merge_scoped_contexts(self.PARENT, child)
        assert merged.tables == "public.products"
        # Whole-segment override: parent metrics/sqls dropped.
        assert merged.metrics is None
        assert merged.sqls is None

    def test_child_with_metrics_replaces_whole_segment(self):
        child = ScopedContext(metrics="orders_per_day")
        merged = merge_scoped_contexts(self.PARENT, child)
        assert merged.metrics == "orders_per_day"
        assert merged.tables is None

    def test_both_none_returns_empty(self):
        merged = merge_scoped_contexts(None, None)
        assert merged.is_empty
        assert merged.datasource == ""

    def test_parent_none_yields_child_copy(self):
        child = ScopedContext(tables="t1")
        merged = merge_scoped_contexts(None, child)
        assert merged.tables == "t1"
        # Returned object is a copy, not the same instance.
        assert merged is not child

    def test_parent_returned_as_copy(self):
        merged = merge_scoped_contexts(self.PARENT, None)
        assert merged is not self.PARENT
        # Mutating the merged copy doesn't touch the parent.
        merged.tables = "xxx"
        assert self.PARENT.tables == "public.users,public.orders"

    def test_scoped_context_merge_with_method_delegates(self):
        merged = self.PARENT.merge_with(None)
        assert merged.tables == self.PARENT.tables
        assert merged is not self.PARENT


@pytest.mark.ci
class TestSubAgentConfigEffectiveScopedContext:
    """``SubAgentConfig.with_effective_scoped_context`` returns an immutable copy."""

    def test_returns_new_instance_with_merged_scope(self):
        parent = ScopedContext(tables="public.users")
        child = SubAgentConfig(system_prompt="x")  # no scope → inherit parent
        effective = child.with_effective_scoped_context(parent)
        assert effective is not child
        assert isinstance(effective.scoped_context, ScopedContext)
        assert effective.scoped_context.tables == "public.users"
        # Original child unchanged.
        assert child.scoped_context is None

    def test_child_scope_overrides_parent_segment(self):
        parent = ScopedContext(tables="public.users", metrics="rev")
        child = SubAgentConfig(
            system_prompt="x",
            scoped_context=ScopedContext(tables="public.orders"),
        )
        effective = child.with_effective_scoped_context(parent)
        assert effective.scoped_context.tables == "public.orders"
        assert effective.scoped_context.metrics is None  # whole-segment replacement

    def test_no_parent_keeps_child_scope(self):
        child_sc = ScopedContext(tables="public.users")
        child = SubAgentConfig(system_prompt="x", scoped_context=child_sc)
        effective = child.with_effective_scoped_context(None)
        assert effective.scoped_context.tables == "public.users"
