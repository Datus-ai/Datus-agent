# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.schemas.agent_models.SubAgentConfig``.

Primary focus: the ``subagents`` field normalization validator.
"""

import pytest

from datus.schemas.agent_models import SubAgentConfig


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
