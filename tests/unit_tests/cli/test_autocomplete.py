# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/autocomplete.py — SubagentCompleter.

Tests cover:
- SubagentCompleter initialization and refresh
- _load_subagents: filtering by namespace, excluding chat and SYS_SUB_AGENTS
- get_completions: basic completion for subagent names

NO MOCK EXCEPT LLM.
"""

from prompt_toolkit.document import Document

from datus.cli.autocomplete import SubagentCompleter
from datus.utils.constants import SYS_SUB_AGENTS


class TestSubagentCompleterInit:
    """Tests for SubagentCompleter initialization."""

    def test_init_creates_completer(self, real_agent_config):
        """SubagentCompleter initializes with agent_config and loads subagents."""
        completer = SubagentCompleter(real_agent_config)

        assert completer.agent_config is real_agent_config
        assert isinstance(completer._available_subagents, list)
        assert len(completer._available_subagents) > 0

    def test_loaded_subagents_include_sys_sub_agents(self, real_agent_config):
        """Loaded subagents include all SYS_SUB_AGENTS."""
        completer = SubagentCompleter(real_agent_config)

        for sys_sub in SYS_SUB_AGENTS:
            assert sys_sub in completer._available_subagents

    def test_loaded_subagents_exclude_chat(self, real_agent_config):
        """Loaded subagents exclude the 'chat' node."""
        completer = SubagentCompleter(real_agent_config)

        assert "chat" not in completer._available_subagents

    def test_loaded_subagents_include_custom_nodes(self, real_agent_config):
        """Custom agentic_nodes not in SYS_SUB_AGENTS and not 'chat' are included."""
        completer = SubagentCompleter(real_agent_config)

        # real_agent_config has "gensql", "compare", "gen_report" which are not in SYS_SUB_AGENTS
        assert "gensql" in completer._available_subagents
        assert "compare" in completer._available_subagents
        assert "gen_report" in completer._available_subagents


class TestSubagentCompleterLoadSubagents:
    """Tests for _load_subagents filtering logic."""

    def test_namespace_filtered_subagent_excluded(self, real_agent_config):
        """Sub-agent with different namespace is excluded from completions."""
        # Add a sub-agent with a different namespace
        real_agent_config.agentic_nodes["foreign_sub"] = {
            "system_prompt": "foreign_sub",
            "scoped_context": {
                "namespace": "other_namespace",
            },
        }
        completer = SubagentCompleter(real_agent_config)

        assert "foreign_sub" not in completer._available_subagents

    def test_namespace_matching_subagent_included(self, real_agent_config):
        """Sub-agent with matching namespace is included in completions."""
        real_agent_config.agentic_nodes["local_sub"] = {
            "system_prompt": "local_sub",
            "scoped_context": {
                "namespace": real_agent_config.current_namespace,
            },
        }
        completer = SubagentCompleter(real_agent_config)

        assert "local_sub" in completer._available_subagents

    def test_subagent_without_scoped_context_included(self, real_agent_config):
        """Sub-agent without scoped_context is included (no namespace restriction)."""
        real_agent_config.agentic_nodes["unrestricted_sub"] = {
            "system_prompt": "unrestricted_sub",
        }
        completer = SubagentCompleter(real_agent_config)

        assert "unrestricted_sub" in completer._available_subagents

    def test_refresh_reloads_subagents(self, real_agent_config):
        """refresh() reloads the subagent list reflecting config changes."""
        completer = SubagentCompleter(real_agent_config)
        original_count = len(completer._available_subagents)

        # Add a new sub-agent
        real_agent_config.agentic_nodes["new_sub"] = {
            "system_prompt": "new_sub",
        }
        completer.refresh()

        assert len(completer._available_subagents) == original_count + 1
        assert "new_sub" in completer._available_subagents


class TestSubagentCompleterGetCompletions:
    """Tests for SubagentCompleter.get_completions."""

    def test_get_completions_returns_matches(self, real_agent_config):
        """get_completions returns matching subagent names for /gen prefix."""
        completer = SubagentCompleter(real_agent_config)
        document = Document("/gen", cursor_position=4)

        completions = list(completer.get_completions(document))

        # Should match gen_* subagents
        texts = [c.text for c in completions]
        assert len(texts) > 0

    def test_get_completions_slash_only_returns_all(self, real_agent_config):
        """get_completions with '/' returns all subagents."""
        completer = SubagentCompleter(real_agent_config)
        document = Document("/", cursor_position=1)

        completions = list(completer.get_completions(document))

        assert len(completions) == len(completer._available_subagents)

    def test_get_completions_no_slash_returns_nothing(self, real_agent_config):
        """get_completions without leading slash returns no completions."""
        completer = SubagentCompleter(real_agent_config)
        document = Document("gen", cursor_position=3)

        completions = list(completer.get_completions(document))

        assert len(completions) == 0
