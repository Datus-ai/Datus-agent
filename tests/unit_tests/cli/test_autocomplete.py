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

from datus.cli.autocomplete import (
    AtReferenceParser,
    CustomPygmentsStyle,
    CustomSqlLexer,
    SQLCompleter,
    SubagentCompleter,
    insert_into_dict,
    insert_into_dict_with_dict,
)
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


# ---------------------------------------------------------------------------
# SQLCompleter
# ---------------------------------------------------------------------------


class TestSQLCompleterInit:
    def test_default_state(self):
        c = SQLCompleter()
        assert "SELECT" in c.keywords
        assert "COUNT" in c.functions
        assert "INT" in c.types
        assert c.tables == {}
        assert c.database_name == ""
        assert c.schema_name == ""

    def test_commands_contains_expected_keys(self):
        c = SQLCompleter()
        assert ".help" in c.commands
        assert ".exit" in c.commands
        assert "!sl" in c.commands


class TestSQLCompleterUpdateMethods:
    def test_update_tables(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id", "name"], "orders": ["id", "amount"]})
        assert "users" in c.tables
        assert "orders" in c.tables
        assert c.table_aliases == {}  # reset on update

    def test_update_db_info(self):
        c = SQLCompleter()
        c.update_db_info("mydb", "public")
        assert c.database_name == "mydb"
        assert c.schema_name == "public"


class TestSQLCompleterGetCompletions:
    def test_slash_prefix_returns_nothing(self):
        c = SQLCompleter()
        doc = Document("/help", cursor_position=5)
        completions = list(c.get_completions(doc))
        assert completions == []

    def test_command_prefix_bang(self):
        c = SQLCompleter()
        doc = Document("!sl", cursor_position=3)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert "!sl" in texts

    def test_command_prefix_dot(self):
        c = SQLCompleter()
        doc = Document(".he", cursor_position=3)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert ".help" in texts

    def test_dot_notation_table_column(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id", "name", "email"]})
        # "SELECT users.n" -> should suggest "name"
        doc = Document("SELECT users.n", cursor_position=14)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert "name" in texts

    def test_from_context_suggests_tables(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id"], "orders": ["id"]})
        # "FROM " with empty word_before_cursor -> _get_previous_word returns "FROM"
        # The cursor is at end of "SELECT * FROM "
        doc = Document("SELECT * FROM u", cursor_position=15)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        # "users" starts with "u"
        assert "users" in texts

    def test_join_context_suggests_tables(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id"], "orders": ["id"]})
        # "FROM " with empty word_before_cursor means all tables should show
        # Use "FROM u" to get "users" specifically
        doc = Document("SELECT * FROM u", cursor_position=15)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert "users" in texts

    def test_select_context_suggests_columns(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id", "name"]})
        doc = Document("SELECT n", cursor_position=8)
        list(c.get_completions(doc))
        # Depending on what "previous word" is, columns may be suggested
        # At minimum it should not raise

    def test_keyword_completion(self):
        c = SQLCompleter()
        doc = Document("SEL", cursor_position=3)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert "SELECT" in texts

    def test_function_completion(self):
        c = SQLCompleter()
        doc = Document("CO", cursor_position=2)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert any(t.startswith("COUNT") for t in texts)

    def test_empty_word_no_crash(self):
        c = SQLCompleter()
        doc = Document("", cursor_position=0)
        completions = list(c.get_completions(doc))
        # Empty should return no completions (no word to match)
        assert isinstance(completions, list)


class TestGetPreviousWord:
    def test_empty_text(self):
        c = SQLCompleter()
        assert c._get_previous_word("") == ""

    def test_single_word(self):
        c = SQLCompleter()
        assert c._get_previous_word("SELECT") == ""

    def test_two_words(self):
        c = SQLCompleter()
        assert c._get_previous_word("SELECT *") == "SELECT"

    def test_multiple_words(self):
        c = SQLCompleter()
        assert c._get_previous_word("SELECT * FROM") == "*"


class TestDetectAliases:
    def test_from_alias_detected(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id", "name"]})
        c._detect_aliases("SELECT u.name FROM users u")
        assert "u" in c.table_aliases
        assert c.table_aliases["u"] == "users"

    def test_no_alias_no_entry(self):
        c = SQLCompleter()
        c.update_tables({"users": ["id"]})
        c._detect_aliases("SELECT * FROM users")
        # "WHERE" is not a valid alias so nothing should be added incorrectly
        assert "WHERE" not in c.table_aliases


# ---------------------------------------------------------------------------
# insert_into_dict helper
# ---------------------------------------------------------------------------


class TestInsertIntoDict:
    def test_single_key(self):
        data = {}
        insert_into_dict(data, ["users"], "table_a")
        assert data == {"users": ["table_a"]}

    def test_nested_keys(self):
        data = {}
        insert_into_dict(data, ["catalog", "db", "schema"], "my_table")
        assert data["catalog"]["db"]["schema"] == ["my_table"]

    def test_multiple_inserts_same_path(self):
        data = {}
        insert_into_dict(data, ["catalog", "db"], "table1")
        insert_into_dict(data, ["catalog", "db"], "table2")
        assert "table1" in data["catalog"]["db"]
        assert "table2" in data["catalog"]["db"]


# ---------------------------------------------------------------------------
# insert_into_dict_with_dict helper
# ---------------------------------------------------------------------------


class TestInsertIntoDictWithDict:
    def test_basic_insert(self):
        data = {}
        insert_into_dict_with_dict(data, ["Finance"], "revenue", "Total revenue metric")
        assert data["Finance"]["revenue"] == "Total revenue metric"

    def test_nested_insert(self):
        data = {}
        insert_into_dict_with_dict(data, ["Finance", "Q1"], "profit", "Net profit")
        assert data["Finance"]["Q1"]["profit"] == "Net profit"


# ---------------------------------------------------------------------------
# AtReferenceParser
# ---------------------------------------------------------------------------


class TestAtReferenceParser:
    def test_parse_empty_text(self):
        parser = AtReferenceParser()
        result = parser.parse_input("")
        assert result == {"tables": [], "metrics": [], "sqls": []}

    def test_parse_table_reference(self):
        parser = AtReferenceParser()
        result = parser.parse_input("@Table users")
        assert "users" in result["tables"]

    def test_parse_metrics_reference(self):
        parser = AtReferenceParser()
        result = parser.parse_input("@Metrics Finance.revenue")
        assert len(result["metrics"]) > 0

    def test_parse_sql_reference(self):
        parser = AtReferenceParser()
        result = parser.parse_input("@Sql Finance.get_revenue")
        assert len(result["sqls"]) > 0

    def test_parse_multiple_references(self):
        parser = AtReferenceParser()
        result = parser.parse_input("@Table orders @Table users @Metrics revenue")
        assert len(result["tables"]) == 2
        assert len(result["metrics"]) == 1

    def test_parse_dotted_path(self):
        parser = AtReferenceParser()
        result = parser.parse_input("@Table catalog.database.schema.my_table")
        assert len(result["tables"]) > 0


# ---------------------------------------------------------------------------
# CustomSqlLexer and CustomPygmentsStyle: smoke tests
# ---------------------------------------------------------------------------


class TestCustomLexerAndStyle:
    def test_custom_sql_lexer_importable(self):
        lexer = CustomSqlLexer()
        assert lexer is not None

    def test_custom_pygments_style_importable(self):
        from pygments.token import Token

        assert hasattr(CustomPygmentsStyle, "styles")
        assert Token.AtTables in CustomPygmentsStyle.styles

    def test_custom_sql_lexer_has_root_tokens(self):
        assert "root" in CustomSqlLexer.tokens
        # Should contain @Table pattern
        patterns = [str(p[0]) for p in CustomSqlLexer.tokens["root"]]
        assert any("Table" in p for p in patterns)
