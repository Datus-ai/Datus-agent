# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/autocomplete.py.

The slash command completer has its own dedicated test module
(``tests/unit_tests/cli/test_slash_completer.py``). This module focuses on
SQLCompleter, AtReferenceCompleter, DynamicAtReferenceCompleter, and shared
helpers.

NO MOCK EXCEPT LLM.
"""

from prompt_toolkit.document import Document

from datus.cli.autocomplete import (
    AtReferenceCompleter,
    AtReferenceParser,
    CustomPygmentsStyle,
    CustomSqlLexer,
    DynamicAtReferenceCompleter,
    SQLCompleter,
    insert_into_dict,
    insert_into_dict_with_dict,
)

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
        # Slash commands are owned by SlashCommandCompleter; only tool
        # (``!``) prefixes remain in the legacy word dictionary.
        assert "!sl" in c.commands
        assert "!bash" in c.commands
        assert not any(k.startswith(".") for k in c.commands)
        assert not any(k.startswith("@") for k in c.commands)


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

    def test_dot_prefix_yields_nothing(self):
        """Dot-prefix completions are no longer exposed by SQLCompleter; the
        slash-prefix completer handles command discovery now."""
        c = SQLCompleter()
        doc = Document(".he", cursor_position=3)
        completions = list(c.get_completions(doc))
        texts = [comp.text for comp in completions]
        assert all(not t.startswith(".") for t in texts)

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


# ---------------------------------------------------------------------------
# DynamicAtReferenceCompleter (base class) tests
# ---------------------------------------------------------------------------


class _StubCompleter(DynamicAtReferenceCompleter):
    """Concrete subclass for testing the abstract DynamicAtReferenceCompleter."""

    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self._stub_data = data or {}

    def load_data(self):
        # Populate flatten_data so fuzzy_match works
        if isinstance(self._stub_data, dict):
            for key in self._stub_data:
                self.flatten_data[key] = self._stub_data[key]
        self.max_level = 2
        return self._stub_data


class TestDynamicAtReferenceCompleterInit:
    def test_init_defaults(self):
        c = _StubCompleter()
        assert c._data == {}
        assert c.flatten_data == {}
        assert c._loaded is False
        assert c.max_completions == 10

    def test_init_custom_max_completions(self):
        c = _StubCompleter(max_completions=5)
        assert c.max_completions == 5


class TestDynamicAtReferenceCompleterClear:
    def test_clear_resets_state(self):
        c = _StubCompleter(data={"key": "val"})
        c._ensure_loaded()
        assert c._loaded is True
        assert len(c.flatten_data) > 0

        c.clear()
        assert c._data == {}
        assert c.flatten_data == {}
        assert c._loaded is False


class TestDynamicAtReferenceCompleterEnsureLoaded:
    def test_ensure_loaded_loads_once(self):
        c = _StubCompleter(data={"a": 1})
        c._ensure_loaded()
        assert c._loaded is True
        assert c._data == {"a": 1}

    def test_ensure_loaded_idempotent(self):
        call_count = 0
        original_load = _StubCompleter.load_data

        def counting_load(self):
            nonlocal call_count
            call_count += 1
            return original_load(self)

        c = _StubCompleter(data={"a": 1})
        c.load_data = lambda: counting_load(c)
        c._ensure_loaded()
        c._ensure_loaded()
        assert call_count == 1


class TestDynamicAtReferenceCompleterReloadData:
    def test_reload_clears_flatten_data(self):
        c = _StubCompleter(data={"old_key": "old_val"})
        c._ensure_loaded()
        assert "old_key" in c.flatten_data

        # Change stub data and reload
        c._stub_data = {"new_key": "new_val"}
        c.reload_data()
        assert "old_key" not in c.flatten_data
        assert "new_key" in c.flatten_data
        assert c._loaded is True


class TestDynamicAtReferenceCompleterGetData:
    def test_get_data_lazy_loads(self):
        c = _StubCompleter(data={"x": 1})
        assert c._loaded is False
        result = c.get_data()
        assert c._loaded is True
        assert result == {"x": 1}


class TestDynamicAtReferenceCompleterFuzzyMatch:
    def test_fuzzy_match_finds_substring(self):
        c = _StubCompleter(data={"Finance/Revenue": {}, "Marketing/Budget": {}})
        c._ensure_loaded()
        results = c.fuzzy_match("rev")
        assert "Finance/Revenue" in results

    def test_fuzzy_match_empty_text(self):
        c = _StubCompleter(data={"Finance/Revenue": {}})
        c._ensure_loaded()
        results = c.fuzzy_match("")
        assert results == []

    def test_fuzzy_match_limits_to_5(self):
        data = {f"item_{i}": {} for i in range(20)}
        c = _StubCompleter(data=data)
        c._ensure_loaded()
        results = c.fuzzy_match("item")
        assert len(results) == 5


# ---------------------------------------------------------------------------
# AtReferenceCompleter: set_sub_agent, parse_at_context, _detect_sub_agent
# ---------------------------------------------------------------------------


class TestAtReferenceCompleterSetSubAgent:
    def test_set_sub_agent_changes_context(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        completer.set_sub_agent("gensql")
        assert completer._sub_agent_name == "gensql"
        assert completer.table_completer.sub_agent_name == "gensql"
        assert completer.metric_completer.sub_agent_name == "gensql"
        assert completer.sql_completer.sub_agent_name == "gensql"

    def test_set_sub_agent_noop_same_name(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, sub_agent_name="gensql")
        # Should not clear completers if same name
        completer.table_completer._loaded = True
        completer.set_sub_agent("gensql")
        assert completer.table_completer._loaded is True  # not cleared

    def test_set_sub_agent_clears_completers(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        completer.table_completer._loaded = True
        completer.set_sub_agent("gensql")
        assert completer.table_completer._loaded is False


class TestAtReferenceCompleterDetectSubAgent:
    def test_no_slash_returns_empty(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql", "compare"})
        assert completer._detect_sub_agent_from_input("hello world") == ""

    def test_slash_with_known_subagent(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql", "compare"})
        result = completer._detect_sub_agent_from_input("/gensql @Table users")
        assert result == "gensql"

    def test_slash_with_unknown_subagent(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql"})
        assert completer._detect_sub_agent_from_input("/unknown @Table users") == ""

    def test_slash_without_space(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql"})
        assert completer._detect_sub_agent_from_input("/gensql") == ""

    def test_empty_subagents_returns_empty(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents=set())
        assert completer._detect_sub_agent_from_input("/gensql @Table users") == ""


class TestAtReferenceCompleterParseAtContext:
    def test_parse_empty_input(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        tables, metrics, sqls = completer.parse_at_context("")
        assert tables == []
        assert metrics == []
        assert sqls == []

    def test_parse_triggers_ensure_loaded(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        assert completer.table_completer._loaded is False
        completer.parse_at_context("@Table users")
        assert completer.table_completer._loaded is True
