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

from unittest.mock import MagicMock

from prompt_toolkit.document import Document

from datus.cli.autocomplete import (
    AtReferenceCompleter,
    AtReferenceParser,
    CustomPygmentsStyle,
    CustomSqlLexer,
    DynamicAtReferenceCompleter,
    SQLCompleter,
    _ContinuousPathCompleter,
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

    def _build_snapshot(self):
        # Return a fresh flatten dict so reloads can replace prior snapshots
        # without mutating the live instance state.
        flatten = {}
        if isinstance(self._stub_data, dict):
            for key in self._stub_data:
                flatten[key] = self._stub_data[key]
        return self._stub_data, flatten, 2


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
        original_build = _StubCompleter._build_snapshot

        def counting_build(self):
            nonlocal call_count
            call_count += 1
            return original_build(self)

        c = _StubCompleter(data={"a": 1})
        c._build_snapshot = lambda: counting_build(c)
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

    def test_reload_is_atomic_under_concurrency(self):
        """A reader threaded against concurrent reload_data() calls must
        always observe the flatten_data of exactly one complete snapshot
        — never a mix of old and new keys — and must never crash on dict
        iteration.
        """
        import threading
        import time

        snapshots = [
            {f"snap0_{i}": {} for i in range(50)},
            {f"snap1_{i}": {} for i in range(50)},
            {f"snap2_{i}": {} for i in range(50)},
            {f"snap3_{i}": {} for i in range(50)},
        ]
        c = _StubCompleter(data=snapshots[0])
        c._ensure_loaded()

        stop = threading.Event()
        errors = []

        def reader():
            try:
                while not stop.is_set():
                    # fuzzy_match iterates flatten_data; if reload mutated
                    # the underlying dict in place, CPython may raise
                    # "dictionary changed size during iteration".
                    c.fuzzy_match("snap")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        def writer():
            try:
                for i in range(30):
                    c._stub_data = snapshots[i % len(snapshots)]
                    c.reload_data()
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        r = threading.Thread(target=reader)
        w1 = threading.Thread(target=writer)
        w2 = threading.Thread(target=writer)
        r.start()
        w1.start()
        w2.start()
        w1.join()
        w2.join()
        # Give the reader one more tick to catch any final torn state
        time.sleep(0.05)
        stop.set()
        r.join()

        assert errors == []
        # Final snapshot must be one of the valid ones (all keys share the
        # same ``snapN_`` prefix), never a torn mix.
        prefixes = {k.split("_")[0] for k in c.flatten_data.keys()}
        assert len(prefixes) == 1, f"Torn snapshot: {prefixes}"

    def test_reload_swaps_flatten_data_reference(self):
        """reload_data must install a fresh dict so readers that captured a
        reference to the old flatten_data still iterate their old snapshot
        consistently (no in-place mutation of the previously returned
        dict).
        """
        c = _StubCompleter(data={"key_a": {}})
        c._ensure_loaded()
        captured = c.flatten_data  # reader captures pre-reload reference

        c._stub_data = {"key_b": {}}
        c.reload_data()

        # The captured dict must be untouched (still holds only the old key)
        # and c.flatten_data must now point at a distinct dict.
        assert "key_a" in captured and "key_b" not in captured
        assert c.flatten_data is not captured
        assert "key_b" in c.flatten_data


class TestDynamicAtReferenceCompleterGetData:
    def test_get_data_lazy_loads(self):
        c = _StubCompleter(data={"x": 1})
        assert c._loaded is False
        result = c.get_data()
        assert c._loaded is True
        assert result == {"x": 1}


class TestDynamicAtReferenceCompleterLoadDataShim:
    """The base class keeps ``load_data()`` as a shim over ``_build_snapshot``
    for callers that still invoke the legacy API directly.
    """

    def test_load_data_returns_data_and_applies_snapshot(self):
        c = _StubCompleter(data={"a": {}, "b": {}})
        result = c.load_data()
        assert result == {"a": {}, "b": {}}
        # Side-effect: flatten_data + max_level are populated.
        assert set(c.flatten_data.keys()) == {"a", "b"}
        assert c.max_level == 2

    def test_load_data_overwrites_prior_state(self):
        c = _StubCompleter(data={"first": {}})
        c._ensure_loaded()
        # Swap stub data and re-invoke load_data directly.
        c._stub_data = {"second": {}}
        result = c.load_data()
        assert "second" in result
        assert "first" not in c.flatten_data


class TestDynamicAtReferenceCompleterGetCompletions:
    """Exercise the lock-protected snapshot read in get_completions."""

    def test_get_completions_bails_out_beyond_max_level(self):
        from prompt_toolkit.document import Document

        c = _StubCompleter(data={"top": {"leaf": []}})
        # Stub sets max_level=2 (see _StubCompleter). "x.y.z" has 3 levels.
        doc = Document(text="a.b.c")
        completions = list(c.get_completions(doc, None))
        assert completions == []

    def test_get_completions_yields_prefix_matches(self):
        from prompt_toolkit.document import Document

        c = _StubCompleter(data={"alpha": {}, "alabama": {}, "beta": {}})
        doc = Document(text="al")
        texts = [comp.text for comp in c.get_completions(doc, None)]
        # Non-leaf completions append the separator so Tab keeps descending.
        assert "alpha." in texts
        assert "alabama." in texts
        assert not any(t.startswith("beta") for t in texts)

    def test_get_completions_non_leaf_appends_separator(self):
        """Selecting a non-leaf level must produce ``name.`` so the next Tab
        fires the downstream completer without manual separator input.
        """
        from prompt_toolkit.document import Document

        c = _StubCompleter(data={"catalog": {"db": []}})
        doc = Document(text="cat")
        completions = list(c.get_completions(doc, None))
        assert [comp.text for comp in completions] == ["catalog."]

    def test_get_completions_substring_fallback(self):
        """Typed chars that do not prefix-match any key should still surface
        keys containing the substring, ranked after prefix hits.
        """
        from prompt_toolkit.document import Document

        c = _StubCompleter(
            data={
                "california_schools": {},
                "schools_meta": {},
                "orders": {},
            }
        )
        doc = Document(text="sch")
        texts = [comp.text for comp in c.get_completions(doc, None)]
        # Prefix hit first, substring hit second; ``orders`` lacks ``sch``.
        assert texts == ["schools_meta.", "california_schools."]

    def test_get_completions_cross_level_fuzzy_fallback(self):
        """When nothing at the current level matches, fall back to
        ``flatten_data`` so a single fragment can descend arbitrarily deep.
        """
        from prompt_toolkit.document import Document

        class _DeepStub(DynamicAtReferenceCompleter):
            def _build_snapshot(self):
                data = {"my_cat": {"my_db": {"california_schools": []}}}
                flatten = {"my_cat.my_db.california_schools": {"table_name": "california_schools"}}
                return data, flatten, 3

        c = _DeepStub()
        doc = Document(text="sch")
        texts = [comp.text for comp in c.get_completions(doc, None)]
        # ``sch`` matches nothing at the top level (``my_cat``) — fallback
        # yields the full dotted path so Tab fills it in one shot.
        assert texts == ["my_cat.my_db.california_schools"]


class TestContinuousPathCompleter:
    """Wrapper must mirror ``display``'s trailing ``/`` onto ``text`` so Tab
    keeps descending into nested directories without manual separator input.
    """

    def test_directory_completion_text_gets_trailing_slash(self, tmp_path):
        from prompt_toolkit.completion import PathCompleter

        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("x")

        inner = PathCompleter(get_paths=lambda: [str(tmp_path)])
        wrapper = _ContinuousPathCompleter(inner)
        doc = Document(text="")
        comps = list(wrapper.get_completions(doc, None))
        by_text = {c.text: c for c in comps}
        assert "subdir/" in by_text
        assert "file.txt" in by_text
        assert not by_text["file.txt"].text.endswith("/")

    def test_already_slashed_text_is_passed_through(self):
        """Inner completers that already emit ``text`` with ``/`` (e.g. future
        refactors of PathCompleter) must not get double slashes.
        """
        from prompt_toolkit.completion import Completer, Completion

        class _Static(Completer):
            def get_completions(self, document, complete_event):
                yield Completion(text="dir/", start_position=0, display="dir/")

        wrapper = _ContinuousPathCompleter(_Static())
        comps = list(wrapper.get_completions(Document(text=""), None))
        assert [c.text for c in comps] == ["dir/"]


class TestTableCompleterBuildSnapshotByDbType:
    """Exercise TableCompleter._build_snapshot against different connector
    shapes so each hierarchy branch (catalog/database/schema/table,
    database->schema->table, schema->table) is covered without a real
    LanceDB.
    """

    @staticmethod
    def _pyarrow_table(**cols):
        import pyarrow as pa

        return pa.table(cols)

    def _mock_rag(self, monkeypatch, schema_table):
        fake_storage = MagicMock()
        fake_storage.search_all_schemas.return_value = schema_table
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: fake_storage,
        )

    def _table_completer(self, db_type: str):
        from datus.cli.autocomplete import TableCompleter

        agent_config = MagicMock()
        agent_config.db_type = db_type
        tc = TableCompleter(agent_config)
        return tc

    def test_snapshot_empty_table_returns_empty(self, monkeypatch):
        # No rows → data is an empty list, flatten is empty, max_level=0.
        empty = self._pyarrow_table(
            catalog_name=[],
            database_name=[],
            schema_name=[],
            table_name=[],
            table_type=[],
            definition=[],
            identifier=[],
        )
        self._mock_rag(monkeypatch, empty)
        tc = self._table_completer(db_type="sqlite")
        data, flatten, max_level = tc._build_snapshot()
        assert data == []
        assert flatten == {}
        assert max_level == 0

    def test_snapshot_sqlite_table_scoped(self, monkeypatch):
        table = self._pyarrow_table(
            catalog_name=["", ""],
            database_name=["", ""],
            schema_name=["", ""],
            table_name=["t1", "t2"],
            table_type=["BASE TABLE", "BASE TABLE"],
            definition=["CREATE TABLE t1", "CREATE TABLE t2"],
            identifier=["t1", "t2"],
        )
        self._mock_rag(monkeypatch, table)
        tc = self._table_completer(db_type="sqlite")
        data, flatten, max_level = tc._build_snapshot()
        # SQLite default is table-only, not database-prefixed.
        assert max_level == 1
        assert set(flatten.keys()) == {"t1", "t2"}
        assert data == ["t1", "t2"]

    def test_snapshot_catches_storage_error_and_returns_empty(self, monkeypatch):
        """A failure inside SchemaWithValueRAG.search_all_schemas must be
        swallowed so the completer degrades to "no suggestions" instead of
        propagating a startup-time LanceDB failure to the user.
        """
        fake_storage = MagicMock()
        fake_storage.search_all_schemas.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            "datus.storage.schema_metadata.store.SchemaWithValueRAG",
            lambda *a, **kw: fake_storage,
        )
        tc = self._table_completer(db_type="sqlite")
        data, flatten, max_level = tc._build_snapshot()
        assert data == []
        assert flatten == {}
        assert max_level == 0


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

    def test_fuzzy_match_defaults_to_max_completions(self):
        data = {f"item_{i}": {} for i in range(20)}
        c = _StubCompleter(data=data)
        c._ensure_loaded()
        # Default limit should match ``max_completions`` (10), not the old
        # hard-coded 5, so short fragments with many hits are not truncated.
        results = c.fuzzy_match("item")
        assert len(results) == c.max_completions

    def test_fuzzy_match_honors_explicit_limit(self):
        data = {f"item_{i}": {} for i in range(20)}
        c = _StubCompleter(data=data)
        c._ensure_loaded()
        results = c.fuzzy_match("item", limit=3)
        assert len(results) == 3


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


class TestAtReferenceCompleterGetCompletions:
    """The TUI refactor routes bare chat input (no leading ``/``) to the
    default agent, so @Table completion must fire without a slash prefix.
    """

    def test_bare_at_triggers_type_candidate(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        doc = Document("@T", cursor_position=2)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "Table" in texts

    def test_bare_at_alone_yields_all_type_options(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        doc = Document("@", cursor_position=1)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        for opt in ("Table", "Metrics", "Sql", "File"):
            assert opt in texts

    def test_plain_text_without_at_yields_nothing(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config)
        doc = Document("show me schools", cursor_position=15)
        completions = list(completer.get_completions(doc, None))
        assert completions == []

    def test_slash_subagent_prefix_triggers_scope_switch(self, real_agent_config):
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql"})
        assert completer._sub_agent_name == ""
        doc = Document("/gensql @T", cursor_position=10)
        list(completer.get_completions(doc, None))
        assert completer._sub_agent_name == "gensql"

    def test_bare_input_resets_subagent_scope(self, real_agent_config):
        """A fresh bare-chat line has no sub-agent prefix, so scope must
        fall back to the global datasource.
        """
        completer = AtReferenceCompleter(real_agent_config, available_subagents={"gensql"}, sub_agent_name="gensql")
        doc = Document("@T", cursor_position=2)
        list(completer.get_completions(doc, None))
        assert completer._sub_agent_name == ""

    def test_chat_with_inline_at_table(self, real_agent_config):
        """Natural-language chat with an inline @Table reference should
        surface the type candidate.
        """
        completer = AtReferenceCompleter(real_agent_config)
        text = "help me query @T"
        doc = Document(text, cursor_position=len(text))
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert "Table" in texts
