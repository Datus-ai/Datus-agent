# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.cli.status_bar."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from datus.cli.status_bar import StatusBarProvider, StatusBarState, _humanize_tokens
from datus.utils.constants import DBType


class TestHumanizeTokens:
    def test_zero_and_negative_render_zero_k(self):
        assert _humanize_tokens(0) == "0K"
        assert _humanize_tokens(-5) == "0K"

    def test_sub_kilo_values_show_one_decimal(self):
        # 512 / 1024 = 0.5
        assert _humanize_tokens(512) == "0.5K"
        assert _humanize_tokens(1024) == "1.0K"
        # 5000 / 1024 ≈ 4.88
        assert _humanize_tokens(5000) == "4.9K"

    def test_ten_k_and_above_rounds_to_integer_k(self):
        # 10240 / 1024 == 10 → "10K"
        assert _humanize_tokens(10_240) == "10K"
        # 54321 / 1024 ≈ 53.05 → "53K"
        assert _humanize_tokens(54_321) == "53K"
        # 1024 * 1024 = 1048576 → "1024K"
        assert _humanize_tokens(1_048_576) == "1024K"

    def test_invalid_input_returns_zero_k(self):
        assert _humanize_tokens(None) == "0K"
        assert _humanize_tokens("not a number") == "0K"


class TestStatusBarState:
    def test_format_plain_renders_brand_and_values_without_labels(self):
        state = StatusBarState(
            agent="chat",
            model="claude-sonnet-4-6",
            cumulative_tokens=12_288,  # 12K exactly
            context_used=55_296,  # 54K exactly
            context_total=1_048_576,  # 1024K exactly
        )
        text = state.format_plain()
        assert "Datus" in text
        assert "chat" in text
        assert "claude-sonnet-4-6" in text
        assert "12K" in text
        assert "54K/1024K 5%" in text
        assert " │ " in text
        # Labels must be gone — only values remain
        assert "Agent " not in text
        assert "Model " not in text
        assert "Tokens " not in text
        assert "Ctx " not in text

    def test_format_plain_zero_context_uses_placeholder(self):
        state = StatusBarState()
        text = state.format_plain()
        assert "chat" in text
        assert " - " in text  # model "-" rendered as value
        assert "0K" in text
        assert "0K/0K 0%" in text

    def test_tokens_display_includes_cached_suffix(self):
        state = StatusBarState(cumulative_tokens=28_672, cached_tokens=20_480)
        assert state.tokens_display() == "28K(20K cached)"
        assert "28K(20K cached)" in state.format_plain()

    def test_tokens_display_without_cache(self):
        state = StatusBarState(cumulative_tokens=28_672, cached_tokens=0)
        assert state.tokens_display() == "28K"
        assert "cached" not in state.format_plain()

    def test_format_plain_includes_plan_marker_when_active(self):
        state = StatusBarState(plan_mode=True, agent="chat")
        text = state.format_plain()
        # Brand → PLAN → agent order preserved
        assert text.index("Datus") < text.index("PLAN") < text.index("chat")

    def test_to_formatted_tokens_uses_styled_segments(self):
        state = StatusBarState(agent="gen_sql", model="m")
        tokens = state.to_formatted_tokens()
        styles = [style for style, _ in tokens]
        assert "class:status-bar.brand" in styles
        assert "class:status-bar.agent" in styles
        assert "class:status-bar.model" in styles
        assert "class:status-bar.tokens" in styles
        assert "class:status-bar.ctx" in styles
        assert "class:status-bar.sep" in styles
        # labels were removed entirely — no label class should ever render
        assert "class:status-bar.label" not in styles
        assert "class:status-bar.plan" not in styles

    def test_to_formatted_tokens_includes_plan_marker(self):
        state = StatusBarState(plan_mode=True)
        tokens = state.to_formatted_tokens()
        styles = [style for style, _ in tokens]
        assert "class:status-bar.plan" in styles

    def test_format_plain_renders_connector_between_agent_and_model(self):
        state = StatusBarState(
            agent="chat",
            connector="starrocks: benchmark",
            model="claude-sonnet-4-6",
        )
        text = state.format_plain()
        assert "starrocks: benchmark" in text
        assert text.index("chat") < text.index("starrocks: benchmark") < text.index("claude-sonnet-4-6")

    def test_format_plain_omits_connector_when_empty(self):
        # Connector is conditional: when empty it must not produce an empty
        # segment nor an extra separator.
        without = StatusBarState(agent="chat", model="m").format_plain()
        with_conn = StatusBarState(agent="chat", model="m", connector="sqlite: demo").format_plain()
        # The connector variant has exactly one more " │ " than the empty one.
        assert with_conn.count(" │ ") == without.count(" │ ") + 1
        assert "sqlite: demo" not in without

    def test_to_formatted_tokens_includes_connector_class_when_set(self):
        state = StatusBarState(connector="mysql: prod")
        styles = [style for style, _ in state.to_formatted_tokens()]
        assert "class:status-bar.connector" in styles

    def test_to_formatted_tokens_omits_connector_class_when_empty(self):
        state = StatusBarState()
        styles = [style for style, _ in state.to_formatted_tokens()]
        assert "class:status-bar.connector" not in styles

    def test_to_formatted_tokens_appends_running_segment_when_agent_running(self):
        # The TUI path sets ``agent_running`` true while the worker thread is
        # dispatching; the status bar must surface a ``● running`` indicator
        # as the trailing segment so users can tell the REPL is busy at a
        # glance even though input remains editable.
        state = StatusBarState(agent="chat", agent_running=True)
        tokens = state.to_formatted_tokens()
        styles = [style for style, _ in tokens]
        texts = [text for _, text in tokens]
        assert "class:status-bar.running" in styles
        running_text = texts[styles.index("class:status-bar.running")]
        assert "running" in running_text
        # The running token must be the last meaningful segment (followed
        # only by the trailing pad so the bar has a breathing space at the
        # terminal edge).
        last_non_pad = None
        for style, text in reversed(tokens):
            if style == "class:status-bar":
                continue
            last_non_pad = style
            break
        assert last_non_pad == "class:status-bar.running"

    def test_to_formatted_tokens_omits_running_segment_when_idle(self):
        # The legacy PromptSession path never sets ``agent_running`` true,
        # so the running indicator must never appear outside the TUI.
        state = StatusBarState(agent="chat")
        styles = [style for style, _ in state.to_formatted_tokens()]
        assert "class:status-bar.running" not in styles


class TestStatusBarProviderAgent:
    def test_subagent_name_wins_over_default(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name="gen_sql", current_node=None),
            default_agent="other",
            agent_config=None,
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().agent == "gen_sql"

    def test_default_agent_used_when_no_subagent(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="planner",
            agent_config=None,
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().agent == "planner"

    def test_falls_back_to_chat_when_nothing_set(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=None,
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().agent == "chat"

    def test_missing_chat_commands_falls_back_to_default(self):
        cli = SimpleNamespace(default_agent="gen_sql", agent_config=None, plan_mode_active=False)
        provider = StatusBarProvider(cli)
        assert provider.current_state().agent == "gen_sql"


class TestStatusBarProviderModel:
    def test_reads_model_from_live_node_config(self):
        node = SimpleNamespace(
            model=SimpleNamespace(model_config=SimpleNamespace(model="claude-opus-4-7")),
            session_id=None,
            actions=[],
            context_length=0,
        )
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=node),
            default_agent="",
            agent_config=SimpleNamespace(active_model=lambda: SimpleNamespace(model="fallback-model")),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "claude-opus-4-7"

    def test_falls_back_to_agent_config_when_no_node(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=SimpleNamespace(active_model=lambda: SimpleNamespace(model="claude-sonnet-4-6")),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "claude-sonnet-4-6"

    def test_returns_dash_when_all_sources_fail(self):
        def raise_error():
            raise RuntimeError("boom")

        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=SimpleNamespace(active_model=raise_error),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "-"

    def test_provider_model_format_from_node(self):
        node = SimpleNamespace(
            model=SimpleNamespace(model_config=SimpleNamespace(model="gpt-5.4")),
            session_id=None,
            actions=[],
            context_length=0,
        )
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=node),
            default_agent="",
            agent_config=SimpleNamespace(
                active_model=lambda: SimpleNamespace(model="gpt-5.4"),
                _target_provider="openai",
            ),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "openai/gpt-5.4"

    def test_provider_model_format_from_config_fallback(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=SimpleNamespace(
                active_model=lambda: SimpleNamespace(model="kimi-k2.5"),
                _target_provider="kimi",
            ),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "kimi/kimi-k2.5"

    def test_no_provider_prefix_for_legacy_target(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=SimpleNamespace(
                active_model=lambda: SimpleNamespace(model="custom-model"),
                _target_provider=None,
            ),
            plan_mode_active=False,
        )
        provider = StatusBarProvider(cli)
        assert provider.current_state().model == "custom-model"


class TestStatusBarProviderTokens:
    def _make_cli(self, node, plan_mode=False):
        return SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=node),
            default_agent="",
            agent_config=None,
            plan_mode_active=plan_mode,
        )

    def test_cumulative_and_cached_tokens_read_from_session_manager(self):
        session_manager = MagicMock()
        session_manager.get_detailed_usage.return_value = {"total": {"total_tokens": 98_765, "cached_tokens": 20_480}}
        node = SimpleNamespace(
            model=SimpleNamespace(session_manager=session_manager, model_config=SimpleNamespace(model="m")),
            session_id="sess-1",
            actions=[],
            context_length=0,
        )
        state = StatusBarProvider(self._make_cli(node)).current_state()
        assert state.cumulative_tokens == 98_765
        assert state.cached_tokens == 20_480
        session_manager.get_detailed_usage.assert_called_once_with("sess-1")

    def test_cached_tokens_zero_when_missing_from_totals(self):
        session_manager = MagicMock()
        session_manager.get_detailed_usage.return_value = {"total": {"total_tokens": 4096}}
        node = SimpleNamespace(
            model=SimpleNamespace(session_manager=session_manager, model_config=SimpleNamespace(model="m")),
            session_id="sess-nocache",
            actions=[],
            context_length=0,
        )
        state = StatusBarProvider(self._make_cli(node)).current_state()
        assert state.cumulative_tokens == 4096
        assert state.cached_tokens == 0

    def test_cumulative_tokens_zero_without_session_id(self):
        node = SimpleNamespace(model=SimpleNamespace(), session_id=None, actions=[], context_length=0)
        state = StatusBarProvider(self._make_cli(node)).current_state()
        assert state.cumulative_tokens == 0
        assert state.cached_tokens == 0

    def test_cumulative_tokens_zero_on_session_manager_error(self):
        session_manager = MagicMock()
        session_manager.get_detailed_usage.side_effect = RuntimeError("db missing")
        node = SimpleNamespace(
            model=SimpleNamespace(session_manager=session_manager, model_config=SimpleNamespace(model="m")),
            session_id="sess-err",
            actions=[],
            context_length=0,
        )
        state = StatusBarProvider(self._make_cli(node)).current_state()
        assert state.cumulative_tokens == 0
        assert state.cached_tokens == 0

    def test_context_used_picks_last_call_input_from_latest_assistant_action(self):
        actions = [
            SimpleNamespace(output={"usage": {"last_call_input_tokens": 1024}}),
            SimpleNamespace(output={"usage": {"last_call_input_tokens": 8500}}),
        ]
        node = SimpleNamespace(model=None, session_id=None, actions=actions, context_length=128_000)
        provider = StatusBarProvider(self._make_cli(node))
        state = provider.current_state()
        assert state.context_used == 8500
        assert state.context_total == 128_000

    def test_context_used_falls_back_to_input_tokens(self):
        actions = [SimpleNamespace(output={"usage": {"input_tokens": 4200}})]
        node = SimpleNamespace(model=None, session_id=None, actions=actions, context_length=0)
        provider = StatusBarProvider(self._make_cli(node))
        assert provider.current_state().context_used == 4200

    def test_context_used_zero_when_no_usage_data(self):
        node = SimpleNamespace(model=None, session_id=None, actions=[], context_length=None)
        provider = StatusBarProvider(self._make_cli(node))
        state = provider.current_state()
        assert state.context_used == 0
        assert state.context_total == 0


class TestStatusBarProviderPlanMode:
    def test_plan_mode_flag_propagates(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=None,
            plan_mode_active=True,
        )
        assert StatusBarProvider(cli).current_state().plan_mode is True

    def test_plan_mode_default_false(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=None,
        )
        assert StatusBarProvider(cli).current_state().plan_mode is False


class TestStatusBarProviderNoNode:
    def test_empty_session_state(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=SimpleNamespace(active_model=lambda: SimpleNamespace(model="claude-sonnet-4-6")),
            plan_mode_active=False,
        )
        state = StatusBarProvider(cli).current_state()
        assert state.agent == "chat"
        assert state.model == "claude-sonnet-4-6"
        assert state.cumulative_tokens == 0
        assert state.cached_tokens == 0
        assert state.context_used == 0
        assert state.context_total == 0
        assert state.plan_mode is False
        assert state.connector == ""


class TestStatusBarProviderConnector:
    @staticmethod
    def _make_cli(cli_context=None, db_connector=None):
        return SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=None,
            plan_mode_active=False,
            cli_context=cli_context,
            db_connector=db_connector,
        )

    def test_returns_type_and_name_when_both_present(self):
        ctx = SimpleNamespace(current_logic_db_name="benchmark", current_db_name="benchmark_raw")
        connector = SimpleNamespace(dialect="starrocks")
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "starrocks: benchmark"

    def test_logic_db_name_wins_over_current_db_name(self):
        ctx = SimpleNamespace(current_logic_db_name="prod", current_db_name="raw")
        connector = SimpleNamespace(dialect="mysql")
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "mysql: prod"

    def test_falls_back_to_current_db_name_when_logic_missing(self):
        ctx = SimpleNamespace(current_logic_db_name=None, current_db_name="raw")
        connector = SimpleNamespace(dialect="duckdb")
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "duckdb: raw"

    def test_dialect_is_lowercased(self):
        ctx = SimpleNamespace(current_logic_db_name="benchmark", current_db_name=None)
        connector = SimpleNamespace(dialect="StarRocks")
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "starrocks: benchmark"

    def test_dialect_enum_uses_value_not_repr(self):
        # DBType is a (str, Enum) mixin; on Python 3.11+ str(DBType.SQLITE)
        # returns "DBType.SQLITE". The resolver must pull .value so users see
        # "sqlite: ..." rather than "dbtype.sqlite: ...".
        ctx = SimpleNamespace(current_logic_db_name="california_schools", current_db_name=None)
        connector = SimpleNamespace(dialect=DBType.SQLITE)
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "sqlite: california_schools"

    def test_returns_bare_name_when_db_connector_missing(self):
        ctx = SimpleNamespace(current_logic_db_name="only_name", current_db_name=None)
        state = StatusBarProvider(self._make_cli(ctx, None)).current_state()
        assert state.connector == "only_name"

    def test_returns_bare_name_when_dialect_missing(self):
        ctx = SimpleNamespace(current_logic_db_name="demo", current_db_name=None)
        connector = SimpleNamespace(dialect=None)
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == "demo"

    def test_returns_empty_when_no_db_name(self):
        ctx = SimpleNamespace(current_logic_db_name=None, current_db_name=None)
        connector = SimpleNamespace(dialect="mysql")
        state = StatusBarProvider(self._make_cli(ctx, connector)).current_state()
        assert state.connector == ""

    def test_returns_empty_when_cli_context_absent(self):
        cli = SimpleNamespace(
            chat_commands=SimpleNamespace(current_subagent_name=None, current_node=None),
            default_agent="",
            agent_config=None,
            plan_mode_active=False,
        )
        state = StatusBarProvider(cli).current_state()
        assert state.connector == ""

    def test_resolver_swallows_attribute_exceptions(self):
        class ExplodingCtx:
            @property
            def current_logic_db_name(self):
                raise RuntimeError("ctx boom")

            @property
            def current_db_name(self):
                raise RuntimeError("ctx boom")

        class ExplodingConnector:
            @property
            def dialect(self):
                raise RuntimeError("connector boom")

        cli = self._make_cli(ExplodingCtx(), ExplodingConnector())
        # Must not propagate; both sources fail → empty connector.
        state = StatusBarProvider(cli).current_state()
        assert state.connector == ""


@pytest.mark.parametrize(
    "used,total,expected_fragment",
    [
        (0, 1_048_576, "0K/1024K 0%"),
        (512, 1024, "0.5K/1.0K 50%"),
        (256_000, 1_024_000, "250K/1000K 25%"),
    ],
)
def test_format_plain_context_percentage(used, total, expected_fragment):
    state = StatusBarState(context_used=used, context_total=total)
    assert expected_fragment in state.format_plain()
