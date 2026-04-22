# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Status bar provider for the Datus CLI prompt.

Renders a single-line status bar above the input prompt that carries the
Datus brand, current agent, model, cumulative session tokens, and context
usage. All token counters are expressed in KiB units where 1K = 1024 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

_TOKEN_UNIT = 1024  # 1K == 1024 tokens


def _humanize_tokens(n: int) -> str:
    """Format a token count using K (1024) as the sole unit.

    - 0 tokens renders as ``0K``
    - sub-kilo values keep one decimal (e.g. ``0.5K``) so they remain visible
    - values >= 10K render as rounded integers (e.g. ``54K``, ``1024K``)
    """
    if n is None:
        return "0K"
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "0K"
    if n <= 0:
        return "0K"
    k = n / _TOKEN_UNIT
    if k < 10:
        return f"{k:.1f}K"
    return f"{round(k)}K"


@dataclass
class StatusBarState:
    """Snapshot of status bar data rendered before each prompt."""

    agent: str = "chat"
    model: str = "-"
    connector: str = ""
    cumulative_tokens: int = 0
    cached_tokens: int = 0
    context_used: int = 0
    context_total: int = 0
    plan_mode: bool = False
    agent_running: bool = False

    def context_display(self) -> str:
        if self.context_total > 0:
            pct = self.context_used / self.context_total * 100
            return f"{_humanize_tokens(self.context_used)}/{_humanize_tokens(self.context_total)} {pct:.0f}%"
        return "0K/0K 0%"

    def tokens_display(self) -> str:
        base = _humanize_tokens(self.cumulative_tokens)
        if self.cached_tokens > 0:
            return f"{base}({_humanize_tokens(self.cached_tokens)} cached)"
        return base

    def format_plain(self) -> str:
        """Render the status bar as a plain string (used for tests and logs)."""
        segments = ["Datus"]
        if self.plan_mode:
            segments.append("PLAN")
        segments.append(self.agent)
        if self.connector:
            segments.append(self.connector)
        segments.extend(
            [
                self.model,
                self.tokens_display(),
                self.context_display(),
            ]
        )
        return " " + " │ ".join(segments) + " "

    def to_formatted_tokens(self) -> List[Tuple[str, str]]:
        """Return prompt_toolkit formatted text tokens with styled segments."""
        sep: Tuple[str, str] = ("class:status-bar.sep", " │ ")
        pad: Tuple[str, str] = ("class:status-bar", " ")

        tokens: List[Tuple[str, str]] = [pad, ("class:status-bar.brand", "Datus")]
        if self.plan_mode:
            tokens.extend([sep, ("class:status-bar.plan", "PLAN")])
        tokens.extend([sep, ("class:status-bar.agent", self.agent)])
        if self.connector:
            tokens.extend([sep, ("class:status-bar.connector", self.connector)])
        tokens.extend(
            [
                sep,
                ("class:status-bar.model", self.model),
                sep,
                ("class:status-bar.tokens", self.tokens_display()),
                sep,
                ("class:status-bar.ctx", self.context_display()),
            ]
        )
        if self.agent_running:
            # Trailing indicator signals that the REPL is busy. It is only
            # surfaced by the TUI path, which sets ``agent_running`` via
            # ``StatusBarProvider.current_state``; the legacy PromptSession
            # never exposes a truthy value so its rendering is unchanged.
            tokens.extend([sep, ("class:status-bar.running", "● running")])
        tokens.append(pad)
        return tokens


class StatusBarProvider:
    """Collects status bar data from the live CLI state with defensive fallbacks."""

    def __init__(self, cli: "DatusCLI"):
        self._cli = cli

    def current_state(self) -> StatusBarState:
        cumulative, cached = self._resolve_session_totals()
        tui_app = getattr(self._cli, "tui_app", None)
        agent_running = False
        if tui_app is not None:
            try:
                agent_running = tui_app.agent_running.is_set()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"status_bar: failed to read agent_running: {e}")
        return StatusBarState(
            agent=self._resolve_agent(),
            model=self._resolve_model(),
            connector=self._resolve_connector(),
            cumulative_tokens=cumulative,
            cached_tokens=cached,
            context_used=self._resolve_context_used(),
            context_total=self._resolve_context_total(),
            plan_mode=bool(getattr(self._cli, "plan_mode_active", False)),
            agent_running=agent_running,
        )

    def _current_node(self):
        chat_commands = getattr(self._cli, "chat_commands", None)
        return getattr(chat_commands, "current_node", None) if chat_commands else None

    def _resolve_agent(self) -> str:
        chat_commands = getattr(self._cli, "chat_commands", None)
        subagent = getattr(chat_commands, "current_subagent_name", None) if chat_commands else None
        if subagent:
            return subagent
        default = getattr(self._cli, "default_agent", "") or ""
        return default or "chat"

    def _resolve_model(self) -> str:
        node = self._current_node()
        try:
            if node is not None and getattr(node, "model", None) is not None:
                model_cfg = getattr(node.model, "model_config", None)
                name = getattr(model_cfg, "model", None)
                if name:
                    return self._format_model_label(str(name))
        except Exception as e:
            logger.debug(f"status_bar: failed to read model from node: {e}")
        try:
            agent_config = getattr(self._cli, "agent_config", None)
            if agent_config is not None:
                return self._format_model_label(str(agent_config.active_model().model))
        except Exception as e:
            logger.debug(f"status_bar: failed to read active model from config: {e}")
        return "-"

    def _format_model_label(self, model_name: str) -> str:
        agent_config = getattr(self._cli, "agent_config", None)
        if agent_config is None:
            return model_name
        provider = getattr(agent_config, "_target_provider", None)
        if provider:
            return f"{provider}/{model_name}"
        return model_name

    def _resolve_connector(self) -> str:
        """Return the current connector as ``"<db_type>: <db_name>"``.

        Falls back to the bare db name when dialect is unavailable, or an
        empty string when no database is connected.
        """
        db_name = ""
        try:
            ctx = getattr(self._cli, "cli_context", None)
            if ctx is not None:
                db_name = getattr(ctx, "current_logic_db_name", None) or getattr(ctx, "current_db_name", None) or ""
        except Exception as e:
            logger.debug(f"status_bar: failed to read db name: {e}")

        db_type = ""
        try:
            connector = getattr(self._cli, "db_connector", None)
            if connector is not None:
                dialect = getattr(connector, "dialect", None)
                if dialect:
                    # For Enum values (e.g. DBType.SQLITE), str(...) yields
                    # "DBType.SQLITE" on Python 3.11+; prefer .value so we get
                    # the canonical lowercase name ("sqlite").
                    raw = getattr(dialect, "value", dialect)
                    db_type = str(raw).lower()
        except Exception as e:
            logger.debug(f"status_bar: failed to read db dialect: {e}")

        if not db_name:
            return ""
        if db_type:
            return f"{db_type}: {db_name}"
        return str(db_name)

    def _resolve_session_totals(self) -> Tuple[int, int]:
        """Return ``(cumulative_total_tokens, cumulative_cached_tokens)``."""
        node = self._current_node()
        if node is None:
            return 0, 0
        session_id = getattr(node, "session_id", None)
        if not session_id:
            return 0, 0
        model = getattr(node, "model", None)
        if model is None:
            return 0, 0
        try:
            session_manager = model.session_manager
        except Exception as e:
            logger.debug(f"status_bar: session_manager unavailable: {e}")
            return 0, 0
        try:
            usage = session_manager.get_detailed_usage(session_id)
            total = usage.get("total", {}) if isinstance(usage, dict) else {}
            return int(total.get("total_tokens", 0) or 0), int(total.get("cached_tokens", 0) or 0)
        except Exception as e:
            logger.debug(f"status_bar: get_detailed_usage failed: {e}")
            return 0, 0

    def _resolve_context_used(self) -> int:
        node = self._current_node()
        if node is None:
            return 0
        for action in reversed(getattr(node, "actions", []) or []):
            output = getattr(action, "output", None)
            if not isinstance(output, dict):
                continue
            usage = output.get("usage")
            if not isinstance(usage, dict):
                continue
            used = usage.get("last_call_input_tokens") or usage.get("input_tokens") or 0
            try:
                return int(used)
            except (TypeError, ValueError):
                return 0
        return 0

    def _resolve_context_total(self) -> int:
        node = self._current_node()
        if node is not None:
            length = getattr(node, "context_length", None)
            if length:
                try:
                    return int(length)
                except (TypeError, ValueError):
                    pass
        return 0
