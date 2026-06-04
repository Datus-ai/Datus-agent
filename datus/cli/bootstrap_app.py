# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/bootstrap`` multi-tab picker.

Mirrors :class:`datus.cli.model_app.ModelApp` styling: a horizontal tab
strip across the top, a per-tab form below, and a footer hint.
``Ctrl+R`` runs the **currently active tab only** — one init category
per invocation, matching ``datus-agent bootstrap-kb --components <one>``
semantics.

Each tab now exposes only the fields the corresponding ``stream_*``
helper consumes; ``build_mode`` is replaced by a single ``[ ] overwrite``
:class:`Checkbox` (Space toggles when focused — checked → ``overwrite``,
unchecked → ``incremental``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from datus.cli.tui.wizard_host import EmbeddedWizard

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Checkbox, TextArea
from rich.console import Console

from datus.cli.cli_styles import render_tui_title_bar
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class _Tab(Enum):
    SCHEMA = "metadata"
    SQL = "reference_sql"
    TEMPLATE = "reference_template"
    SEMANTIC = "semantic_model"
    METRICS = "metrics"


_TAB_ORDER: Tuple[_Tab, ...] = (
    _Tab.SCHEMA,
    _Tab.SQL,
    _Tab.TEMPLATE,
    _Tab.SEMANTIC,
    _Tab.METRICS,
)


_TAB_LABELS: Dict[_Tab, str] = {
    _Tab.SCHEMA: " Schema ",
    _Tab.SQL: " SQL ",
    _Tab.TEMPLATE: " Template ",
    _Tab.SEMANTIC: " Semantic ",
    _Tab.METRICS: " Metrics ",
}


# Public names mirror old multi-panel API where possible so callers /
# tests that only need ``TaskSpec.name`` keep working.
PANEL_NAMES: Tuple[str, ...] = tuple(t.value for t in _TAB_ORDER)


@dataclass
class TaskSpec:
    """One bootstrap init category and its user-supplied options."""

    name: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapPlan:
    """Outcome of :class:`BootstrapApp.run` — exactly one task per invocation."""

    task: TaskSpec


def _make_field(prompt: str, default: str = "") -> TextArea:
    return TextArea(
        height=1,
        multiline=False,
        prompt=prompt,
        text=default,
        focus_on_click=True,
    )


def _make_overwrite() -> Checkbox:
    """One checkbox row per tab — Space toggles when focused."""
    return Checkbox(text=" overwrite  (Space to toggle — checked = overwrite, otherwise incremental)")


def _build_mode_from_checkbox(cb: Checkbox) -> str:
    return "overwrite" if cb.checked else "incremental"


# Focus chain entries: TextArea exposes ``.window``; Checkbox exposes
# ``.window`` too (it wraps a single ``Window``). Both are matched
# uniformly via ``current is entry.window`` — see :meth:`_advance_focus`.
_FocusEntry = Union[TextArea, Checkbox]


class BootstrapApp:
    """Six-tab :mod:`prompt_toolkit` Application for ``/bootstrap``."""

    def __init__(
        self,
        console: Console,
        *,
        datasource_default: str = "",
    ) -> None:
        self._console = console
        self._tab: _Tab = _Tab.SCHEMA
        self._error_message: Optional[str] = None
        self._result: Optional[BootstrapPlan] = None

        ds = datasource_default

        # ── SCHEMA ─────────────────────────────────────────────────────
        self._schema_datasource = _make_field("datasource:        ", ds)
        self._schema_overwrite = _make_overwrite()

        # ── SQL ────────────────────────────────────────────────────────
        self._sql_datasource = _make_field("datasource:        ", ds)
        self._sql_dir = _make_field("sql_dir:           ")
        self._sql_pool = _make_field("pool_size:         ", "3")
        self._sql_subject_tree = _make_field("subject_tree:      ")
        self._sql_overwrite = _make_overwrite()

        # ── TEMPLATE ───────────────────────────────────────────────────
        self._tpl_datasource = _make_field("datasource:        ", ds)
        self._tpl_dir = _make_field("template_dir:      ")
        self._tpl_pool = _make_field("pool_size:         ", "3")
        self._tpl_subject_tree = _make_field("subject_tree:      ")
        self._tpl_overwrite = _make_overwrite()

        # ── SEMANTIC ───────────────────────────────────────────────────
        self._sem_datasource = _make_field("datasource:        ", ds)
        self._sem_success_story = _make_field("success_story:     ")
        self._sem_overwrite = _make_overwrite()

        # ── METRICS ────────────────────────────────────────────────────
        self._met_datasource = _make_field("datasource:        ", ds)
        self._met_success_story = _make_field("success_story:     ")
        self._met_pool = _make_field("pool_size:         ", "3")
        self._met_subject_tree = _make_field("subject_tree:      ")
        self._met_overwrite = _make_overwrite()

        # Dual-mode finish hook — see EffortApp.
        self._on_done: Optional[Callable[[Optional[BootstrapPlan]], None]] = None
        self._app: Optional[Application] = None
        self._initial_focus: Optional[Any] = None

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> Optional[BootstrapPlan]:
        """Run as a transient standalone Application (non-TUI fallback)."""
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        self._app = Application(
            layout=Layout(root, focused_element=self._initial_focus),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._on_done = lambda result: self._app.exit(result=result)
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("BootstrapApp crashed: %s", exc, exc_info=True)
            return None
        finally:
            self._on_done = None
            self._app = None

    def build_embedded_panel(self, done_future: "asyncio.Future") -> "EmbeddedWizard":
        from datus.cli.tui.wizard_host import EmbeddedWizard, resolve_cancel, resolve_with

        def _done(result):
            if result is None:
                resolve_cancel(done_future)
            else:
                resolve_with(done_future, result)

        self._on_done = _done
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        return EmbeddedWizard(
            container=root,
            key_bindings=kb,
            first_focus=self._initial_focus,
            done_future=done_future,
        )

    def _finish(self, result: Optional["BootstrapPlan"]) -> None:
        if self._on_done is None:
            return
        self._on_done(result)

    def _layout(self) -> Optional[Layout]:
        app = getattr(self, "_app", None)
        if app is not None:
            return app.layout
        try:
            from prompt_toolkit.application import get_app

            return get_app().layout
        except Exception:
            return None

    def _focus(self, target) -> None:
        layout = self._layout()
        if layout is None or target is None:
            return
        try:
            layout.focus(target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("BootstrapApp focus(%r) failed: %s", target, exc)

    # ── Layout ──────────────────────────────────────────────────────────

    def _build_root_container(self, kb: KeyBindings) -> HSplit:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Datus Bootstrap")),
            height=1,
        )
        tab_window = Window(
            content=FormattedTextControl(self._render_tab_strip, focusable=False),
            height=1,
            style="class:bootstrap.tabs",
        )

        section_header = Window(
            content=FormattedTextControl(self._render_section_header, focusable=False),
            height=Dimension(min=1, max=2),
        )

        def _wrap_form(*fields: AnyContainer) -> HSplit:
            """Pad a per-tab form with blank rows so it breathes visually."""
            return HSplit(
                [
                    Window(height=1),  # blank line above the form fields
                    *fields,
                    Window(height=1),  # blank line below the form fields
                ]
            )

        body_map: Dict[_Tab, AnyContainer] = {
            _Tab.SCHEMA: _wrap_form(
                self._schema_datasource,
                self._schema_overwrite,
            ),
            _Tab.SQL: _wrap_form(
                self._sql_datasource,
                self._sql_dir,
                self._sql_pool,
                self._sql_subject_tree,
                self._sql_overwrite,
            ),
            _Tab.TEMPLATE: _wrap_form(
                self._tpl_datasource,
                self._tpl_dir,
                self._tpl_pool,
                self._tpl_subject_tree,
                self._tpl_overwrite,
            ),
            _Tab.SEMANTIC: _wrap_form(
                self._sem_datasource,
                self._sem_success_story,
                self._sem_overwrite,
            ),
            _Tab.METRICS: _wrap_form(
                self._met_datasource,
                self._met_success_story,
                self._met_pool,
                self._met_subject_tree,
                self._met_overwrite,
            ),
        }

        body = DynamicContainer(lambda: body_map[self._tab])

        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:bootstrap.error", f"  {self._error_message or ''}")]),
                height=Dimension(min=0, max=1),
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=Dimension(min=0, max=1),
            style="class:bootstrap.hint",
        )

        sep = lambda: Window(  # noqa: E731
            height=1,
            char="\u2500",
            style="class:bootstrap.separator",
        )

        self._initial_focus = self._schema_datasource
        return HSplit(
            [
                title_bar,
                tab_window,
                sep(),
                section_header,
                sep(),
                body,
                error_window,
                sep(),
                hint_window,
            ]
        )

    _SECTION_DESCRIPTIONS: Dict[_Tab, str] = {
        _Tab.SCHEMA: "Crawl the live database schema into the metadata RAG.",
        _Tab.SQL: "Index every SQL file under the directory as reference SQL.",
        _Tab.TEMPLATE: "Index every Jinja2 template under the directory.",
        _Tab.SEMANTIC: "Generate semantic models from a success-story CSV.",
        _Tab.METRICS: "Extract core metrics from a success-story CSV.",
    }

    def _render_section_header(self) -> List[Tuple[str, str]]:
        label = _TAB_LABELS[self._tab].strip()
        desc = self._SECTION_DESCRIPTIONS[self._tab]
        return [
            ("bold", f"  {label}\n"),
            ("class:bootstrap.dim", f"  {desc}"),
        ]

    def _render_tab_strip(self) -> List[Tuple[str, str]]:
        parts: List[Tuple[str, str]] = [("", "  ")]
        for tab in _TAB_ORDER:
            style = "reverse bold" if tab == self._tab else ""
            parts.append((style, _TAB_LABELS[tab]))
            parts.append(("", " "))
        parts.append(("class:bootstrap.tabs-hint", "  (Tab or \u2190/\u2192 to switch)"))
        return parts

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [
            (
                "class:bootstrap.hint",
                ("  \u2191\u2193/Tab next field   \u2190/\u2192 switch tab   Ctrl+R run this tab   Esc cancel"),
            ),
        ]

    # ── Key bindings ────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("escape", eager=True)
        def _(event):
            self._finish(None)

        @kb.add("c-r")
        def _(event):
            self._submit()

        @kb.add("left")
        def _(event):
            self._cycle_tab(-1)

        @kb.add("right")
        def _(event):
            self._cycle_tab(+1)

        @kb.add("c-left")
        def _(event):
            self._cycle_tab(-1)

        @kb.add("c-right")
        def _(event):
            self._cycle_tab(+1)

        @kb.add("tab")
        def _(event):
            self._advance_focus(+1, event)

        @kb.add("s-tab")
        def _(event):
            self._advance_focus(-1, event)

        @kb.add("down")
        def _(event):
            self._advance_focus(+1, event)

        @kb.add("up")
        def _(event):
            self._advance_focus(-1, event)

        return kb

    def _cycle_tab(self, delta: int) -> None:
        idx = _TAB_ORDER.index(self._tab)
        self._tab = _TAB_ORDER[(idx + delta) % len(_TAB_ORDER)]
        self._error_message = None
        try:
            self._focus(self._focus_chain()[0])
        except Exception:
            pass

    def _advance_focus(self, delta: int, event) -> None:
        chain = self._focus_chain()
        try:
            current = event.app.layout.current_window
        except Exception:
            current = None
        cur_idx = -1
        for i, area in enumerate(chain):
            if current is area.window:
                cur_idx = i
                break
        next_idx = (cur_idx + delta) % len(chain) if cur_idx >= 0 else 0
        try:
            event.app.layout.focus(chain[next_idx])
        except Exception:
            pass

    # ── Per-tab focus chain (only the editable widgets) ────────────────

    def _focus_chain(self) -> List[_FocusEntry]:
        if self._tab == _Tab.SCHEMA:
            return [self._schema_datasource, self._schema_overwrite]
        if self._tab == _Tab.SQL:
            return [
                self._sql_datasource,
                self._sql_dir,
                self._sql_pool,
                self._sql_subject_tree,
                self._sql_overwrite,
            ]
        if self._tab == _Tab.TEMPLATE:
            return [
                self._tpl_datasource,
                self._tpl_dir,
                self._tpl_pool,
                self._tpl_subject_tree,
                self._tpl_overwrite,
            ]
        if self._tab == _Tab.SEMANTIC:
            return [
                self._sem_datasource,
                self._sem_success_story,
                self._sem_overwrite,
            ]
        if self._tab == _Tab.METRICS:
            return [
                self._met_datasource,
                self._met_success_story,
                self._met_pool,
                self._met_subject_tree,
                self._met_overwrite,
            ]
        return []

    # ── Submit ─────────────────────────────────────────────────────────

    def _submit(self) -> None:
        try:
            options = self._collect_for(self._tab)
        except _ValidationError as exc:
            self._error_message = str(exc)
            return
        self._result = BootstrapPlan(task=TaskSpec(name=self._tab.value, options=options))
        self._finish(self._result)

    def _collect_for(self, tab: _Tab) -> Dict[str, Any]:
        if tab == _Tab.SCHEMA:
            return {
                "datasource": _require_nonempty(self._schema_datasource.text, field="datasource"),
                "build_mode": _build_mode_from_checkbox(self._schema_overwrite),
            }
        if tab == _Tab.SQL:
            return {
                "datasource": _require_nonempty(self._sql_datasource.text, field="datasource"),
                "sql_dir": _require_nonempty(self._sql_dir.text, field="sql_dir"),
                "pool_size": _validate_int(self._sql_pool.text, field="pool_size"),
                "subject_tree": self._sql_subject_tree.text.strip(),
                "build_mode": _build_mode_from_checkbox(self._sql_overwrite),
            }
        if tab == _Tab.TEMPLATE:
            return {
                "datasource": _require_nonempty(self._tpl_datasource.text, field="datasource"),
                "template_dir": _require_nonempty(self._tpl_dir.text, field="template_dir"),
                "pool_size": _validate_int(self._tpl_pool.text, field="pool_size"),
                "subject_tree": self._tpl_subject_tree.text.strip(),
                "build_mode": _build_mode_from_checkbox(self._tpl_overwrite),
            }
        if tab == _Tab.SEMANTIC:
            return {
                "datasource": _require_nonempty(self._sem_datasource.text, field="datasource"),
                "success_story": _require_nonempty(self._sem_success_story.text, field="success_story"),
                "build_mode": _build_mode_from_checkbox(self._sem_overwrite),
            }
        if tab == _Tab.METRICS:
            return {
                "datasource": _require_nonempty(self._met_datasource.text, field="datasource"),
                "success_story": _require_nonempty(self._met_success_story.text, field="success_story"),
                "pool_size": _validate_int(self._met_pool.text, field="pool_size"),
                "subject_tree": self._met_subject_tree.text.strip(),
                "build_mode": _build_mode_from_checkbox(self._met_overwrite),
            }
        raise _ValidationError(f"Unsupported tab: {self._tab}")


# ── Validation helpers ────────────────────────────────────────────────


class _ValidationError(ValueError):
    pass


def _validate_int(raw: str, *, field: str) -> int:
    value = (raw or "").strip()
    try:
        out = int(value)
    except ValueError as exc:
        raise _ValidationError(f"{field} must be an integer") from exc
    if out < 1:
        raise _ValidationError(f"{field} must be >= 1")
    return out


def _require_nonempty(raw: str, *, field: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise _ValidationError(f"{field} is required")
    return value


__all__ = [
    "BootstrapApp",
    "BootstrapPlan",
    "TaskSpec",
    "PANEL_NAMES",
]
