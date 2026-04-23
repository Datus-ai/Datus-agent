# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/model`` picker rendered as a single prompt_toolkit
:class:`Application`.

The legacy implementation chained multiple :class:`select_choice` calls,
each spawning its own nested Application. In TUI mode that produced
``stdin`` contention with the persistent :class:`DatusApp` and manifested
as the "second-level menu doesn't show up", "freeze", and "lag" bugs the
user reported. Here the whole interaction — tab switching, provider
drill-down, credential capture, custom-model add — lives inside **one**
Application, so the outer TUI only needs to release ``stdin`` once via
:meth:`DatusApp.suspend_input`.

External async flows (browser OAuth, subscription token auto-detection)
remain synchronous and run **after** :meth:`ModelApp.run` returns, so the
Application itself never blocks on I/O.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.configuration.agent_config import AgentConfig, ProviderConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_CODING_PLAN_PROVIDERS = frozenset({"alibaba_coding", "glm_coding", "minimax_coding", "kimi_coding"})
# Providers that live on a separate "Plans" tab: everything that isn't a
# straightforward api_key endpoint — Chinese coding plans (all use
# Anthropic-compatible gateways), the Claude Code subscription path, and the
# Codex OAuth path. Keeping them out of the main Providers tab reduces noise
# and groups the "bring-your-own-plan" options together.
_PLAN_PROVIDERS = frozenset({"claude_subscription", "codex"}) | _CODING_PLAN_PROVIDERS

_MASKED_KEY_PLACEHOLDER = "••••••••"

# Display-name overrides shown in the UI. The internal key (used as the
# ``agent.providers`` map key and in ``providers.yml``) stays unchanged so
# no config migration is required.
_DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "claude_subscription": "claude code",
}


def _display_name(provider: str) -> str:
    """Return the human-readable label for a provider key.

    Explicit overrides win; otherwise underscores are replaced with spaces
    so keys like ``alibaba_coding`` surface as ``alibaba coding``. The
    underlying key (used in ``agent.providers`` and ``providers.yml``) is
    never rewritten.
    """
    if provider in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[provider]
    return provider.replace("_", " ")


class _Tab(Enum):
    PROVIDERS = "providers"
    PLANS = "plans"
    CUSTOM = "custom"


_TAB_CYCLE: Tuple[_Tab, ...] = (_Tab.PROVIDERS, _Tab.PLANS, _Tab.CUSTOM)


class _View(Enum):
    PROVIDER_LIST = "provider_list"
    PROVIDER_MODELS = "provider_models"
    PROVIDER_CRED_FORM = "provider_cred_form"
    PROVIDER_TOKEN_FORM = "provider_token_form"
    CUSTOM_LIST = "custom_list"
    ADD_MODEL_FORM = "add_model_form"


@dataclass
class ModelSelection:
    """Outcome of a :class:`ModelApp` run.

    ``kind`` discriminates the payload:

    - ``"provider_model"`` — ``provider`` + ``model`` filled.
    - ``"custom"`` — ``name`` filled (key in ``agent.models``).
    - ``"needs_oauth"`` — caller should drive the OAuth handshake for
      ``provider`` and (optionally) ``model``, then persist + re-run.
    - ``"add_custom"`` — caller should persist the custom model dict in
      ``payload`` under ``name`` and activate it.
    - ``"delete_custom"`` — caller should drop ``name`` from
      ``agent.models`` (both memory + YAML).
    """

    kind: str
    provider: Optional[str] = None
    model: Optional[str] = None
    name: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class ModelApp:
    """Two-tab provider/custom picker.

    The caller is expected to:

    1. Wrap ``app.run()`` in ``tui_app.suspend_input()`` when the REPL is
       in TUI mode (no-op otherwise).
    2. Apply the returned :class:`ModelSelection` via
       :meth:`AgentConfig.set_active_provider_model` /
       :meth:`AgentConfig.set_active_custom` or by re-entering this app
       after resolving credentials / running OAuth.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        console: Console,
        *,
        seed_provider: Optional[str] = None,
        seed_tab: Optional[str] = None,
    ) -> None:
        self._cfg = agent_config
        self._console = console
        self._seed_provider = seed_provider
        self._seed_tab = seed_tab

        self._tab: _Tab = _Tab.PROVIDERS
        self._view: _View = _View.PROVIDER_LIST

        self._providers: List[str] = []
        self._provider_meta: Dict[str, Dict[str, Any]] = {}
        self._availability: Dict[str, bool] = {}
        self._list_cursor: int = 0
        self._list_offset: int = 0

        self._active_provider: Optional[str] = None
        self._provider_models: List[str] = []
        self._custom_names: List[str] = []

        self._current_provider, self._current_model = self._read_current_selection()
        self._current_custom = self._cfg.target if self._cfg.target in (self._cfg.models or {}) else None

        self._load_providers()
        self._refresh_custom_names()

        self._result: Optional[ModelSelection] = None
        self._error_message: Optional[str] = None

        # Form buffers. Built up-front so key bindings can reference them
        # directly; a ConditionalContainer decides which ones are visible.
        self._cred_api_key = TextArea(height=1, multiline=False, prompt="API Key: ", password=True, focus_on_click=True)
        self._cred_base_url = TextArea(height=1, multiline=False, prompt="Base URL: ", focus_on_click=True)
        self._token_input = TextArea(height=1, multiline=False, prompt="Token:   ", password=True, focus_on_click=True)
        self._add_type = TextArea(height=1, multiline=False, prompt="type:      ", focus_on_click=True)
        self._add_model = TextArea(height=1, multiline=False, prompt="model:     ", focus_on_click=True)
        self._add_base_url = TextArea(height=1, multiline=False, prompt="base_url:  ", focus_on_click=True)
        self._add_api_key = TextArea(
            height=1, multiline=False, prompt="api_key:   ", password=True, focus_on_click=True
        )
        self._add_name = TextArea(height=1, multiline=False, prompt="name:      ", focus_on_click=True)

        self._form_focus_order: List[TextArea] = []
        self._form_focus_idx: int = 0
        # Set to the name of the custom entry that the user has pressed
        # ``d`` on once. A second press confirms deletion; any other key
        # clears the pending state.
        self._pending_delete_custom: Optional[str] = None

        # title bar(1) + tab strip(1) + 2 separators(2) + error bar(1) + footer(1) + scroll indicator(1) = 7 lines chrome
        term_height = shutil.get_terminal_size((120, 40)).lines
        self._max_visible: int = max(3, min(15, term_height - 7))

        self._app = self._build_application()

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(self) -> Optional[ModelSelection]:
        """Run the Application. Returns ``None`` on cancel."""
        early = self._apply_seed()
        if early is not None:
            return early
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("ModelApp crashed: %s", exc)
            print_error(self._console, f"/model error: {exc}")
            return None

    def _apply_seed(self) -> Optional[ModelSelection]:
        """Pre-position the state machine based on seed kwargs.

        Returns a result when the seed implies an immediate caller
        hand-off (currently: unavailable oauth provider). Otherwise,
        mutates internal state and returns ``None`` so :meth:`run` can
        start the Application in the target view.
        """
        if self._seed_tab == "custom":
            self._enter_custom_list()
            return None
        if self._seed_tab == "plans":
            self._enter_provider_list(_Tab.PLANS)
            return None
        provider = self._seed_provider
        if not provider or provider not in self._provider_meta:
            return None
        # Remember which tab the seeded provider lives in, so Esc from the
        # cred/token form returns to the correct tab.
        self._tab = _Tab.PLANS if provider in _PLAN_PROVIDERS else _Tab.PROVIDERS
        meta = self._provider_meta.get(provider) or {}
        auth_type = str(meta.get("auth_type", "api_key"))
        if self._availability.get(provider):
            self._enter_provider_models(provider)
            return None
        if auth_type == "api_key":
            self._enter_cred_form(provider)
            return None
        if auth_type == "subscription":
            token = self._try_auto_detect_subscription_token(provider)
            if token:
                self._cfg.set_provider_config(
                    provider=provider,
                    api_key=token,
                    base_url=str(meta.get("base_url") or "") or None,
                    auth_type="subscription",
                )
                self._availability[provider] = True
                self._enter_provider_models(provider)
                return None
            self._enter_token_form(provider)
            return None
        if auth_type == "oauth":
            return ModelSelection(kind="needs_oauth", provider=provider)
        return None

    # ─────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────

    def _read_current_selection(self) -> Tuple[Optional[str], Optional[str]]:
        return getattr(self._cfg, "_target_provider", None), getattr(self._cfg, "_target_model", None)

    def _load_providers(self) -> None:
        catalog = self._cfg.provider_catalog
        providers_meta = catalog.get("providers", {}) if isinstance(catalog, dict) else {}
        if not isinstance(providers_meta, dict) or not providers_meta:
            self._providers = []
            self._provider_meta = {}
            return
        # Cache availability once per app instance; provider_available()
        # hits disk for subscription/oauth providers so we avoid calling
        # it every render.
        self._availability = {name: self._safe_available(name) for name in providers_meta.keys()}
        self._providers = sorted(
            providers_meta.keys(),
            key=lambda name: (not self._availability.get(name, False), name),
        )
        self._provider_meta = {name: providers_meta[name] for name in self._providers}

    def _safe_available(self, provider: str) -> bool:
        try:
            return bool(self._cfg.provider_available(provider))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("provider_available(%s) raised: %s", provider, exc)
            return False

    def _refresh_custom_names(self) -> None:
        models_map = self._cfg.models or {}
        self._custom_names = sorted(models_map.keys())

    # ─────────────────────────────────────────────────────────────────
    # Layout construction
    # ─────────────────────────────────────────────────────────────────

    def _build_application(self) -> Application:
        tab_window = Window(
            content=FormattedTextControl(self._render_tab_strip, focusable=False),
            height=1,
            style="class:model-app.tabs",
        )

        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            style="class:model-app.list",
            height=Dimension(min=3),
        )

        # Per-form hint rows are intentionally omitted; the global footer
        # already surfaces the same Tab / Enter / Esc bindings.
        cred_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_cred_header, focusable=False),
                    height=Dimension(min=1, max=3),
                ),
                self._cred_api_key,
                self._cred_base_url,
            ]
        )
        token_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_token_header, focusable=False),
                    height=Dimension(min=1, max=3),
                ),
                self._token_input,
            ]
        )
        add_form = HSplit(
            [
                Window(
                    FormattedTextControl(self._render_add_header, focusable=False),
                    height=Dimension(min=1, max=2),
                ),
                self._add_name,
                self._add_type,
                self._add_model,
                self._add_base_url,
                self._add_api_key,
            ]
        )

        def _body_container():
            if self._view == _View.PROVIDER_CRED_FORM:
                return cred_form
            if self._view == _View.PROVIDER_TOKEN_FORM:
                return token_form
            if self._view == _View.ADD_MODEL_FORM:
                return add_form
            return list_window

        body = DynamicContainer(_body_container)

        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
            style="class:model-app.hint",
        )
        error_window = ConditionalContainer(
            content=Window(
                FormattedTextControl(lambda: [("class:model-app.error", f"  {self._error_message or ''}")]),
                height=1,
                style="class:model-app.error",
            ),
            filter=Condition(lambda: bool(self._error_message)),
        )

        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Model Selection")),
            height=1,
        )

        root = HSplit(
            [
                title_bar,
                tab_window,
                Window(height=1, char="\u2500", style="class:model-app.separator"),
                body,
                error_window,
                Window(height=1, char="\u2500", style="class:model-app.separator"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=None),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    # ─────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────

    def _render_tab_strip(self) -> List[Tuple[str, str]]:
        parts: List[Tuple[str, str]] = [("", "  ")]
        for tab, label in (
            (_Tab.PROVIDERS, " Providers "),
            (_Tab.PLANS, " Plans "),
            (_Tab.CUSTOM, " Custom "),
        ):
            style = "reverse bold" if tab == self._tab else ""
            parts.append((style, label))
            parts.append(("", " "))
        parts.append(("class:model-app.tabs-hint", "  (Tab or \u2190/\u2192 to switch)"))
        return parts

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        if self._view == _View.PROVIDER_LIST:
            hint = "  \u2191\u2193 navigate   Enter select   e edit credentials   Tab/\u2190\u2192 switch   Esc back   Ctrl+C cancel"
        elif self._view == _View.CUSTOM_LIST:
            hint = (
                "  \u2191\u2193 navigate   Enter select   d delete   Tab/\u2190\u2192 switch   Esc back   Ctrl+C cancel"
            )
        elif self._view == _View.PROVIDER_MODELS:
            hint = "  \u2191\u2193 navigate   Enter select   Tab/\u2190\u2192 switch   Esc back   Ctrl+C cancel"
        else:
            hint = "  Tab next field   Enter submit   Esc back   Ctrl+C cancel"
        return [("class:model-app.hint", hint)]

    def _render_list(self) -> List[Tuple[str, str]]:
        items = self._current_items()
        if not items:
            return [("class:model-app.dim", "  (nothing to show)\n")]
        self._clamp_cursor(len(items))
        visible = self._visible_slice(len(items))
        lines: List[Tuple[str, str]] = []
        start, end = visible
        if end - start < len(items):
            lines.append(("class:model-app.scroll", f"  ({start + 1}-{end} of {len(items)})\n"))
        for i in range(start, end):
            label, style = items[i]
            if i == self._list_cursor:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append((style, f"    {label}\n"))
        return lines

    def _render_cred_header(self) -> List[Tuple[str, str]]:
        provider = self._active_provider or ""
        default_base = str((self._provider_meta.get(provider) or {}).get("base_url", ""))
        env = (self._provider_meta.get(provider) or {}).get("api_key_env")
        hint = f"  Hint: leave API Key empty to use env var {env}" if env else ""
        return [
            ("bold", f"  Configure provider: {_display_name(provider)}\n"),
            ("class:model-app.dim", f"  Default base_url: {default_base}\n"),
            ("class:model-app.dim", hint),
        ]

    def _render_token_header(self) -> List[Tuple[str, str]]:
        provider = self._active_provider or ""
        return [
            ("bold", f"  Configure provider: {_display_name(provider)}\n"),
            ("class:model-app.dim", "  Paste your subscription token (e.g. sk-ant-oat01-...)\n"),
        ]

    def _render_add_header(self) -> List[Tuple[str, str]]:
        return [
            ("bold", "  Add custom model\n"),
            (
                "class:model-app.dim",
                "  Saved under agent.models.<name>; Ctrl+S to save from any field.\n",
            ),
        ]

    # ─────────────────────────────────────────────────────────────────
    # List content (data model per view)
    # ─────────────────────────────────────────────────────────────────

    def _current_items(self) -> List[Tuple[str, str]]:
        if self._view == _View.PROVIDER_LIST:
            return self._provider_items()
        if self._view == _View.PROVIDER_MODELS:
            return self._provider_models_items()
        if self._view == _View.CUSTOM_LIST:
            return self._custom_items()
        return []

    def _providers_for_tab(self, tab: _Tab) -> List[str]:
        """Return the provider keys surfaced under ``tab``.

        ``Providers`` hides the curated plan / subscription / OAuth entries
        (they live on the ``Plans`` tab); ``Plans`` does the inverse.
        Any tab other than these two returns an empty list (``Custom`` is
        not provider-backed).
        """
        if tab == _Tab.PLANS:
            return [p for p in self._providers if p in _PLAN_PROVIDERS]
        if tab == _Tab.PROVIDERS:
            return [p for p in self._providers if p not in _PLAN_PROVIDERS]
        return []

    def _provider_items(self) -> List[Tuple[str, str]]:
        """Render rows for the provider list under the active tab.

        The ``Plans`` tab omits the ``(coding plan / subscription / oauth)``
        parenthesised tags — those categories are already implicit from the
        tab membership, so the row reads as a clean display name. The
        ``Providers`` tab never has tag-worthy entries (plan / auth-typed
        providers live on ``Plans``), so the code path converges on a
        name-only label there as well.
        """
        out: List[Tuple[str, str]] = []
        on_plans = self._tab == _Tab.PLANS
        for name in self._providers_for_tab(self._tab):
            meta = self._provider_meta.get(name) or {}
            auth_type = str(meta.get("auth_type", "api_key"))
            suffix = ""
            if not on_plans:
                tags: List[str] = []
                if name in _CODING_PLAN_PROVIDERS:
                    tags.append("coding plan")
                if auth_type in ("subscription", "oauth"):
                    tags.append(auth_type)
                if tags:
                    suffix = f"  ({', '.join(tags)})"
            # \u2713 = ✓ for configured; "[needs setup]" stays as a word tag
            # because its absence is the signal — a bare checkmark for
            # "configured" pairs better with dim styling for unconfigured.
            status = "\u2713" if self._availability.get(name) else "[needs setup]"
            label = f"{_display_name(name)}{suffix}  {status}"
            is_current = name == self._current_provider
            if is_current:
                label += "  \u2190 current"
            if not self._availability.get(name):
                style = "class:model-app.dim"
            elif is_current:
                style = CLR_CURRENT
            else:
                style = ""
            out.append((label, style))
        return out

    def _provider_models_items(self) -> List[Tuple[str, str]]:
        provider = self._active_provider or ""
        out: List[Tuple[str, str]] = []
        for model in self._provider_models:
            label = model
            is_current = provider == self._current_provider and model == self._current_model
            if is_current:
                label += "  \u2190 current"
            out.append((label, CLR_CURRENT if is_current else ""))
        return out

    def _custom_items(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        models_map = self._cfg.models or {}
        for name in self._custom_names:
            cfg = models_map.get(name)
            desc = f"  [{cfg.type}/{cfg.model}]" if cfg is not None else ""
            label = f"{name}{desc}"
            is_current = name == self._current_custom
            if is_current:
                label += "  \u2190 current"
            out.append((label, CLR_CURRENT if is_current else ""))
        out.append(("+ Add model\u2026", "class:model-app.accent"))
        return out

    # ─────────────────────────────────────────────────────────────────
    # Cursor / scroll helpers
    # ─────────────────────────────────────────────────────────────────

    def _clamp_cursor(self, total: int) -> None:
        if total <= 0:
            self._list_cursor = 0
            self._list_offset = 0
            return
        if self._list_cursor >= total:
            self._list_cursor = total - 1
        if self._list_cursor < 0:
            self._list_cursor = 0

    def _visible_slice(self, total: int) -> Tuple[int, int]:
        max_visible = self._max_visible
        if total <= max_visible:
            self._list_offset = 0
            return 0, total
        if self._list_cursor < self._list_offset:
            self._list_offset = self._list_cursor
        elif self._list_cursor >= self._list_offset + max_visible:
            self._list_offset = self._list_cursor - max_visible + 1
        start = max(0, min(self._list_offset, total - max_visible))
        return start, start + max_visible

    # ─────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────

    def _enter_provider_list(self, tab: _Tab = _Tab.PROVIDERS) -> None:
        """Enter the provider list view filtered by ``tab``.

        ``tab`` must be :attr:`_Tab.PROVIDERS` or :attr:`_Tab.PLANS`; any
        other value falls back to ``PROVIDERS``.
        """
        if tab not in (_Tab.PROVIDERS, _Tab.PLANS):
            tab = _Tab.PROVIDERS
        self._tab = tab
        self._view = _View.PROVIDER_LIST
        visible = self._providers_for_tab(tab)
        self._list_cursor = self._initial_cursor(visible, self._current_provider)
        self._list_offset = 0
        self._error_message = None

    def _enter_custom_list(self) -> None:
        self._tab = _Tab.CUSTOM
        self._view = _View.CUSTOM_LIST
        self._refresh_custom_names()
        if self._current_custom and self._current_custom in self._custom_names:
            self._list_cursor = self._custom_names.index(self._current_custom)
        else:
            self._list_cursor = 0
        self._list_offset = 0
        self._error_message = None

    def _enter_provider_models(self, provider: str) -> None:
        meta = self._provider_meta.get(provider) or {}
        models = meta.get("models") or []
        self._active_provider = provider
        self._provider_models = [str(m) for m in models]
        self._view = _View.PROVIDER_MODELS
        self._error_message = None
        if provider == self._current_provider and self._current_model in self._provider_models:
            self._list_cursor = self._provider_models.index(self._current_model)
        else:
            default = str(meta.get("default_model") or "")
            self._list_cursor = self._provider_models.index(default) if default in self._provider_models else 0
        self._list_offset = 0

    def _enter_cred_form(self, provider: str) -> None:
        meta = self._provider_meta.get(provider) or {}
        user_cfg = (getattr(self._cfg, "providers", None) or {}).get(provider)
        self._active_provider = provider
        has_saved_key = user_cfg is not None and bool(getattr(user_cfg, "api_key", None))
        self._cred_api_key.text = _MASKED_KEY_PLACEHOLDER if has_saved_key else ""
        saved_base = getattr(user_cfg, "base_url", None) if user_cfg is not None else None
        self._cred_base_url.text = str(saved_base or meta.get("base_url", ""))
        self._view = _View.PROVIDER_CRED_FORM
        self._form_focus_order = [self._cred_api_key, self._cred_base_url]
        self._form_focus_idx = 0
        self._app.layout.focus(self._cred_api_key)
        self._error_message = None

    def _enter_token_form(self, provider: str) -> None:
        self._active_provider = provider
        self._token_input.text = ""
        self._view = _View.PROVIDER_TOKEN_FORM
        self._form_focus_order = [self._token_input]
        self._form_focus_idx = 0
        self._app.layout.focus(self._token_input)
        self._error_message = None

    def _enter_add_model_form(self) -> None:
        self._view = _View.ADD_MODEL_FORM
        for ta in (self._add_name, self._add_type, self._add_model, self._add_base_url, self._add_api_key):
            ta.text = ""
        self._form_focus_order = [
            self._add_name,
            self._add_type,
            self._add_model,
            self._add_base_url,
            self._add_api_key,
        ]
        self._form_focus_idx = 0
        self._app.layout.focus(self._add_name)
        self._error_message = None

    def _initial_cursor(self, items: List[str], current: Optional[str]) -> int:
        if current and current in items:
            return items.index(current)
        return 0

    # ─────────────────────────────────────────────────────────────────
    # Actions triggered from key bindings
    # ─────────────────────────────────────────────────────────────────

    def _on_provider_enter(self) -> None:
        visible = self._providers_for_tab(self._tab)
        if not visible:
            return
        if self._list_cursor < 0 or self._list_cursor >= len(visible):
            return
        provider = visible[self._list_cursor]
        meta = self._provider_meta.get(provider) or {}
        auth_type = str(meta.get("auth_type", "api_key"))
        available = self._availability.get(provider, False)
        if available:
            self._enter_provider_models(provider)
            return
        if auth_type == "api_key":
            self._enter_cred_form(provider)
        elif auth_type == "subscription":
            # Try auto-detection first; only ask if it fails.
            token = self._try_auto_detect_subscription_token(provider)
            if token:
                self._cfg.set_provider_config(
                    provider=provider,
                    api_key=token,
                    base_url=str(meta.get("base_url") or "") or None,
                    auth_type="subscription",
                )
                self._availability[provider] = True
                self._enter_provider_models(provider)
            else:
                self._enter_token_form(provider)
        elif auth_type == "oauth":
            # OAuth launches a browser + local HTTP server, which is incompatible
            # with running inside this Application's event loop. Hand off to the
            # caller by exiting with a ``needs_oauth`` result.
            self._result = ModelSelection(kind="needs_oauth", provider=provider)
            self._app.exit(result=self._result)
        else:
            self._error_message = f"Unknown auth_type `{auth_type}` for provider `{provider}`"

    def _on_edit_credentials(self) -> None:
        """Force-open the credential form for the currently highlighted provider.

        Invoked by the ``e`` shortcut on the provider list. Unlike
        :meth:`_on_provider_enter`, this bypasses the "already available →
        drill into models" short-circuit and the auto-detected-token
        optimisation, so the user can overwrite credentials they previously
        saved. OAuth providers re-enter the browser handshake via the
        ``needs_oauth`` hand-off.
        """
        visible = self._providers_for_tab(self._tab)
        if not visible or self._list_cursor < 0 or self._list_cursor >= len(visible):
            return
        provider = visible[self._list_cursor]
        meta = self._provider_meta.get(provider) or {}
        auth_type = str(meta.get("auth_type", "api_key"))
        if auth_type == "api_key":
            self._enter_cred_form(provider)
        elif auth_type == "subscription":
            self._enter_token_form(provider)
        elif auth_type == "oauth":
            self._result = ModelSelection(kind="needs_oauth", provider=provider)
            self._app.exit(result=self._result)
        else:
            self._error_message = f"Unknown auth_type `{auth_type}` for provider `{provider}`"

    def _on_model_enter(self) -> None:
        if not self._provider_models:
            return
        model = self._provider_models[self._list_cursor]
        provider = self._active_provider or ""
        self._result = ModelSelection(kind="provider_model", provider=provider, model=model)
        self._app.exit(result=self._result)

    def _on_custom_enter(self) -> None:
        total = len(self._custom_names)
        if self._list_cursor == total:
            # "+ Add model..." row
            self._enter_add_model_form()
            return
        if 0 <= self._list_cursor < total:
            name = self._custom_names[self._list_cursor]
            self._result = ModelSelection(kind="custom", name=name)
            self._app.exit(result=self._result)

    def _on_delete_custom(self) -> None:
        """Delete the highlighted ``agent.models`` entry with a two-press guard.

        Pressing ``d`` on a custom-name row the first time arms deletion
        and surfaces a confirmation in the error bar. A second ``d`` press
        on the same row commits the deletion and exits with
        ``kind="delete_custom"``. Any cursor movement, tab switch, or form
        entry clears the pending state.
        """
        if not self._custom_names:
            self._pending_delete_custom = None
            return
        if self._list_cursor < 0 or self._list_cursor >= len(self._custom_names):
            # Cursor is on the trailing "+ Add model..." row — nothing to delete.
            self._pending_delete_custom = None
            return
        name = self._custom_names[self._list_cursor]
        if self._pending_delete_custom == name:
            self._pending_delete_custom = None
            self._result = ModelSelection(kind="delete_custom", name=name)
            self._app.exit(result=self._result)
            return
        self._pending_delete_custom = name
        self._error_message = f"Delete `{name}`? Press d again to confirm, any other key to cancel."

    def _submit_cred_form(self) -> None:
        provider = self._active_provider or ""
        raw_key = self._cred_api_key.text.strip()
        api_key: str | None = None if raw_key == _MASKED_KEY_PLACEHOLDER else raw_key
        base_url = self._cred_base_url.text.strip()
        try:
            self._cfg.set_provider_config(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                auth_type="api_key",
            )
        except Exception as exc:
            self._error_message = f"Failed to save: {exc}"
            return
        available = self._safe_available(provider)
        self._availability[provider] = available
        if not available:
            env = (self._provider_meta.get(provider) or {}).get("api_key_env")
            self._error_message = f"API key is required or set env var {env}" if env else "API key is required"
            return
        self._enter_provider_models(provider)

    def _submit_token_form(self) -> None:
        provider = self._active_provider or ""
        token = self._token_input.text.strip()
        if not token:
            self._error_message = "Token cannot be empty"
            return
        meta = self._provider_meta.get(provider) or {}
        try:
            self._cfg.set_provider_config(
                provider=provider,
                api_key=token,
                base_url=str(meta.get("base_url") or "") or None,
                auth_type="subscription",
            )
        except Exception as exc:
            self._error_message = f"Failed to save: {exc}"
            return
        self._availability[provider] = True
        self._enter_provider_models(provider)

    def _submit_add_model_form(self) -> None:
        name = self._add_name.text.strip()
        type_ = self._add_type.text.strip()
        model_name = self._add_model.text.strip()
        base_url = self._add_base_url.text.strip()
        api_key = self._add_api_key.text.strip()
        if not name:
            self._error_message = "name is required"
            return
        if not type_:
            self._error_message = "type is required (e.g. openai / claude / deepseek)"
            return
        if not model_name:
            self._error_message = "model is required"
            return
        if name in (self._cfg.models or {}):
            self._error_message = f"Custom model `{name}` already exists"
            return
        payload: Dict[str, Any] = {"type": type_, "model": model_name}
        if base_url:
            payload["base_url"] = base_url
        if api_key:
            payload["api_key"] = api_key
        self._result = ModelSelection(kind="add_custom", name=name, payload=payload)
        self._app.exit(result=self._result)

    # ─────────────────────────────────────────────────────────────────
    # External helpers (synchronous, no stdin access)
    # ─────────────────────────────────────────────────────────────────

    def _try_auto_detect_subscription_token(self, provider: str = "claude_subscription") -> Optional[str]:
        try:
            from datus.auth.claude_credential import get_claude_subscription_token

            user_cfg = self._cfg.providers.get(provider, ProviderConfig())
            token, _ = get_claude_subscription_token(api_key_from_config=user_cfg.api_key or "")
            return token or None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────
    # Key bindings
    # ─────────────────────────────────────────────────────────────────

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_list = Condition(lambda: self._view in {_View.PROVIDER_LIST, _View.PROVIDER_MODELS, _View.CUSTOM_LIST})
        is_provider_list = Condition(lambda: self._view == _View.PROVIDER_LIST)
        is_custom_list = Condition(lambda: self._view == _View.CUSTOM_LIST)
        is_form = Condition(
            lambda: self._view in {_View.PROVIDER_CRED_FORM, _View.PROVIDER_TOKEN_FORM, _View.ADD_MODEL_FORM}
        )

        def _clear_pending_delete() -> None:
            # Any key press other than ``d`` on the same row cancels a
            # pending two-press confirmation. Call this from every list
            # binding that meaningfully changes focus/state.
            self._pending_delete_custom = None

        @kb.add("up", filter=is_list)
        def _(event):
            items = self._current_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor - 1) % len(items)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("down", filter=is_list)
        def _(event):
            items = self._current_items()
            if not items:
                return
            self._list_cursor = (self._list_cursor + 1) % len(items)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("pageup", filter=is_list)
        def _(event):
            self._list_cursor = max(0, self._list_cursor - 10)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("pagedown", filter=is_list)
        def _(event):
            items = self._current_items()
            self._list_cursor = min(max(0, len(items) - 1), self._list_cursor + 10)
            self._error_message = None
            _clear_pending_delete()

        @kb.add("enter", filter=is_list)
        def _(event):
            _clear_pending_delete()
            if self._view == _View.PROVIDER_LIST:
                self._on_provider_enter()
            elif self._view == _View.PROVIDER_MODELS:
                self._on_model_enter()
            elif self._view == _View.CUSTOM_LIST:
                self._on_custom_enter()

        @kb.add("e", filter=is_provider_list)
        def _(event):
            self._on_edit_credentials()

        @kb.add("d", filter=is_custom_list)
        def _(event):
            # The handler itself manages the pending-delete flag, so this
            # binding must *not* invoke ``_clear_pending_delete``.
            self._on_delete_custom()

        @kb.add("tab", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(+1)

        @kb.add("s-tab", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(-1)

        @kb.add("right", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(+1)

        @kb.add("left", filter=is_list)
        def _(event):
            _clear_pending_delete()
            self._cycle_tab(-1)

        @kb.add("escape", filter=is_list)
        def _(event):
            _clear_pending_delete()
            if self._view == _View.PROVIDER_MODELS:
                self._enter_provider_list(self._tab)
            else:
                event.app.exit(result=None)

        # Form navigation --------------------------------------------------
        @kb.add("tab", filter=is_form)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("s-tab", filter=is_form)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("down", filter=is_form)
        def _(event):
            self._advance_form_focus(+1)

        @kb.add("up", filter=is_form)
        def _(event):
            self._advance_form_focus(-1)

        @kb.add("enter", filter=is_form)
        def _(event):
            # On the last field, submit; otherwise advance.
            if self._form_focus_idx >= len(self._form_focus_order) - 1:
                self._submit_current_form()
            else:
                self._advance_form_focus(+1)

        @kb.add("c-s", filter=is_form)
        def _(event):
            self._submit_current_form()

        @kb.add("escape", filter=is_form)
        def _(event):
            if self._view == _View.ADD_MODEL_FORM:
                self._enter_custom_list()
            else:
                # Cred / token forms were opened from a plan or provider tab;
                # return to whichever one is active.
                self._enter_provider_list(self._tab)

        # Global cancel ----------------------------------------------------
        @kb.add("c-c")
        def _(event):
            event.app.exit(result=None)

        return kb

    # ─────────────────────────────────────────────────────────────────
    # Form focus helpers
    # ─────────────────────────────────────────────────────────────────

    def _advance_form_focus(self, delta: int) -> None:
        if not self._form_focus_order:
            return
        self._form_focus_idx = (self._form_focus_idx + delta) % len(self._form_focus_order)
        self._app.layout.focus(self._form_focus_order[self._form_focus_idx])

    def _submit_current_form(self) -> None:
        if self._view == _View.PROVIDER_CRED_FORM:
            self._submit_cred_form()
        elif self._view == _View.PROVIDER_TOKEN_FORM:
            self._submit_token_form()
        elif self._view == _View.ADD_MODEL_FORM:
            self._submit_add_model_form()

    def _cycle_tab(self, direction: int = 1) -> None:
        """Rotate the active tab by ``direction`` (±1) in ``_TAB_CYCLE`` order."""
        try:
            idx = _TAB_CYCLE.index(self._tab)
        except ValueError:
            idx = 0
        next_tab = _TAB_CYCLE[(idx + direction) % len(_TAB_CYCLE)]
        if next_tab == _Tab.CUSTOM:
            self._enter_custom_list()
        else:
            self._enter_provider_list(next_tab)


__all__ = ["ModelApp", "ModelSelection"]
