# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/model`` slash command — self-contained two-tab picker.

Entry point
-----------

  - ``/model``                      — open the interactive picker.
  - ``/model openai/gpt-4.1``       — direct provider/model switch (no UI).
  - ``/model custom:my-internal``   — direct custom-entry switch.
  - ``/model openai``               — open the picker already drilled into
    ``openai`` (its model list, or credential form if unavailable).

The interactive path delegates to :class:`datus.cli.model_app.ModelApp`, a
single prompt_toolkit Application that hosts tab switching, provider
drill-down, credential capture, and custom-model creation in **one**
event loop. In TUI mode we wrap the run in
:meth:`datus.cli.tui.app.DatusApp.suspend_input` so the outer persistent
Application releases ``stdin`` exactly once for the entire flow — no
nested sub-Applications, no ``stdin`` contention.

OAuth providers remain outside the Application because the browser
handshake spins its own HTTP callback server. When ``ModelApp`` returns
``needs_oauth``, this module runs :func:`configure_codex_oauth` directly
and re-enters the Application with the provider now pre-available.
"""

from __future__ import annotations

import shlex
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

from datus.cli.cli_styles import print_error, print_success, print_warning
from datus.cli.model_app import ModelApp, ModelSelection, _display_name
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

_MAX_OAUTH_RETRY_LOOPS = 2


class ModelCommands:
    """Handlers for the ``/model`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console
        self.agent_config = cli.agent_config

    # ── Entry point ──────────────────────────────────────────────────────

    def cmd_model(self, args: str) -> None:
        """Dispatch ``/model`` based on its argument shape."""
        token = (args or "").strip()
        if not token:
            self._run_menu()
            return
        first = shlex.split(token)[0]
        if first.startswith("custom:"):
            self._switch_to_custom(first[len("custom:") :])
            return
        if "/" in first:
            provider, _, model = first.partition("/")
            self._switch_to_provider_model(provider, model)
            return
        # Provider-only shortcut: open the picker seeded on that provider.
        self._run_menu(seed_provider=first)

    # ── Menu flow ────────────────────────────────────────────────────────

    def _run_menu(self, seed_provider: Optional[str] = None) -> None:
        """Loop until the user cancels or a terminal selection is made.

        ``needs_oauth`` is the only non-terminal result: we drive the OAuth
        handshake outside the Application, persist the credentials, then
        re-enter with the provider pre-selected so the user lands directly
        on its model list.
        """
        seed = seed_provider
        for _ in range(_MAX_OAUTH_RETRY_LOOPS):
            app = ModelApp(self.agent_config, self.console, seed_provider=seed)
            seed = None
            selection = self._run_app(app)
            if selection is None:
                return
            if selection.kind == "provider_model":
                self._switch_to_provider_model(selection.provider or "", selection.model or "")
                return
            if selection.kind == "custom":
                self._switch_to_custom(selection.name or "")
                return
            if selection.kind == "add_custom":
                if self._persist_custom_model(selection.name or "", selection.payload or {}):
                    self._switch_to_custom(selection.name or "")
                return
            if selection.kind == "delete_custom":
                self._delete_custom_model(selection.name or "")
                return
            if selection.kind == "needs_oauth":
                if self._run_oauth_flow(selection.provider or ""):
                    seed = selection.provider
                    continue
                return
            logger.debug("ModelApp returned unknown kind=%s", selection.kind)
            return

    def _run_app(self, app: ModelApp) -> Optional[ModelSelection]:
        """Run ``app`` with stdin handed over by the outer TUI (if any)."""
        tui_app = getattr(self.cli, "tui_app", None)
        if tui_app is not None:
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    # ── OAuth handoff ───────────────────────────────────────────────────

    def _run_oauth_flow(self, provider: str) -> bool:
        """Execute the browser OAuth handshake for ``provider``.

        The model name used for the post-handshake connectivity probe is
        seeded from the provider's ``default_model`` (or first model in
        the catalog). The subsequent :class:`ModelApp` re-entry asks the
        user to pick a model explicitly so this probe choice has no
        user-visible effect.
        """
        meta = self._provider_meta(provider)
        if not meta:
            print_error(self.console, f"Unknown provider: {provider}")
            return False
        from datus.cli.provider_auth_flows import configure_codex_oauth

        probe_model = str(meta.get("default_model") or next(iter(meta.get("models") or []), "") or "")
        tui_app = getattr(self.cli, "tui_app", None)
        ctx = tui_app.suspend_input() if tui_app is not None else nullcontext()
        with ctx:
            result = configure_codex_oauth(self.console, provider, meta, model_name=probe_model or None)
        if not result:
            return False
        try:
            self.agent_config.set_provider_config(
                provider=provider,
                api_key=None,
                base_url=result.get("base_url"),
                auth_type="oauth",
            )
        except Exception as exc:
            print_error(self.console, f"Failed to persist OAuth credentials: {exc}")
            return False
        return True

    # ── Custom-model persistence ────────────────────────────────────────

    def _persist_custom_model(self, name: str, payload: Dict[str, Any]) -> bool:
        """Register a new ``agent.models[<name>]`` entry in memory and on disk."""
        if not name or not payload:
            print_error(self.console, "Refusing to persist empty custom model")
            return False
        try:
            from datus.configuration.agent_config import load_model_config
            from datus.configuration.agent_config_loader import configuration_manager

            mgr = configuration_manager()
            mgr.update_item("models", {name: dict(payload)}, delete_old_key=False, save=True)
            self.agent_config.models[name] = load_model_config(dict(payload))
        except Exception as exc:
            print_error(self.console, f"Failed to save custom model `{name}`: {exc}")
            return False
        print_success(self.console, f"Saved custom model `{name}`")
        return True

    def _delete_custom_model(self, name: str) -> bool:
        """Remove an ``agent.models[<name>]`` entry from memory and YAML.

        If the deleted entry was the currently active target, the caller
        is left responsible for picking a new one — we only drop the
        custom binding so the user can re-open ``/model`` without the
        stale row in the list.
        """
        if not name:
            return False
        models_map = self.agent_config.models or {}
        if name not in models_map:
            print_warning(self.console, f"Custom model `{name}` not found")
            return False
        try:
            from datus.configuration.agent_config_loader import configuration_manager

            mgr = configuration_manager()
            remaining = dict((mgr.get("models", {}) or {}))
            remaining.pop(name, None)
            mgr.update_item("models", remaining, delete_old_key=True, save=True)
            del models_map[name]
            if getattr(self.agent_config, "target", "") == name:
                self.agent_config.target = ""
        except Exception as exc:
            print_error(self.console, f"Failed to delete custom model `{name}`: {exc}")
            return False
        print_success(self.console, f"Deleted custom model `{name}`")
        return True

    # ── Switch (no agent rebuild) ───────────────────────────────────────

    def _switch_to_provider_model(self, provider: str, model: str) -> None:
        try:
            self.agent_config.set_active_provider_model(provider, model)
        except Exception as e:
            print_error(self.console, f"Failed to switch: {e}")
            return
        print_success(self.console, f"Switched to {_display_name(provider)}/{model}")

    def _switch_to_custom(self, name: str) -> None:
        try:
            self.agent_config.set_active_custom(name)
        except Exception as e:
            print_error(self.console, f"Failed to switch: {e}")
            return
        print_success(self.console, f"Switched to custom:{name}")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _provider_meta(self, provider: str) -> Dict[str, Any]:
        catalog = self.agent_config.provider_catalog
        providers_meta = catalog.get("providers", {}) if isinstance(catalog, dict) else {}
        meta = providers_meta.get(provider) if isinstance(providers_meta, dict) else None
        return meta if isinstance(meta, dict) else {}
