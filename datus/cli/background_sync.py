# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Background metadata RAG sync coordinator for the CLI.

Keeps the LanceDB schema index in ``~/.datus/data/{project}/datus_db/``
aligned with the current datasource so ``@Table`` autocompletion reflects
freshly added tables without a manual ``datus agent init`` rebuild. The
coordinator is intentionally a thin scheduler on top of the already-async
``init_local_schema_async`` helper: cancel-before-restart, coalesce
duplicate requests, funnel failures to ``warning`` logs rather than
surfacing stack traces to the user.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import weakref
from typing import TYPE_CHECKING, Optional

from datus.utils.loggings import get_logger

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class BackgroundSchemaSyncManager:
    """Single-slot async scheduler for metadata RAG incremental refresh.

    At most one sync runs at a time. Scheduling a new sync cancels the
    pending one so rapid ``/datasource`` toggling never piles up work on
    the background loop. All state mutation is guarded by ``self._lock``
    so :meth:`is_running` (called from the status bar on the prompt
    thread) stays race-free versus :meth:`schedule` / :meth:`cancel`.
    """

    def __init__(self, cli: "DatusCLI"):
        # weakref avoids keeping DatusCLI alive past shutdown if this
        # manager outlives its owner (e.g. during test teardown).
        self._cli_ref = weakref.ref(cli)
        self._lock = threading.RLock()
        self._current_future: Optional[concurrent.futures.Future] = None
        self._current_datasource: Optional[str] = None
        self._shutdown = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def schedule(self, *, datasource: str, reason: str = "switch") -> None:
        """Queue an incremental metadata sync for ``datasource``.

        Returns immediately; the actual sync runs on ``cli._bg_loop``.
        ``reason`` is only used for logs so operators can trace which CLI
        path triggered the refresh.
        """
        if self._shutdown:
            return
        cli = self._cli_ref()
        if cli is None:
            return

        ac = getattr(cli.agent_config, "autocomplete", None)
        if ac is None or not ac.background_sync_enabled:
            logger.debug("background sync skipped: feature disabled (reason=%s)", reason)
            return
        if not datasource:
            return

        with self._lock:
            prev = self._current_future
            prev_ds = self._current_datasource
            # Coalesce: a running sync for the same datasource supersedes
            # any duplicate request. An in-flight cancel will no-op here.
            if prev is not None and not prev.done() and prev_ds == datasource:
                logger.debug("background sync coalesced for %s (reason=%s)", datasource, reason)
                return
            # Replace: a sync for a *different* datasource is now stale.
            if prev is not None and not prev.done():
                logger.debug("background sync cancelling previous task for %s", prev_ds)
                prev.cancel()

            include_values = bool(ac.background_sync_include_values)
            coro = self._run_sync(cli, datasource, reason, include_values)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, cli._bg_loop)
            except RuntimeError as exc:
                # Bg loop already stopped (shutdown path). Close the
                # coroutine explicitly so Python does not warn about an
                # un-awaited coroutine on GC.
                coro.close()
                logger.debug("background sync skipped: loop unavailable (%s)", exc)
                return
            self._current_future = future
            self._current_datasource = datasource
            future.add_done_callback(self._on_done)
            logger.info("background sync scheduled for %s (reason=%s)", datasource, reason)

        self._notify_status_change()

    def cancel(self) -> None:
        """Cancel any running sync. Subsequent :meth:`is_running` returns ``False``."""
        with self._lock:
            fut = self._current_future
        if fut is not None and not fut.done():
            fut.cancel()

    def is_running(self) -> bool:
        with self._lock:
            fut = self._current_future
            return fut is not None and not fut.done()

    def shutdown(self) -> None:
        """Disable further scheduling and cancel any pending sync.

        Called during ``DatusCLI`` exit so we do not leak a background
        task past the bg loop teardown.
        """
        self._shutdown = True
        self.cancel()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _run_sync(
        self,
        cli: "DatusCLI",
        datasource: str,
        reason: str,
        include_values: bool,
    ) -> None:
        # Defer imports to avoid pulling storage stack into CLI startup
        # when the feature is disabled.
        from datus.storage.schema_metadata.local_init import init_local_schema_async
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        try:
            # Drift guard: main thread may have switched datasource again
            # between scheduling and execution. Proceed only if the CLI's
            # current datasource still matches; otherwise the next scheduled
            # sync will take over.
            current = getattr(cli.agent_config, "current_datasource", "")
            if current != datasource:
                logger.debug(
                    "background sync skipped for %s: current datasource drifted to %s",
                    datasource,
                    current,
                )
                return

            store = SchemaWithValueRAG(cli.agent_config)
            await init_local_schema_async(
                store,
                cli.agent_config,
                cli.db_manager,
                build_mode="incremental",
                table_type="full" if include_values else "table",
            )

            # Swap autocomplete cache under its RLock. The completer's
            # build-then-swap path keeps readers from seeing torn state.
            completer = getattr(cli, "at_completer", None)
            if completer is not None:
                try:
                    completer.table_completer.reload_data()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("failed to reload table completer after sync: %s", exc)

            logger.info("background sync finished for %s (reason=%s)", datasource, reason)
        except asyncio.CancelledError:
            logger.debug("background sync cancelled for %s", datasource)
            raise
        except Exception as exc:
            logger.warning("background sync failed for %s: %s", datasource, exc)

    def _on_done(self, future: concurrent.futures.Future) -> None:
        with self._lock:
            if self._current_future is future:
                self._current_future = None
                self._current_datasource = None
        self._notify_status_change()

    def _notify_status_change(self) -> None:
        """Prompt the status bar / TUI to redraw so ``is_running`` flips."""
        cli = self._cli_ref()
        if cli is None:
            return
        tui_app = getattr(cli, "tui_app", None)
        if tui_app is not None:
            try:
                tui_app.invalidate()
            except Exception:  # pragma: no cover - defensive
                pass
