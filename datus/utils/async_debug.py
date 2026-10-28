# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Async task stack dumping for diagnosing hangs.

When a request appears stuck — e.g. a tool awaiting an
:class:`~datus.cli.execution_state.InteractionBroker` confirmation that never
arrives in a non-interactive surface — the event loop itself is *not* blocked:
it keeps running while some coroutine sits on ``await future``. This module
walks every live :class:`asyncio.Task` and renders its stack so we can see
exactly which coroutine is parked and where.

Two entry points:

* :func:`format_async_tasks` / :func:`dump_async_tasks_to_log` — pull the
  snapshot on demand (e.g. from a debug HTTP endpoint that still responds
  because the loop is alive).
* :func:`install_task_dump_signal_handler` — install a ``SIGUSR1`` handler so
  an operator can ``kill -USR1 <pid>`` a daemonised server and get the dump in
  the log without an open HTTP connection.
"""

import asyncio
import gc
import io
import linecache
import signal
from types import FrameType
from typing import Any, List, Optional

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Resolved at import time so the default is safe on platforms lacking ``SIGUSR1``
# (e.g. Windows): ``getattr`` yields ``None`` there instead of raising, letting
# ``install_task_dump_signal_handler`` return ``False`` gracefully.
DEFAULT_TASK_DUMP_SIGNAL: Optional[int] = getattr(signal, "SIGUSR1", None)

# Loop captured at signal-handler install time. ``asyncio.all_tasks()`` needs a
# loop reference, and a signal handler firing inside the running loop cannot
# rely on ``get_running_loop()`` being callable, so we stash it here.
_dump_loop: Optional[asyncio.AbstractEventLoop] = None


def _resolve_loop(loop: Optional[asyncio.AbstractEventLoop]) -> Optional[asyncio.AbstractEventLoop]:
    """Pick the loop to introspect: explicit arg, then running, then captured."""
    if loop is not None:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return _dump_loop


def _frame_of(obj: Any) -> Optional[FrameType]:
    """Return the code frame backing a coroutine / async-gen / generator."""
    for attr in ("cr_frame", "ag_frame", "gi_frame"):
        frame = getattr(obj, attr, None)
        if frame is not None:
            return frame
    return None


def _next_awaitable(obj: Any) -> Any:
    """Follow what ``obj`` is currently awaiting / yielding from, if anything.

    Coroutines/async-gens/generators expose the link directly. The
    ``async_generator_asend`` wrapper produced by ``async for`` / ``__anext__``
    exposes nothing (a CPython dead end), so we bridge it via ``gc`` referents:
    the wrapper holds a reference to the underlying async generator, which does
    carry a frame and ``ag_await``. Without this hop the chain would stop at the
    ``async for`` consumer and never reach the generator's real block.
    """
    for attr in ("cr_await", "ag_await", "gi_yieldfrom"):
        nxt = getattr(obj, attr, None)
        if nxt is not None:
            return nxt
    # Bridge async-generator boundaries: find a frame-bearing referent.
    if hasattr(obj, "__await__") and _frame_of(obj) is None:
        try:
            for ref in gc.get_referents(obj):
                if _frame_of(ref) is not None:
                    return ref
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _format_await_chain(root: Any, limit: Optional[int] = None) -> str:
    """Render the *suspended* frame chain reachable from ``root``.

    ``Task.print_stack`` stops at the directly-suspended coroutine and does not
    descend across async-generator (``async for``) or nested-await boundaries —
    so a consumer parked on ``async for action in gen`` shows only that line,
    hiding where ``gen`` itself is actually blocked. This walks the
    ``cr_await`` / ``ag_await`` / ``gi_yieldfrom`` links to surface the deepest
    frame (e.g. the ``await broker.request(...)`` that never returns).

    Returns one ``File ... line ... in ...`` block per frame, innermost last,
    mirroring ``traceback`` ordering. Cycle- and depth-guarded so a malformed
    chain can never spin.
    """
    lines: List[str] = []
    seen: set[int] = set()
    obj = root
    max_depth = 200  # hard backstop independent of ``limit``
    while obj is not None and id(obj) not in seen and len(lines) < max_depth:
        seen.add(id(obj))
        frame = _frame_of(obj)
        if frame is not None:
            code = frame.f_code
            lineno = frame.f_lineno
            src = linecache.getline(code.co_filename, lineno).strip()
            lines.append(f'  File "{code.co_filename}", line {lineno}, in {code.co_name}')
            if src:
                lines.append(f"    {src}")
        obj = _next_awaitable(obj)

    if limit is not None and limit > 0:
        # Keep the innermost ``limit`` frames (2 text lines each) — those are
        # the ones nearest the actual block.
        lines = lines[-(limit * 2) :]
    if not lines:
        return "  <no descendable frames; coroutine may be awaiting a bare Future>\n"
    return "\n".join(lines) + "\n"


def format_async_tasks(loop: Optional[asyncio.AbstractEventLoop] = None, limit: Optional[int] = None) -> str:
    """Render every live asyncio task on ``loop`` with its full suspended chain.

    For each task this prints two views:

    * the ``Task.print_stack`` output (the caller chain up to the directly
      suspended coroutine), and
    * an ``await chain`` that descends across ``async for`` / nested-await
      boundaries to the innermost parked frame — the part ``print_stack`` omits
      and where the real block (e.g. ``await broker.request(...)``) lives.

    Args:
        loop: Event loop to introspect. Defaults to the running loop, falling
            back to the loop captured by :func:`install_task_dump_signal_handler`.
        limit: Max stack frames per view (``None`` for the full stack).

    Returns:
        A human-readable multi-task dump. Never raises — introspection errors
        are folded into the returned text so callers (signal handler, HTTP
        endpoint) stay simple.
    """
    target = _resolve_loop(loop)
    if target is None:
        return "async-task-dump: no event loop available to introspect\n"

    try:
        tasks: List[asyncio.Task] = list(asyncio.all_tasks(target))
    except Exception as e:  # pragma: no cover - defensive
        return f"async-task-dump: failed to enumerate tasks: {e!r}\n"

    buf = io.StringIO()
    buf.write(f"=== async task dump: {len(tasks)} live task(s) ===\n")
    # Stable, readable ordering: done tasks last, otherwise by name.
    for task in sorted(tasks, key=lambda t: (t.done(), t.get_name())):
        coro = task.get_coro()
        coro_name = getattr(coro, "__qualname__", repr(coro))
        buf.write(f"\n--- Task {task.get_name()!r} done={task.done()} coro={coro_name} ---\n")
        try:
            task.print_stack(limit=limit, file=buf)
        except Exception as e:  # pragma: no cover - defensive
            buf.write(f"<failed to print stack: {e!r}>\n")
        # The await chain is the diagnostically useful part: it crosses the
        # async-generator boundary that print_stack stops at.
        try:
            chain = _format_await_chain(coro, limit=limit)
        except Exception as e:  # pragma: no cover - defensive
            chain = f"  <failed to walk await chain: {e!r}>\n"
        buf.write("await chain (innermost last):\n")
        buf.write(chain)
    buf.write("\n=== end async task dump ===\n")
    return buf.getvalue()


def dump_async_tasks_to_log(loop: Optional[asyncio.AbstractEventLoop] = None, limit: Optional[int] = None) -> str:
    """Format the task dump and emit it to the logger at WARNING.

    Returns the same text it logs so HTTP callers can echo it back.
    """
    text = format_async_tasks(loop=loop, limit=limit)
    logger.warning("Async task stack dump requested:\n%s", text)
    return text


def install_task_dump_signal_handler(
    loop: Optional[asyncio.AbstractEventLoop] = None, sig: Optional[int] = DEFAULT_TASK_DUMP_SIGNAL
) -> bool:
    """Install a signal handler that dumps async task stacks to the log.

    Captures ``loop`` (or the running loop) so the dump works even when the
    handler fires while the loop is mid-iteration. ``SIGUSR1`` is unused by
    uvicorn/Datus, so it is safe to repurpose for diagnostics.

    Returns ``True`` if the handler was installed. Returns ``False`` on
    platforms without the signal (e.g. Windows lacks ``SIGUSR1``) or when no
    loop can be resolved — never raises, so server startup is unaffected.
    """
    global _dump_loop

    resolved = _resolve_loop(loop)
    if resolved is None:
        logger.debug("Task-dump signal handler not installed: no event loop available")
        return False
    _dump_loop = resolved

    if sig is None or not hasattr(signal, "Signals"):  # pragma: no cover - defensive
        return False

    def _handler(*_args) -> None:
        # Keep the handler body minimal and exception-proof: a raise here would
        # propagate into whatever the loop was running.
        try:
            dump_async_tasks_to_log(loop=_dump_loop)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Async task dump signal handler failed: %r", e)

    # Prefer loop.add_signal_handler (runs in the loop thread, async-safe).
    # Fall back to signal.signal where the loop API is unavailable (e.g. the
    # handler is installed off the main thread on some platforms).
    try:
        resolved.add_signal_handler(sig, _handler)
        logger.info("Installed async task-dump handler on signal %s (send: kill -%s <pid>)", sig, int(sig))
        return True
    except (NotImplementedError, RuntimeError, ValueError) as e:
        logger.debug("loop.add_signal_handler unavailable (%r); falling back to signal.signal", e)

    try:
        signal.signal(sig, _handler)
        logger.info("Installed async task-dump handler on signal %s via signal.signal", sig)
        return True
    except (OSError, ValueError, RuntimeError) as e:
        logger.debug("Task-dump signal handler not installed: %r", e)
        return False
