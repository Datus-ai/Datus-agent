"""Own one MCP stdio child per test, with bounded phases and child diagnostics."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Register before importing the target so a stalled import can also be inspected.
# Keep Python's -c sys.path (including cwd) and the target's original CLI arguments.
_CHILD_BOOTSTRAP = """
import faulthandler
import os
import runpy
import signal
import sys
from pathlib import Path

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
Path(sys.argv[1]).write_text(str(os.getpid()))
sys.argv = sys.argv[2:]
if sys.argv[0] == "-m":
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


class StdioTestServer:
    """Use the SDK transport and cleanup while keeping startup and test timeouts local.

    ``params.command`` must be Python, with a module or script in ``params.args``.
    SDK contexts stay on one owner task even when pytest runs fixture teardown
    on a different task. Borrowed sessions never own the child.
    """

    def __init__(
        self,
        params: StdioServerParameters,
        diagnostics_dir: Path,
        *,
        startup_timeout: float = 30,
        call_timeout: float = 30,
        shutdown_timeout: float = 10,
    ):
        self.diagnostics_dir = diagnostics_dir.resolve()
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self._pid_file = self.diagnostics_dir / "child.pid"
        self._params = params.model_copy(
            update={"args": ["-u", "-c", _CHILD_BOOTSTRAP, str(self._pid_file), *params.args]}
        )
        self._startup_timeout = startup_timeout
        self._call_timeout = call_timeout
        self._shutdown_timeout = shutdown_timeout
        self._session: ClientSession | None = None
        self._started = time.monotonic()

    @property
    def pid(self) -> int | None:
        try:
            return int(self._pid_file.read_text())
        except (OSError, ValueError):
            return None

    def _record(self, phase: str, event: str, **fields) -> None:
        record = {"phase": phase, "event": event, "elapsed_seconds": time.monotonic() - self._started, **fields}
        with (self.diagnostics_dir / "phases.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    async def _dump_child(self, phase: str) -> None:
        pid = self.pid
        self._record(phase, "timeout", pid=pid)
        if pid is not None and hasattr(signal, "SIGUSR1"):
            try:
                os.kill(pid, signal.SIGUSR1)
            except OSError as exc:
                self._record(phase, "stack_unavailable", error=type(exc).__name__)
            else:
                # Allow the signal handler to write stderr before SDK cleanup closes it.
                await asyncio.sleep(0.1)

    def _kill_child(self) -> None:
        pid = self.pid
        if pid is None:
            return
        try:
            if os.name == "posix" and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass

    def _timeout_error(self, phase: str, budget: float) -> TimeoutError:
        return TimeoutError(f"MCP stdio {phase} exceeded {budget:g}s; diagnostics: {self.diagnostics_dir}")

    async def _close(self, stack: AsyncExitStack) -> None:
        self._record("shutdown", "start", pid=self.pid)
        expired = False

        async def watchdog():
            nonlocal expired
            await asyncio.sleep(self._shutdown_timeout)
            expired = True
            await self._dump_child("shutdown")
            self._kill_child()

        # Do not cancel AsyncExitStack.aclose(): SDK task groups must exit on
        # their owning task. Kill an unresponsive child and let the SDK reap it.
        task = asyncio.create_task(watchdog())
        try:
            await stack.aclose()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._record("shutdown", "finished", timed_out=expired)
        if expired:
            raise self._timeout_error("shutdown", self._shutdown_timeout)

    async def _serve(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        stack = AsyncExitStack()
        self._record("startup", "start")
        with (self.diagnostics_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                try:
                    async with asyncio.timeout(self._startup_timeout):
                        read, write = await stack.enter_async_context(stdio_client(self._params, errlog=stderr))
                        self._record("startup", "spawned")
                        session = await stack.enter_async_context(ClientSession(read, write))
                        self._record("startup", "initialize_start")
                        await session.initialize()
                        self._record("startup", "initialize_finished", pid=self.pid)
                        result = await session.list_tools()
                        assert result.tools, "MCP stdio readiness probe returned no tools"
                except TimeoutError as exc:
                    await self._dump_child("startup")
                    raise self._timeout_error("startup", self._startup_timeout) from exc
                self._session = session
                self._record("startup", "ready", pid=self.pid)
                ready.set_result(None)
                await stop.wait()
            finally:
                self._session = None
                await self._close(stack)

    @asynccontextmanager
    async def running(self) -> AsyncIterator[StdioTestServer]:
        ready = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def owner():
            try:
                await self._serve(ready, stop)
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                raise

        task = asyncio.create_task(owner())
        try:
            await asyncio.shield(ready)
            yield self
        finally:
            stop.set()
            await task

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        assert self._session is not None, "Start the stdio server before borrowing its session"
        self._record("test", "start")
        try:
            async with asyncio.timeout(self._call_timeout):
                yield self._session
        except TimeoutError as exc:
            await self._dump_child("test")
            raise self._timeout_error("test", self._call_timeout) from exc
        finally:
            self._record("test", "finished")
