# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
import math
import os
import time
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from agents.mcp import MCPServer

logger = get_logger(__name__)

# MCP servers are connected before the agent produces its first token, so a
# server that accepts the TCP connection but never answers the JSON-RPC
# handshake (a wrong URL that serves HTML, for instance) makes the whole turn
# look frozen. These caps bound that blast radius: per attempt, and across all
# servers of one run.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15.0
DEFAULT_CONNECT_BUDGET_SECONDS = 30.0
RETRY_BACKOFF_SECONDS = 1.0

_CANCEL_SCOPE_ERROR = "Attempted to exit cancel scope in a different task than it was entered in"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"Ignoring non-numeric {name}={raw!r}, falling back to {default}s")
        return default
    # `nan <= 0` is False and `inf` is positive, so both slip past a plain
    # sign check and reach asyncio.timeout() as a deadline that never fires.
    if not math.isfinite(value) or value <= 0:
        logger.warning(f"Ignoring non-positive {name}={raw!r}, falling back to {default}s")
        return default
    return value


def connect_timeout_seconds() -> float:
    """Per-attempt cap on one MCP handshake."""
    return _env_float("DATUS_MCP_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_SECONDS)


def connect_budget_seconds() -> float:
    """Cap on the total time spent connecting every MCP server of one run."""
    return _env_float("DATUS_MCP_CONNECT_BUDGET", DEFAULT_CONNECT_BUDGET_SECONDS)


async def _close_quietly(server_name: str, server: "MCPServer") -> None:
    """Best-effort teardown; MCP transports raise noisy anyio errors when closed
    from a task other than the one that opened them."""
    try:
        await server.__aexit__(None, None, None)
    except asyncio.CancelledError:
        raise
    except RuntimeError as e:
        if _CANCEL_SCOPE_ERROR in str(e):
            logger.debug(f"Suppressed cancel scope error while closing MCP server {server_name}")
        else:
            logger.debug(f"Error while closing MCP server {server_name}: {e}")
    except Exception as e:
        logger.debug(f"Error while closing MCP server {server_name}: {e}")


@asynccontextmanager
async def _safe_connect_server(
    server_name: str,
    server: "MCPServer",
    max_retries: int = 3,
    connect_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
) -> AsyncGenerator["MCPServer", None]:
    """Context-managed safe MCP server connection.

    Args:
        server_name: Name used for logging.
        server: Server instance created via ``MCPManager._create_server_instance``.
        max_retries: How many handshake attempts before giving up.
        connect_timeout: Per-attempt cap in seconds; defaults to
            ``DATUS_MCP_CONNECT_TIMEOUT``.
        deadline: Optional ``time.monotonic()`` deadline shared across servers.
            Attempts are shortened to fit it and stop once it passes.
    """
    if connect_timeout is None:
        connect_timeout = connect_timeout_seconds()

    last_error: Optional[BaseException] = None

    for attempt in range(max_retries):
        attempt_timeout = connect_timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"MCP server {server_name}: connect budget exhausted, giving up")
                break
            attempt_timeout = min(attempt_timeout, remaining)

        logger.info(
            f"Attempting to connect to MCP server {server_name} "
            f"(attempt {attempt + 1}/{max_retries}, timeout {attempt_timeout:.1f}s)"
        )
        logger.debug(f"MCP server {server_name} type: {type(server)}")

        try:
            # Only the handshake is bounded — the caller's work below runs
            # outside the timeout and must never be cancelled by it.
            async with asyncio.timeout(attempt_timeout):
                await server.__aenter__()
        except asyncio.CancelledError:
            logger.debug(f"MCP server {server_name} connection cancelled")
            raise
        except TimeoutError as e:
            last_error = e
            logger.error(
                f"Timeout connecting to MCP server {server_name} after {attempt_timeout:.1f}s (attempt {attempt + 1})"
            )
            await _close_quietly(server_name, server)
        except Exception as e:
            last_error = e
            logger.error(f"Failed to connect MCP server {server_name} (attempt {attempt + 1}): {str(e)}")
            await _close_quietly(server_name, server)
        else:
            logger.info(f"MCP server {server_name} connected successfully")
            try:
                yield server
            except GeneratorExit:
                logger.debug(f"MCP server {server_name} generator being closed")
                raise
            finally:
                await _close_quietly(server_name, server)
            return

        if attempt < max_retries - 1:
            backoff = RETRY_BACKOFF_SECONDS
            if deadline is not None:
                # Sleeping past the deadline would push the caller's first
                # token out beyond the cap the budget promises.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(f"MCP server {server_name}: connect budget exhausted, giving up")
                    break
                backoff = min(backoff, remaining)

            if backoff > 0:
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    logger.debug(f"MCP server {server_name} retry cancelled")
                    raise

    raise last_error if last_error else TimeoutError(f"Could not connect MCP server {server_name} within budget")


@asynccontextmanager
async def multiple_mcp_servers(mcp_servers: Dict[str, Any], connect_budget: Optional[float] = None):
    """Context manager for managing multiple MCP servers.

    Servers that cannot be reached are skipped rather than failing the run, and
    the whole connect phase is capped by ``connect_budget`` so one unreachable
    server cannot stall the agent for minutes.

    Args:
        mcp_servers: Dictionary of MCP servers to manage
        connect_budget: Total seconds allowed for connecting all servers;
            defaults to ``DATUS_MCP_CONNECT_BUDGET``.

    Yields:
        Dictionary of connected MCP servers
    """
    connected_servers = {}
    stack = AsyncExitStack()

    if connect_budget is None:
        connect_budget = connect_budget_seconds()
    per_attempt_timeout = connect_timeout_seconds()
    deadline = time.monotonic() + connect_budget

    try:
        if mcp_servers:
            logger.info(
                f"Attempting to connect {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())} "
                f"(budget {connect_budget:.1f}s)"
            )

        for server_name, server in mcp_servers.items():
            if time.monotonic() >= deadline:
                logger.warning(
                    f"Skipping MCP server {server_name}: connect budget of {connect_budget:.1f}s exhausted; "
                    f"the agent runs without it"
                )
                continue

            try:
                logger.info(f"Connecting MCP server: {server_name}")
                cm = _safe_connect_server(
                    server_name,
                    server,
                    connect_timeout=per_attempt_timeout,
                    deadline=deadline,
                )
                connected_server = await stack.enter_async_context(cm)
                connected_servers[server_name] = connected_server
                logger.info(f"Successfully connected MCP server: {server_name}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Failed to start MCP server {server_name}: {str(e)}; continuing without it")

        if mcp_servers and not connected_servers:
            logger.warning("No MCP servers were successfully connected; the agent runs with built-in tools only")

        yield connected_servers

    finally:
        logger.debug("Cleaning up all MCP servers via AsyncExitStack")
        try:
            await stack.aclose()
        except RuntimeError as e:
            if _CANCEL_SCOPE_ERROR in str(e):
                # This is a known anyio issue that can be safely ignored during cleanup
                logger.debug("Suppressed cancel scope error during MCP server cleanup")
            else:
                raise
