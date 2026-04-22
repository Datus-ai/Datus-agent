#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
CLI entry point for the Datus Gateway IM gateway.

Supports foreground and daemon (background) modes with
start/stop/restart/status actions.
"""

import argparse
import asyncio
import atexit
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from datus import __version__
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)

_SERVICE_NAME = "datus-gateway"


def _default_paths(config_path: str = "") -> Tuple[Path, Path]:
    """Return default pid and log file paths."""
    from datus.configuration.agent_config_loader import get_agent_home
    from datus.utils.path_manager import DatusPathManager

    path_manager = DatusPathManager(get_agent_home(config_path))
    pid_file = path_manager.pid_file_path(_SERVICE_NAME)
    log_file = path_manager.logs_dir / f"{_SERVICE_NAME}.log"
    return pid_file, log_file


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid_text = pid_file.read_text().strip()
        if not pid_text:
            return None
        pid = int(pid_text)
        return pid if pid > 0 else None
    except Exception:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _write_pid_file(pid_file: Path, pid: int) -> None:
    _ensure_parent_dir(pid_file)
    pid_file.write_text(str(pid))


def _remove_pid_file(pid_file: Path) -> None:
    try:
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass


def _redirect_stdio(log_file: Path) -> None:
    _ensure_parent_dir(log_file)
    with open(os.devnull, "rb", buffering=0) as si:
        os.dup2(si.fileno(), sys.stdin.fileno())
    with open(log_file, "ab", buffering=0) as so:
        os.dup2(so.fileno(), sys.stdout.fileno())
    with open(log_file, "ab", buffering=0) as se:
        os.dup2(se.fileno(), sys.stderr.fileno())


def _run_gateway(args: argparse.Namespace) -> None:
    """Load config and start the Datus Gateway (blocking)."""
    from datus.configuration.agent_config_loader import load_agent_config

    logger.info("Loading agent configuration...")
    agent_config = load_agent_config(
        config=args.config or "",
        datasource=args.datasource,
    )
    # Gateway runs non-interactively — no broker to confirm out-of-workspace
    # file access. Force filesystem strict mode so nodes reject EXTERNAL
    # paths instead of hanging on a prompt.
    agent_config.filesystem_strict = True

    try:
        am = agent_config.active_model()
        logger.info("Active model: %s/%s", am.type, am.model)
    except Exception:
        logger.warning(
            "No active LLM model configured. Incoming messages will receive error responses "
            "until a model is set up via 'datus' CLI + /model command."
        )

    channels_config = getattr(agent_config, "channels_config", {})
    if not channels_config:
        logger.error("No 'channels' section found in agent configuration. Nothing to start.")
        raise SystemExit(1)

    from datus.gateway.runtime import DatusGateway

    gateway = DatusGateway(
        agent_config=agent_config,
        channels_config=channels_config,
        host=args.host,
        port=args.port,
    )

    logger.info("Starting Datus Gateway...")
    asyncio.run(gateway.start())


def _daemon_worker(args: argparse.Namespace, pid_file: Path, log_file: Path) -> None:
    """Worker function that runs in the daemon process."""
    if sys.platform != "win32":
        os.setsid()
        os.umask(0o022)

    log_dir = str(log_file.parent)
    configure_logging(args.debug, log_dir=log_dir, console_output=False)
    logging.getLogger().setLevel(getattr(logging, args.log_level, logging.INFO))
    _redirect_stdio(log_file)
    _write_pid_file(pid_file, os.getpid())

    def _cleanup(*_args):
        _remove_pid_file(pid_file)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_a: (_cleanup(), os._exit(0)))

    _run_gateway(args)


def _stop(pid_file: Path, timeout_seconds: float = 10.0) -> int:
    pid = _read_pid(pid_file)
    if not pid:
        logger.info("No PID file found; gateway not running?")
        return 0
    if not _is_process_running(pid):
        logger.info("Stale PID file found; process not running. Cleaning up.")
        _remove_pid_file(pid_file)
        return 0

    logger.info(f"Stopping gateway (pid={pid}) ...")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_pid_file(pid_file)
        return 0

    start = time.time()
    while time.time() - start < timeout_seconds:
        if not _is_process_running(pid):
            _remove_pid_file(pid_file)
            logger.info("Stopped.")
            return 0
        time.sleep(0.2)

    try:
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_pid_file(pid_file)
    logger.info("Force killed.")
    return 0


def _status(pid_file: Path) -> int:
    pid = _read_pid(pid_file)
    if pid and _is_process_running(pid):
        print(f"running (pid={pid})")
        return 0
    print("stopped")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Datus Gateway — IM Channel Gateway")
    parser.add_argument("-v", "--version", action="version", version=f"Datus Gateway {__version__}")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Agent configuration file (default: ./conf/agent.yml > ~/.datus/conf/agent.yml)",
    )
    parser.add_argument(
        "--datasource",
        type=str,
        default=os.getenv("DATUS_DATASOURCE", "default"),
        help="Default datasource (default: DATUS_DATASOURCE env or 'default')",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Health-check bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Health-check bind port (default: 9000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default=os.getenv("DATUS_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO or DATUS_LOG_LEVEL env var)",
    )

    # Daemon control
    parser.add_argument(
        "--action",
        choices=["start", "stop", "restart", "status"],
        default="start",
        help="Daemon action (default: start)",
    )
    parser.add_argument("--daemon", action="store_true", help="Run in background as a daemon")
    parser.add_argument(
        "--pid-file",
        type=str,
        help=f"PID file path (default: ~/.datus/run/{_SERVICE_NAME}.pid)",
    )
    parser.add_argument(
        "--daemon-log-file",
        type=str,
        help=f"Daemon log file path (default: ~/.datus/logs/{_SERVICE_NAME}.log)",
    )

    # Optional subcommands (backwards-compatible: no subcommand = daemon flow).
    subparsers = parser.add_subparsers(dest="subcommand")
    configure_parser = subparsers.add_parser(
        "configure",
        help="Manage Gateway IM channels interactively",
    )
    configure_parser.add_argument(
        "--config",
        dest="configure_config",
        type=str,
        default=None,
        help="Override agent configuration file for this command",
    )

    return parser


def main() -> None:
    """Main entry point for starting the Datus Gateway."""
    if hasattr(multiprocessing, "set_start_method"):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    parser = _build_parser()
    args = parser.parse_args()

    if args.debug:
        args.log_level = "DEBUG"

    # Subcommand dispatch (does not touch daemon/pid paths).
    if getattr(args, "subcommand", None) == "configure":
        from datus.gateway.configure import ChannelConfigurator

        configure_logging(args.debug)
        config_path = getattr(args, "configure_config", None) or args.config or ""
        raise SystemExit(ChannelConfigurator(config_path).run())

    # Resolve defaults for pid/log
    default_pid, default_log = _default_paths(args.config or "")
    pid_file = Path(args.pid_file) if args.pid_file else default_pid
    log_file = Path(args.daemon_log_file) if args.daemon_log_file else default_log

    if args.action in {"status", "stop"}:
        configure_logging(args.debug)
        if args.action == "status":
            raise SystemExit(_status(pid_file))
        if args.action == "stop":
            raise SystemExit(_stop(pid_file))

    if args.action == "restart":
        configure_logging(args.debug)
        _stop(pid_file)
        args.daemon = True
        # fall-through to start as daemon

    if args.daemon:
        if (pid := _read_pid(pid_file)) and _is_process_running(pid):
            print(f"Already running (pid={pid})", file=sys.stderr)
            raise SystemExit(0)

        log_dir = str(log_file.parent)
        configure_logging(args.debug, log_dir=log_dir, console_output=False)
        logger.info(f"Starting Datus Gateway (daemon) on {args.host}:{args.port} | Debug: {args.debug}")

        daemon_process = multiprocessing.Process(target=_daemon_worker, args=(args, pid_file, log_file), daemon=False)
        daemon_process.start()

        time.sleep(0.5)

        if daemon_process.is_alive():
            print(f"Daemon started successfully (pid={daemon_process.pid})")
            os._exit(0)
        else:
            print("Failed to start daemon process", file=sys.stderr)
            daemon_process.join()
            os._exit(1)
    else:
        configure_logging(args.debug)

    logging.getLogger().setLevel(getattr(logging, args.log_level, logging.INFO))

    logger.info(f"Starting Datus Gateway on {args.host}:{args.port}")
    _run_gateway(args)


if __name__ == "__main__":
    main()
