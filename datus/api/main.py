#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Datus Agent FastAPI server startup script.

Single entry point for both the ``datus-api`` console script and
``python -m datus.api.main``. Supports foreground and daemon
(background) modes with start/stop/restart/status actions.
"""

import argparse
import atexit
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import uvicorn

from datus import __version__
from datus.configuration.agent_config_loader import parse_config_path
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def _default_paths(config_path: str = "") -> Tuple[Path, Path]:
    """Return default pid and log file paths."""
    from datus.configuration.agent_config_loader import get_agent_home
    from datus.utils.path_manager import DatusPathManager

    path_manager = DatusPathManager(get_agent_home(config_path))
    pid_file = path_manager.pid_file_path("datus-agent-api")
    log_file = path_manager.logs_dir / "datus-agent-api.log"
    return pid_file, log_file


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        pid_text = pid_file.read_text().strip()
        return int(pid_text) if pid_text else None
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


def _daemon_worker(args: argparse.Namespace, agent_args: argparse.Namespace, pid_file: Path, log_file: Path) -> None:
    """Worker function that runs in the daemon process."""
    os.setsid()
    os.umask(0)

    configure_logging(args.debug, log_dir=str(log_file.parent), console_output=False)
    _redirect_stdio(log_file)
    _write_pid_file(pid_file, os.getpid())

    def _cleanup(*_args):
        _remove_pid_file(pid_file)

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_a: (_cleanup(), os._exit(0)))

    _run_server(args, agent_args)


def _stop(pid_file: Path, timeout_seconds: float = 10.0) -> int:
    pid = _read_pid(pid_file)
    if not pid:
        logger.info("No PID file found; server not running?")
        return 0
    if not _is_process_running(pid):
        logger.info("Stale PID file found; process not running. Cleaning up.")
        _remove_pid_file(pid_file)
        return 0

    logger.info(f"Stopping server (pid={pid}) ...")
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


def _build_agent_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        namespace=args.namespace,
        config=args.config,
        max_steps=args.max_steps,
        workflow=args.workflow,
        load_cp=args.load_cp,
        debug=args.debug,
        source=args.source,
        interactive=args.interactive,
        output_dir=args.output_dir,
        log_level=args.log_level,
        stream_thinking=args.stream_thinking,
    )


def _run_server(args: argparse.Namespace, agent_args: argparse.Namespace) -> None:
    from datus.api.service import create_app

    # reload / multi-worker modes require a string import target, not an app instance.
    if args.reload:
        uvicorn.run(
            "datus.api.service:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level.lower(),
            access_log=True,
        )
        return

    if args.workers and args.workers > 1:
        uvicorn.run(
            "datus.api.service:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level.lower(),
            access_log=True,
        )
        return

    app = create_app(agent_args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level.lower(),
        access_log=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Datus Agent FastAPI Server")
    parser.add_argument("-v", "--version", action="version", version=f"Datus API {__version__}")

    # Server
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Agent configuration file (default: ./conf/agent.yml > ~/.datus/conf/agent.yml)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.getenv("DATUS_NAMESPACE", "default"),
        help="Namespace of databases or benchmark (default: DATUS_NAMESPACE env or 'default')",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=os.getenv("DATUS_OUTPUT_DIR", "./output"),
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default=os.getenv("DATUS_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO or DATUS_LOG_LEVEL env var)",
    )

    # Agent
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum workflow steps")
    parser.add_argument("--workflow", type=str, default="fixed", help="Workflow plan type")
    parser.add_argument("--load_cp", type=str, help="Load workflow from checkpoint file")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["web", "vscode"],
        help=(
            "Default proxy tool source shortcut. 'vscode' -> proxy filesystem_tools.*, "
            "'web' -> proxy write_file/edit_file. "
            "Overridable per request via ChatInput.source."
        ),
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=True,
        help="Enable ask_user interactive tool by default (default: enabled)",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="Disable ask_user (workflow / non-interactive mode)",
    )
    parser.add_argument(
        "--stream",
        dest="stream_thinking",
        action="store_true",
        default=False,
        help="Enable streaming thinking deltas (real-time token-by-token SSE output)",
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
        help="PID file path (default: ~/.datus/run/datus-agent-api.pid)",
    )
    parser.add_argument(
        "--daemon-log-file",
        type=str,
        help="Daemon log file path (default: ~/.datus/logs/datus-agent-api.log)",
    )
    return parser


def main():
    """Main entry point for starting the Datus Agent API server."""
    if hasattr(multiprocessing, "set_start_method"):
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    parser = _build_parser()
    args = parser.parse_args()

    # --debug is a shortcut for --log-level DEBUG; unify early so all
    # downstream consumers (uvicorn, configure_logging, agent_args) agree.
    if args.debug:
        args.log_level = "DEBUG"

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
        # fall-through to start

    # Resolve config file path and export env vars for lifespan / module-level app.
    try:
        config_path = str(parse_config_path(args.config or ""))
    except Exception as e:
        logger.error(f"Failed to locate configuration file: {e}")
        raise SystemExit(1) from e
    args.config = config_path

    os.environ["DATUS_CONFIG"] = config_path
    os.environ["DATUS_NAMESPACE"] = args.namespace
    os.environ["DATUS_OUTPUT_DIR"] = args.output_dir
    os.environ["DATUS_LOG_LEVEL"] = args.log_level

    if args.daemon and args.reload:
        print("--daemon mode is mutually exclusive with --reload. Remove --reload.", file=sys.stderr)
        raise SystemExit(2)

    if args.daemon:
        if (pid := _read_pid(pid_file)) and _is_process_running(pid):
            print(f"Already running (pid={pid})", file=sys.stderr)
            raise SystemExit(0)

        configure_logging(args.debug, log_dir=str(log_file.parent), console_output=False)
        logger.info(
            f"Starting Datus Agent API server (daemon) on {args.host}:{args.port} | "
            f"Workers: {args.workers}, Debug: {args.debug}"
        )

        agent_args = _build_agent_args(args)
        daemon_process = multiprocessing.Process(
            target=_daemon_worker, args=(args, agent_args, pid_file, log_file), daemon=False
        )
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

    logger.info(f"Starting Datus Agent API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}, Debug: {args.debug}")
    logger.info(f"Agent config - Namespace: {args.namespace}, Config: {args.config}")
    agent_args = _build_agent_args(args)
    _run_server(args, agent_args)


if __name__ == "__main__":
    main()
