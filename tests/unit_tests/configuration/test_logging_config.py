# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

import argparse
import logging
from pathlib import Path

import pytest

from datus.configuration.logging_config import LoggingConfig, add_logging_arguments, resolve_logging_arguments
from datus.utils.loggings import configure_entrypoint_logging, configure_logging, get_logger

pytestmark = pytest.mark.usefixtures("isolated_logging")


@pytest.mark.parametrize(
    "yaml_level,env_level,flags,expected,source",
    [
        (None, None, [], "INFO", "default"),
        ("debug", None, [], "DEBUG", "config:agent.logging.level"),
        ("DEBUG", "warning", [], "WARNING", "env:DATUS_LOG_LEVEL"),
        ("DEBUG", "WARNING", ["--log-level", "error"], "ERROR", "cli:--log-level"),
        ("INFO", "ERROR", ["--debug"], "DEBUG", "cli:--debug"),
    ],
)
def test_configuration_priority(tmp_path, monkeypatch, yaml_level, env_level, flags, expected, source):
    config = tmp_path / "agent.yml"
    config.write_text(f"agent:\n  logging:\n    level: {yaml_level}\n" if yaml_level else "agent: {}\n")
    if env_level is None:
        monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("DATUS_LOG_LEVEL", env_level)
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    args = parser.parse_args(flags)
    args.config = str(config)
    result = resolve_logging_arguments(args)
    assert result.level == expected
    assert args.log_level_source == source
    assert args.debug == (expected == "DEBUG")
    assert resolve_logging_arguments(args) == result  # resolved debug never becomes a second override


def test_conflicting_and_invalid_levels():
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--debug", "--log-level", "INFO"])
    with pytest.raises(ValueError, match="Invalid logging level"):
        LoggingConfig.from_dict({"level": "deubg"})


@pytest.mark.parametrize("flag", [["--debug"], ["--log-level", "WARNING"]])
def test_agent_global_logging_options_survive_subcommand(flag, tmp_path, monkeypatch):
    from datus.main import create_parser

    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    config = tmp_path / "agent.yml"
    config.write_text("agent: {}")
    before = create_parser().parse_args([*flag, "service", "list"])
    after = create_parser().parse_args(["service", "list", *flag])
    for args in (before, after):
        args.config = str(config)
        resolve_logging_arguments(args)
    assert before.log_level == after.log_level


@pytest.mark.parametrize("entrypoint", ["api", "gateway"])
def test_service_yaml_level_reaches_business_and_service_loggers(entrypoint, tmp_path, monkeypatch):
    from datus.api.main import _build_parser as api_parser
    from datus.gateway.main import _build_parser as gateway_parser

    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    config = tmp_path / "agent.yml"
    config.write_text("agent:\n  logging:\n    level: ERROR\n")
    args = (api_parser if entrypoint == "api" else gateway_parser)().parse_args(["--config", str(config)])
    manager = configure_entrypoint_logging(args, log_dir=tmp_path / "logs", console_output=False)
    for name in ("datus.agent", "uvicorn.error", "LiteLLM", "web_chatbot"):
        target = logging.getLogger(name)
        # Explicit child levels cannot bypass the file handler's global floor.
        target.info("must-not-appear")
    get_logger("datus.agent").error("expected-error", request_id="raw-provider-id")
    manager.file_handler.flush()
    (record,) = Path(manager.file_handler.baseFilename).read_text().splitlines()
    assert "expected-error" in record
    assert "request_id=raw-provider-id" in record
    assert "exception" not in record
    assert "\x1b[" not in record
    assert args.debug is False
    assert args.log_level == "ERROR"


def test_api_explicit_debug_enables_agent_debug(tmp_path, monkeypatch):
    from datus.api.main import _build_parser

    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    args = _build_parser().parse_args(["--log-level", "DEBUG"])
    manager = configure_entrypoint_logging(args, log_dir=tmp_path, console_output=False)
    get_logger("datus.agent").debug("agent-debug-visible")
    manager.file_handler.flush()
    assert "agent-debug-visible" in Path(manager.file_handler.baseFilename).read_text()
    assert logging.getLogger("uvicorn.error").level == logging.DEBUG


def test_text_logs_preserve_fields_and_redact_without_tracing(tmp_path):
    manager = configure_logging(level="DEBUG", log_dir=tmp_path, console_output=False)
    get_logger("datus.test").debug("safe-event", configuration={"api_key": "private-key", "count": 2})
    try:
        raise ValueError("failure")
    except ValueError:
        get_logger("datus.test").error("no-traceback", exc_info=False)
        get_logger("datus.test").exception("one-traceback")
    manager.file_handler.flush()
    output = Path(manager.file_handler.baseFilename).read_text()
    event = next(line for line in output.splitlines() if "safe-event" in line)
    assert "configuration={'api_key': '[REDACTED]', 'count': 2}" in event
    assert "private-key" not in output
    assert "\x1b[" not in output
    before_exception, traceback_output = output.split("one-traceback", 1)
    assert "no-traceback" in before_exception
    assert "Traceback (most recent call last):" not in before_exception
    assert traceback_output.count("Traceback (most recent call last):") == 1
    assert "ValueError: failure" in traceback_output


def test_api_in_process_initialization_preserves_host_log_destination(tmp_path, monkeypatch):
    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    config = tmp_path / "agent.yml"
    config.write_text("agent:\n  logging:\n    level: WARNING\n")
    args = argparse.Namespace(config=str(config))
    original = configure_entrypoint_logging(args, log_dir=tmp_path / "daemon", console_output=False)
    inherited = configure_entrypoint_logging(args, if_unconfigured=True)
    assert inherited is original
    assert logging.getLogger().handlers == [original.file_handler]
    assert logging.getLogger().level == logging.WARNING


def test_yaml_logging_interpolates_environment_values(tmp_path, monkeypatch):
    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    monkeypatch.setenv("TEST_PROCESS_LOG_LEVEL", "ERROR")
    config = tmp_path / "agent.yml"
    config.write_text("agent:\n  logging:\n    level: ${TEST_PROCESS_LOG_LEVEL}\n")
    assert resolve_logging_arguments(argparse.Namespace(config=str(config))).level == "ERROR"


def test_spawned_worker_resolves_logging_before_agent_construction(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys

    monkeypatch.setenv("DATUS_LOG_LEVEL", "ERROR")
    config = tmp_path / "agent.yml"
    config.write_text("agent:\n  logging:\n    level: DEBUG\n")
    script = """
import argparse, logging, sys
from datus.utils.loggings import configure_entrypoint_logging, get_logger
args = argparse.Namespace(config=sys.argv[1])
manager = configure_entrypoint_logging(args, if_unconfigured=True, log_dir=sys.argv[2], console_output=False)
get_logger("datus.agent").info("hidden-business-log")
get_logger("datus.agent").error("visible-business-log")
logging.getLogger("uvicorn.error").warning("hidden-service-log")
logging.getLogger("uvicorn.error").error("visible-service-log")
manager.file_handler.flush()
print(args.log_level)
"""
    checkout = Path(__file__).resolve().parents[3]
    child = subprocess.run(
        [sys.executable, "-c", script, str(config), str(tmp_path / "worker")],
        cwd=checkout,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(checkout),
            "DATUS_LOG_LEVEL": "ERROR",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "ERROR"
    records = [line for path in (tmp_path / "worker").glob("agent.*.log") for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert "visible-business-log" in records[0]
    assert "visible-service-log" in records[1]
    assert all("hidden-" not in line for line in records)


def test_uvicorn_startup_keeps_service_logs_in_shared_text_file(tmp_path, monkeypatch):
    from datus.api.main import _run_server

    args = argparse.Namespace(reload=False, workers=1, host="127.0.0.1", port=8000, log_level="ERROR")
    manager = configure_logging(level="ERROR", log_dir=tmp_path, console_output=False)
    from importlib import import_module

    monkeypatch.setattr(import_module("datus.api.service"), "create_app", lambda _: object())

    async def serve(self):
        logging.getLogger("uvicorn.error").error("service-error")
        get_logger("datus.agent").error("agent-error")

    monkeypatch.setattr("uvicorn.Server.serve", serve)
    _run_server(args, argparse.Namespace())
    manager.file_handler.flush()
    service_record, agent_record = Path(manager.file_handler.baseFilename).read_text().splitlines()
    assert "service-error" in service_record
    assert "[uvicorn.error]" in service_record
    assert "agent-error" in agent_record
    assert "[datus.agent]" in agent_record


@pytest.mark.parametrize("raw", ["debug", {"level": "verbose"}, {"redact": "all"}])
@pytest.mark.parametrize("entrypoint", ["logging", "agent"])
def test_invalid_yaml_logging_reports_configuration_error(tmp_path, monkeypatch, raw, entrypoint):
    import yaml

    from datus.configuration.agent_config_loader import load_agent_config
    from datus.utils.exceptions import DatusException, ErrorCode

    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    config = tmp_path / "agent.yml"
    config.write_text(yaml.safe_dump({"agent": {"home": str(tmp_path), "logging": raw}}))
    with pytest.raises(DatusException) as error:
        if entrypoint == "logging":
            resolve_logging_arguments(argparse.Namespace(config=str(config)))
        else:
            load_agent_config(config=str(config), reload=True)
    assert error.value.code == ErrorCode.COMMON_CONFIG_ERROR
    assert isinstance(error.value.__cause__, ValueError)


def test_windows_logging_without_colorama(tmp_path, monkeypatch):
    import structlog.dev

    monkeypatch.setattr("datus.utils.loggings.sys.platform", "win32")
    monkeypatch.setattr(structlog.dev, "_IS_WINDOWS", True)
    monkeypatch.setattr(structlog.dev, "colorama", None)
    manager = configure_logging(level="INFO", log_dir=tmp_path, console_output=True)
    record = logging.LogRecord("datus.test", logging.INFO, __file__, 1, "plain-windows-log", (), None)
    for handler in (manager.file_handler, manager.console_handler):
        output = handler.format(record)
        assert "plain-windows-log" in output
        assert "\x1b[" not in output


def test_web_logging_uses_configured_home(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from datus.cli.web.chatbot import run_web_interface
    from datus.utils import loggings, path_manager

    home = tmp_path / "custom-home"
    config = tmp_path / "agent.yml"
    config.write_text(f"agent:\n  home: {home}\n  logging:\n    level: INFO\n")
    monkeypatch.delenv("DATUS_LOG_LEVEL", raising=False)
    # Keep the real path-manager and logging setup while avoiding an HTTP server.
    token = path_manager.set_current_path_manager(str(tmp_path / "old-home"))
    monkeypatch.setattr("datus.cli.web.chatbot.create_web_app", lambda args: object())
    monkeypatch.setattr("datus.cli.web.chatbot._schedule_browser_open", lambda url: None)
    monkeypatch.setattr("uvicorn.Server.serve", AsyncMock())
    try:
        run_web_interface(argparse.Namespace(config=str(config), datasource="test"))
        get_logger("datus.web").info("configured-home-log")
        handler = loggings.get_log_manager().file_handler
        handler.flush()
        assert Path(handler.baseFilename).parent == home / "logs"
        assert "configured-home-log" in Path(handler.baseFilename).read_text()
        assert not (tmp_path / "old-home" / "logs").exists()
    finally:
        path_manager.reset_path_manager(token)


def test_markdown_rendering_keeps_agent_debug_without_parser_noise(tmp_path):
    from io import StringIO

    from rich.console import Console
    from rich.markdown import Markdown

    manager = configure_logging(level="DEBUG", log_dir=tmp_path, console_output=False)
    rendered = StringIO()
    Console(file=rendered, color_system=None).print(Markdown("# Heading\n\nVisible **paragraph**."))
    get_logger("datus.test").debug("agent-debug-preserved")
    parser_logger = logging.getLogger("markdown_it.rules_block.paragraph")
    parser_logger.info("parser-info-hidden")
    parser_logger.warning("parser-warning-preserved")
    parser_logger.error("parser-error-preserved")
    manager.file_handler.flush()

    assert "Visible paragraph." in rendered.getvalue()
    output = Path(manager.file_handler.baseFilename).read_text()
    assert "agent-debug-preserved" in output
    parser_records = [line for line in output.splitlines() if "[markdown_it." in line]
    assert len(parser_records) == 2
    assert "parser-warning-preserved" in parser_records[0]
    assert "parser-error-preserved" in parser_records[1]
