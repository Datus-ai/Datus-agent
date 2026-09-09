# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Shared logging configuration, independent of application initialization."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def normalize_log_level(value: str) -> str:
    level = str(value).strip().upper()
    if level not in LOG_LEVELS:
        raise ValueError(f"Invalid logging level {value!r}; expected one of {', '.join(LOG_LEVELS)}")
    return level


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    redact: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> LoggingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("agent.logging must be a mapping")
        redact = raw.get("redact", {})
        if not isinstance(redact, Mapping):
            raise ValueError("agent.logging.redact must be a mapping")
        return cls(level=normalize_log_level(raw.get("level", "INFO")), redact=dict(redact))


def resolve_log_level(
    *, level: str | None = None, debug: bool = False, config: LoggingConfig | None = None
) -> tuple[str, str]:
    """Return the effective level and its source; false debug is not an override."""
    if debug and level is not None:
        raise ValueError("--debug and --log-level are mutually exclusive")
    if debug:
        return "DEBUG", "cli:--debug"
    if level is not None:
        return normalize_log_level(level), "cli:--log-level"
    if "DATUS_LOG_LEVEL" in os.environ:
        return normalize_log_level(os.environ["DATUS_LOG_LEVEL"]), "env:DATUS_LOG_LEVEL"
    if config is not None:
        return config.level, "config:agent.logging.level"
    return "INFO", "default"


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    # SUPPRESS preserves options supplied before a subcommand using this parent.
    group.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Alias for --log-level DEBUG")
    group.add_argument(
        "--log-level",
        type=normalize_log_level,
        choices=LOG_LEVELS,
        default=argparse.SUPPRESS,
        help="Logging level (CLI > DATUS_LOG_LEVEL > agent.logging.level > INFO)",
    )


def resolve_logging_arguments(args: argparse.Namespace, agent_config: Any = None) -> LoggingConfig:
    """Resolve once at process startup, without constructing an Agent or storage."""
    if getattr(args, "_logging_resolved", False):
        return LoggingConfig(level=args.log_level, redact=getattr(args, "log_redact", {}))
    config = getattr(agent_config, "logging", None)
    if config is None:
        import yaml
        from dotenv import load_dotenv

        from datus.configuration.agent_config_loader import parse_config_path
        from datus.utils.exceptions import DatusException, ErrorCode

        load_dotenv()
        try:
            path = parse_config_path(getattr(args, "config", None) or os.getenv("DATUS_CONFIG", ""))
        except DatusException as exc:
            if exc.code != ErrorCode.COMMON_FILE_NOT_FOUND:
                raise
            # First-run CLI setup and status/stop commands need no agent.yml.
        else:
            with path.open(encoding="utf-8") as stream:
                raw = yaml.safe_load(stream) or {}
            logging_raw = (raw.get("agent") or {}).get("logging")
            if logging_raw is not None:
                from datus.configuration.agent_config import _resolve_nested_value

                config = LoggingConfig.from_dict(_resolve_nested_value(logging_raw))
    effective, source = resolve_log_level(
        level=getattr(args, "log_level", None), debug=getattr(args, "debug", False), config=config
    )
    args.log_level = effective
    args.debug = effective == "DEBUG"
    args.log_redact = config.redact if config else {}
    args.log_level_source = source
    args._logging_resolved = True
    return LoggingConfig(level=effective, redact=args.log_redact)
