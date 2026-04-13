#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""CLI entry point for the Datus Claw IM gateway."""

import argparse
import asyncio
import logging
import os

from datus import __version__
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Datus Claw — IM Channel Gateway")
    parser.add_argument("-v", "--version", action="version", version=f"Datus Claw {__version__}")
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
        help="Default namespace (default: DATUS_NAMESPACE env or 'default')",
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.debug:
        args.log_level = "DEBUG"

    configure_logging(args.debug)
    logging.getLogger().setLevel(getattr(logging, args.log_level, logging.INFO))

    from datus.configuration.agent_config_loader import load_agent_config

    logger.info("Loading agent configuration...")
    agent_config = load_agent_config(
        config=args.config or "",
        namespace=args.namespace,
    )

    channels_config = getattr(agent_config, "channels_config", {})
    if not channels_config:
        logger.error("No 'channels' section found in agent configuration. Nothing to start.")
        raise SystemExit(1)

    from datus.claw.gateway import ClawGateway

    gateway = ClawGateway(
        agent_config=agent_config,
        channels_config=channels_config,
        host=args.host,
        port=args.port,
    )

    logger.info("Starting Datus Claw gateway...")
    asyncio.run(gateway.start())


if __name__ == "__main__":
    main()
