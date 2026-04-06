#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Datus API Server: FastAPI-based REST API for Datus Agent.
Command-line entry point for starting the API service.
"""

import argparse
import os
import sys

import uvicorn

from datus import __version__
from datus.configuration.agent_config_loader import parse_config_path
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class APIServerArgumentParser:
    """Argument parser for Datus API Server."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Datus API Server: FastAPI-based REST API for Datus Agent")
        self._setup_arguments()

    def _setup_arguments(self):
        """Setup command-line arguments."""
        # Version
        self.parser.add_argument("-v", "--version", action="version", version=f"Datus API {__version__}")

        # Configuration
        self.parser.add_argument(
            "--config",
            dest="config",
            type=str,
            default=None,
            help="Agent configuration file (default: ./conf/agent.yml > ~/.datus/conf/agent.yml)",
        )
        self.parser.add_argument(
            "--namespace",
            dest="namespace",
            type=str,
            default=os.getenv("DATUS_NAMESPACE", "default"),
            help="Namespace to use (default: default or DATUS_NAMESPACE env var)",
        )

        # Output and logging
        self.parser.add_argument(
            "--output-dir",
            dest="output_dir",
            type=str,
            default=os.getenv("DATUS_OUTPUT_DIR", "./output"),
            help="Output directory for results (default: ./output)",
        )
        self.parser.add_argument(
            "--log-level",
            dest="log_level",
            type=str,
            default=os.getenv("DATUS_LOG_LEVEL", "INFO"),
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Log level (default: INFO or DATUS_LOG_LEVEL env var)",
        )

        # Server settings
        self.parser.add_argument(
            "--host",
            dest="host",
            type=str,
            default="127.0.0.1",
            help="Server host (default: 127.0.0.1)",
        )
        self.parser.add_argument(
            "--port",
            dest="port",
            type=int,
            default=8000,
            help="Server port (default: 8000)",
        )
        self.parser.add_argument(
            "--reload",
            dest="reload",
            action="store_true",
            help="Enable auto-reload on file changes (development mode)",
        )
        self.parser.add_argument(
            "--workers",
            dest="workers",
            type=int,
            default=1,
            help="Number of worker processes (default: 1)",
        )

    def parse_args(self):
        """Parse and return command-line arguments."""
        return self.parser.parse_args()


def main():
    """Main entry point for Datus API Server."""
    parser = APIServerArgumentParser()
    args = parser.parse_args()

    # Parse config file path (priority: explicit > ./conf/agent.yml > ~/.datus/conf/agent.yml)
    try:
        config_path = str(parse_config_path(args.config or ""))
    except Exception as e:
        logger.error(f"Failed to locate configuration file: {e}")
        sys.exit(1)

    logger.info("Starting Datus API Server")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Namespace: {args.namespace}")
    logger.info(f"  Server: {args.host}:{args.port}")

    # Set environment variables for lifespan to use
    os.environ["DATUS_CONFIG"] = config_path
    os.environ["DATUS_NAMESPACE"] = args.namespace
    os.environ["DATUS_OUTPUT_DIR"] = args.output_dir
    os.environ["DATUS_LOG_LEVEL"] = args.log_level

    # Create app with CLI args (don't import the module-level app)
    from datus.api.service import create_app

    cli_args = argparse.Namespace(
        config=config_path,
        namespace=args.namespace,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )
    app = create_app(cli_args)

    # Run uvicorn
    if args.reload and args.workers > 1:
        logger.warning("--reload is incompatible with --workers > 1; using single worker process")
        args.workers = 1

    if args.reload:
        # reload requires an import string, not an app instance
        uvicorn.run(
            "datus.api.service:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            reload=True,
        )
    elif args.workers > 1:
        # multi-worker mode also requires an import string
        uvicorn.run(
            "datus.api.service:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            workers=args.workers,
        )
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
        )


if __name__ == "__main__":
    main()
