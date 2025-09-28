#!/usr/bin/env python3
"""
Datus-CLI: An AI-powered SQL command-line interface for data engineers.
Main entry point for the CLI application.
"""

import argparse
from pathlib import Path

from datus import __version__
from datus.cli.repl import DatusCLI
from datus.utils.async_utils import setup_windows_policy
from datus.utils.constants import DBType
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Datus: AI-powered SQL command-line interface")
        self._setup_arguments()

    def _setup_arguments(self):
        # Add version argument
        self.parser.add_argument("-v", "--version", action="version", version=f"Datus CLI {__version__}")

        # Database connection settings
        self.parser.add_argument(
            "--db_type",
            dest="db_type",
            choices=[DBType.SQLITE, DBType.SNOWFLAKE, DBType.DUCKDB],
            default=DBType.SQLITE,
            help="Database type to connect to",
        )
        self.parser.add_argument(
            "--db_path", dest="db_path", type=str, help="Path to database file (for SQLite/DuckDB)"
        )

        # Snowflake specific arguments
        # self.parser.add_argument("--sf_account", dest="sf_account", type=str, help="Snowflake account")
        # self.parser.add_argument("--sf_user", dest="sf_user", type=str, help="Snowflake user")
        # self.parser.add_argument("--sf_password", dest="sf_password", type=str, help="Snowflake password")
        # self.parser.add_argument("--sf_warehouse", dest="sf_warehouse", type=str, help="Snowflake warehouse")
        # self.parser.add_argument("--sf_database", dest="sf_database", type=str, help="Snowflake database")
        # self.parser.add_argument("--sf_schema", dest="sf_schema", type=str, help="Snowflake schema")

        # General settings
        self.parser.add_argument(
            "--history_file",
            dest="history_file",
            type=str,
            default=str(Path.home() / ".datus_history"),
            help="Path to history file",
        )
        self.parser.add_argument(
            "--config",
            dest="config",
            type=str,
            help="Path to configuration file (default: conf/agent.yml > ~/.datus/conf/agent.yml)",
        )
        self.parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        self.parser.add_argument("--no_color", dest="no_color", action="store_true", help="Disable colored output")
        self.parser.add_argument(
            "--storage_path", type=str, help="Base path to the storage directory for the agent", default=""
        )

        self.parser.add_argument(
            "--namespace",
            type=str,
            help="Namespace of databases or benchmark",
        )

        self.parser.add_argument("--database", type=str, help="Default database to connect", default="")

        # LLM trace settings
        self.parser.add_argument(
            "--save_llm_trace",
            action="store_true",
            help="Enable saving LLM input/output traces to YAML files",
        )

        # Web interface settings
        self.parser.add_argument(
            "--web",
            action="store_true",
            help="Launch web-based Streamlit chatbot interface",
        )

        self.parser.add_argument(
            "--port",
            type=int,
            default=8501,
            help="Port for web interface (default: 8501)",
        )

        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Host for web interface (default: localhost)",
        )

    def parse_args(self):
        return self.parser.parse_args()


class Application:
    def __init__(self):
        self.arg_parser = ArgumentParser()

    def run(self):
        args = self.arg_parser.parse_args()

        # Configure logging based on debug flag, disable console output
        configure_logging(args.debug, console_output=False)

        # Check if web interface is requested
        if args.web:
            self._run_web_interface(args)
        else:
            # Initialize and run CLI
            cli = DatusCLI(args)
            cli.run()

    def _run_web_interface(self, args):
        """Launch Streamlit web interface"""
        from datus.cli.web_chatbot import run_web_interface

        run_web_interface(args)


def main():
    """Entry point for console scripts"""
    app = Application()
    app.run()


if __name__ == "__main__":
    setup_windows_policy()
    main()
