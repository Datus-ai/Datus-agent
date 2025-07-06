import argparse
import asyncio
import os

import dotenv
import snowflake.connector

from . import server


def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--allow_write",
        required=False,
        default=False,
        action="store_true",
        help="Allow write operations on the database",
    )
    parser.add_argument("--log_dir", required=False, default=None, help="Directory to log to")
    parser.add_argument("--log_level", required=False, default="INFO", help="Logging level")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        dest="prefetch",
        default=False,
        help="Prefetch table descriptions (when enabled, list_tables and describe_table are disabled)",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_false",
        dest="prefetch",
        help="Don't prefetch table descriptions",
    )
    parser.add_argument(
        "--exclude_tools",
        required=False,
        default=[],
        nargs="+",
        help="List of tools to exclude",
    )

    # First, get all the arguments we don't know about
    args, unknown = parser.parse_known_args()

    # Create a dictionary to store our key-value pairs
    connection_args = {}

    # Iterate through unknown args in pairs
    for i in range(0, len(unknown), 2):
        if i + 1 >= len(unknown):
            break

        key = unknown[i]
        value = unknown[i + 1]

        # Make sure it's a keyword argument (starts with --)
        if key.startswith("--"):
            key = key[2:]  # Remove the '--'
            connection_args[key] = value

    # Now we can add the known args to kwargs
    server_args = {
        "allow_write": args.allow_write,
        "log_dir": args.log_dir,
        "log_level": args.log_level,
        "prefetch": args.prefetch,
        "exclude_tools": args.exclude_tools,
    }

    return server_args, connection_args


def main():
    """Main entry point for the package."""

    dotenv.load_dotenv()

    default_connection_args = snowflake.connector.connection.DEFAULT_CONFIGURATION

    connection_args_from_env = {
        k: os.getenv("SNOWFLAKE_" + k.upper())
        for k in default_connection_args
        if os.getenv("SNOWFLAKE_" + k.upper()) is not None
    }

    server_args, connection_args = parse_args()

    connection_args = {**connection_args_from_env, **connection_args}

    assert (
        "database" in connection_args
    ), 'You must provide the account identifier as "--database" argument or "SNOWFLAKE_DATABASE" environment variable.'
    assert (
        "schema" in connection_args
    ), 'You must provide the username as "--schema" argument or "SNOWFLAKE_SCHEMA" environment variable.'

    asyncio.run(
        server.main(
            connection_args=connection_args,
            allow_write=server_args["allow_write"],
            log_dir=server_args["log_dir"],
            prefetch=server_args["prefetch"],
            log_level=server_args["log_level"],
            exclude_tools=server_args["exclude_tools"],
        )
    )


# Optionally expose other important items at package level
__all__ = ["main", "server", "write_detector"]

if __name__ == "__main__":
    main()
