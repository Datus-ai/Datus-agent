#!/usr/bin/env python3
"""
Datus Agent FastAPI server startup script.
"""
import argparse

import uvicorn

from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for starting the Datus Agent API server."""
    parser = argparse.ArgumentParser(description="Start Datus Agent FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Namespace of databases or benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: conf/agent.yml > ~/.datus/conf/agent.yml)",
    )
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum workflow steps")
    parser.add_argument("--plan", type=str, default="fixed", help="Workflow plan type")
    parser.add_argument("--load_cp", type=str, help="Load workflow from checkpoint file")

    args = parser.parse_args()

    logger.info(f"Starting Datus Agent API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}, Log Level: {args.log_level}")
    logger.info(f"Agent config - Namespace: {args.namespace}, Config: {args.config}")

    configure_logging(args.log_level)

    # Create agent args from command line args
    agent_args = argparse.Namespace(
        namespace=args.namespace,
        config=args.config,
        max_steps=args.max_steps,
        plan=args.plan,
        load_cp=args.load_cp,
    )

    # Create app with args
    from datus.api.service import create_app

    app = create_app(agent_args)

    # Start the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
