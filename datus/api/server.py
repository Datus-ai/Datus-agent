#!/usr/bin/env python3
"""
Datus Agent FastAPI server startup script.
"""
import argparse

import uvicorn

from datus.utils.loggings import get_logger

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

    args = parser.parse_args()

    logger.info(f"Starting Datus Agent API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Reload: {args.reload}, Log Level: {args.log_level}")

    # Start the server
    uvicorn.run(
        "datus.api.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't work with multiple workers
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
