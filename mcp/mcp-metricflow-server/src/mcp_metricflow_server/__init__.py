"""MCP MetricFlow Server package."""

import asyncio
import logging

__version__ = "0.1.0"

from .main import main as async_main

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the package."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


__all__ = ["main"]


if __name__ == "__main__":
    main()
