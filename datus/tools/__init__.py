# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool registration module for SQL Agent.

This module automatically registers all available tools from subdirectories.
Each tool directory should contain tool implementations that can be used by the workflow system.
"""

import importlib
import os
from typing import List, Optional

from datus.utils.loggings import get_logger

from .base import BaseTool

logger = get_logger(__name__)


def get_tool_types() -> List[str]:
    """Get all available tool types by scanning the tools directory.

    Returns:
        List of tool type names derived from directory names
    """
    # Get the current directory path
    tools_dir = os.path.dirname(__file__)

    # Scan for tool directories (excluding __pycache__ and files)
    tool_types = [
        d.replace("_tools", "")
        for d in os.listdir(tools_dir)
        if os.path.isdir(os.path.join(tools_dir, d)) and d.endswith("_tools") and not d.startswith("__")
    ]

    return tool_types


def get_tool(tool_type: str, **kwargs) -> Optional[BaseTool]:
    """Get a tool implementation by its type.

    Args:
        tool_type: The type of tool to get

    Returns:
        Tool implementation if found, None otherwise
    """
    try:
        # Convert tool type to directory name
        tool_dir = f"{tool_type}_tools"

        # Import the tool directory's __init__ module
        module = importlib.import_module(f"tools.{tool_dir}")

        # Check if the tool is declared in __all__
        if not hasattr(module, "__all__"):
            return None

        # Find the tool class in __all__
        for tool_name in module.__all__:
            tool_class = getattr(module, tool_name)
            # Return an instance of the first available tool
            return tool_class(**kwargs)
        return None
    except ImportError as e:
        error_msg = f"Failed to import tool module '{tool_dir}': {str(e)}"
        logger.error(error_msg)
        return None
    except AttributeError as e:
        error_msg = f"Tool module '{tool_dir}' has invalid structure: {str(e)}"
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Unexpected error loading tool '{tool_type}' from '{tool_dir}': {str(e)}"
        logger.error(error_msg)
        return None


__all__ = ["get_tool_types", "get_tool", "BaseTool"]
