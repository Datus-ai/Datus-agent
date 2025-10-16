# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Load environment variables from .env file
def try_load_dotenv():
    try:
        load_dotenv()
    except OSError:
        pass


def get_env(key: str, default: Any = None) -> Optional[str]:
    """
    Get environment variable value

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default value
    """
    return os.getenv(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer value from environment variable

    Args:
        key: Environment variable name
        default: Default integer value if not found or invalid

    Returns:
        Integer value from environment variable or default value
    """
    value = get_env(key)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float value from environment variable

    Args:
        key: Environment variable name
        default: Default float value if not found or invalid

    Returns:
        Float value from environment variable or default value
    """
    value = get_env(key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable

    Args:
        key: Environment variable name
        default: Default boolean value if not found

    Returns:
        Boolean value from environment variable or default value
    """
    value = get_env(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "y", "on")


def get_env_list(key: str, default: Optional[list] = None, separator: str = ",") -> list:
    """
    Get list value from environment variable

    Args:
        key: Environment variable name
        default: Default list value if not found
        separator: Separator string to split the value

    Returns:
        List value from environment variable or default value
    """
    value = get_env(key)
    if value is None:
        return default if default is not None else []
    return [item.strip() for item in value.split(separator)]


@lru_cache(maxsize=1)
def load_metricflow_env_settings() -> Optional[Dict[str, Any]]:
    """
    Load MetricFlow environment settings from ~/.datus/metricflow/env_settings.yml

    Returns:
        Dictionary of environment variables if config file exists and is valid, None otherwise
    """
    config_path = Path.home() / ".datus" / "metricflow" / "env_settings.yml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f) or {}
            env_settings = config_data.get("environment_variables", {})
            # Convert all values to strings for environment variable compatibility
            env_settings = {k: str(v) for k, v in env_settings.items()}
            logger.info(f"Loaded env_settings from {config_path}")
            return env_settings
        except Exception as e:
            logger.warning(f"Failed to load env_settings from {config_path}: {e}")
            return None
    return None


def get_metricflow_env(key: str, default: Any = None) -> Optional[str]:
    """
    Get MetricFlow environment variable value, prioritizing ~/.datus/metricflow/env_settings.yml

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value from config file or system environment or default value
    """
    env_settings = load_metricflow_env_settings()
    if env_settings and key in env_settings:
        return env_settings[key]
    return os.getenv(key, default)
