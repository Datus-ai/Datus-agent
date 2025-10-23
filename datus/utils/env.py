# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer value from environment variable

    Args:
        key: Environment variable name
        default: Default integer value if not found or invalid

    Returns:
        Integer value from environment variable or default value
    """
    value = os.getenv(key)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
