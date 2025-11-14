# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Time utility functions for the Datus Agent."""

from datetime import datetime
from typing import Optional


def get_default_current_date(current_date: Optional[str]) -> str:
    """Get current_date or default to today's date if not set.

    Args:
        current_date: Optional date string in format 'YYYY-MM-DD'

    Returns:
        The provided current_date or today's date in 'YYYY-MM-DD' format
    """
    if current_date:
        return current_date
    return datetime.now().strftime("%Y-%m-%d")


def format_duration_human(seconds: float) -> str:
    """Format seconds into human readable (h/m/s) format."""
    seconds = int(seconds)

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return "".join(parts)
