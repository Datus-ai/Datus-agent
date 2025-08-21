"""Time utility functions for the Datus Agent."""

from datetime import datetime
from typing import Optional


def get_default_current_time(current_time: Optional[str]) -> str:
    """Get current_time or default to today's date if not set.

    Args:
        current_time: Optional time string in format 'YYYY-MM-DD'

    Returns:
        The provided current_time or today's date in 'YYYY-MM-DD' format
    """
    if current_time:
        return current_time
    return datetime.now().strftime("%Y-%m-%d")
