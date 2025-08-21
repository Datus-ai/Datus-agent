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
