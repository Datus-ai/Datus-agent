# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Typing compatibility for Python < 3.12
"""

try:
    from typing import override  # Python 3.12+
except ImportError:
    # For Python < 3.12, create a dummy override decorator
    def override(func):
        """Dummy override decorator for Python < 3.12 compatibility."""
        return func


__all__ = ["override"]
