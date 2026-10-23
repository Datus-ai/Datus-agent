# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Errors shared by Agent's Dosi runtime and YAML editor."""


class SemanticCoreException(Exception):
    """Base exception for semantic model and query errors."""

    def __init__(self, message: str, *, execution_info: list[dict] | None = None) -> None:
        super().__init__(message)
        self.execution_info = execution_info
