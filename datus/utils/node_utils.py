# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared utility functions for agentic nodes."""

from typing import Optional


def build_database_context(
    db_type: str,
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> str:
    """Build a formatted database context string for LLM prompts.

    Args:
        db_type: Database dialect type (e.g. "sqlite", "postgresql").
        catalog: Optional catalog name.
        database: Optional database name.
        schema: Optional schema name.

    Returns:
        A markdown-formatted context string.
    """
    context_parts = [f"**Dialect**: {db_type}"]
    if catalog:
        context_parts.append(f"**Catalog**: {catalog}")
    if database:
        context_parts.append(f"**Database**: {database}")
    if schema:
        context_parts.append(f"**Schema**: {schema}")
    return f"Database Context: \n\n{', '.join(context_parts)}"
