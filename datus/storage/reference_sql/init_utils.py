# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import hashlib
from typing import Set

from datus.storage.reference_sql.store import ReferenceSqlRAG


def gen_reference_sql_id(sql: str, comment: str) -> str:
    """Generate MD5 hash ID from SQL and comment fields."""
    combined_text = f"{sql}{comment}"
    return hashlib.md5(combined_text.encode("utf-8")).hexdigest()


def exists_reference_sql(storage: ReferenceSqlRAG, build_mode: str = "overwrite") -> Set[str]:
    """Get existing reference SQL IDs based on build mode."""
    existing_ids = set()
    if build_mode == "overwrite":
        return existing_ids
    if build_mode == "incremental":
        for item in storage.search_all_reference_sql():
            existing_ids.add(str(item["id"]))
    return existing_ids
