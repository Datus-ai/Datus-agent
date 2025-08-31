import hashlib
from typing import Set

from datus.storage.sql_history.store import SqlHistoryRAG


def gen_sql_history_id(sql: str, comment: str) -> str:
    """Generate MD5 hash ID from SQL and comment fields."""
    combined_text = f"{sql}{comment}"
    return hashlib.md5(combined_text.encode("utf-8")).hexdigest()


def exists_sql_history(storage: SqlHistoryRAG, build_mode: str = "overwrite") -> Set[str]:
    """Get existing SQL history IDs based on build mode."""
    existing_ids = set()
    if build_mode == "overwrite":
        return existing_ids
    if build_mode == "incremental":
        for item in storage.search_all_sql_history(""):
            existing_ids.add(str(item["id"]))
    return existing_ids
