#!/usr/bin/env python3
"""
Migrate storage from datus_db_{namespace} directories to unified datus_db.

Execution order:
  1. SQLite (subject_tree.db) — builds old_node_id → new_node_id mapping per namespace
  2. LanceDB — remaps subject_node_id in metrics/reference_sql/ext_knowledge using the mapping

Idempotent: safe to run multiple times. Skips data that has already been migrated
(detected by datasource_id presence).

Usage:
    python scripts/migrate_storage_v2.py --data-dir ~/.datus/data
    python scripts/migrate_storage_v2.py --data-dir ~/.datus/data --dry-run
"""

import argparse
import os
import re
import sqlite3
import sys
from typing import Dict

import lance
import lancedb
import pyarrow as pa

# Register LanceDB embedding adapters so lance can resolve metadata in existing tables
import datus.storage.vector.lance_backend  # noqa: F401 — triggers @register("fastembed")

STANDARD_FIELDS = {
    "datasource_id": pa.string(),
    "creator_id": pa.string(),
    "updator_id": pa.string(),
}

DEFAULT_CREATOR = "datus_agent"
DEFAULT_UPDATOR = "datus_agent"

# SQLite: only subject_tree.db with subject_nodes table
SQLITE_DB_NAME = "subject_tree.db"
SQLITE_TABLE_NAME = "subject_nodes"

# LanceDB tables that contain subject_node_id and need remapping
SUBJECT_NODE_ID_COLUMN = "subject_node_id"
TABLES_WITH_SUBJECT_NODE_ID = {"metrics", "reference_sql", "ext_knowledge"}

# Tables managed by other subsystems (not part of storage 2.0 migration)
# document: managed by DocumentStore with per-platform namespace isolation
# sql_history: legacy table, not migrated
# success_story: deprecated table
SKIP_TABLES = {"document", "sql_history", "success_story"}

# Type alias: namespace → {old_node_id → new_node_id}
NodeIdMapping = Dict[str, Dict[int, int]]


def find_namespace_dirs(data_dir: str) -> list[tuple[str, str]]:
    """Find all datus_db_{namespace} directories and extract namespace names."""
    pattern = re.compile(r"^datus_db_(.+)$")
    results = []
    for entry in sorted(os.listdir(data_dir)):
        match = pattern.match(entry)
        if match:
            full_path = os.path.join(data_dir, entry)
            if os.path.isdir(full_path):
                results.append((match.group(1), full_path))
    return results


def _read_lance_table(source_path: str, table_name: str) -> pa.Table:
    """Read all rows from a LanceDB table without requiring embedding model.

    Uses the lance library directly to bypass lancedb's search API which
    would trigger embedding function lookup.
    """
    table_path = os.path.join(source_path, f"{table_name}.lance")
    ds = lance.dataset(table_path)
    return ds.to_table()


def add_standard_columns(table: pa.Table, namespace: str) -> pa.Table:
    """Add datasource_id, creator_id, updator_id columns to a PyArrow table."""
    n_rows = table.num_rows
    for col_name, col_type in STANDARD_FIELDS.items():
        if col_name in table.column_names:
            continue
        if col_name == "datasource_id":
            values = [namespace] * n_rows
        elif col_name == "creator_id":
            values = [DEFAULT_CREATOR] * n_rows
        elif col_name == "updator_id":
            values = [DEFAULT_UPDATOR] * n_rows
        else:
            values = [""] * n_rows
        table = table.append_column(pa.field(col_name, col_type), pa.array(values, type=col_type))
    return table


def align_schema_to_target(source_data: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Align source data to target schema: add missing columns, drop extra columns.

    - Columns in target but not in source: filled with nulls/defaults
    - Columns in source but not in target: dropped
    - Column order follows target schema
    - Vector columns are kept as-is if dimensions match
    """
    n_rows = source_data.num_rows
    aligned_columns = []

    for field in target_schema:
        if field.name in source_data.column_names:
            src_col = source_data.column(field.name)
            # Cast if types differ (skip vector/list columns — keep source type)
            if (
                src_col.type != field.type
                and not pa.types.is_list(field.type)
                and not pa.types.is_fixed_size_list(field.type)
            ):
                try:
                    src_col = src_col.cast(field.type)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    pass  # keep original type if cast fails
            aligned_columns.append(src_col)
        else:
            # Fill missing column with appropriate defaults
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                fill = pa.array([""] * n_rows, type=field.type)
            elif pa.types.is_integer(field.type):
                fill = pa.array([0] * n_rows, type=field.type)
            elif pa.types.is_boolean(field.type):
                fill = pa.array([False] * n_rows, type=field.type)
            else:
                fill = pa.array([None] * n_rows, type=field.type)
            aligned_columns.append(fill)

    return pa.table(
        {field.name: col for field, col in zip(target_schema, aligned_columns)},
        schema=target_schema,
    )


# ---------------------------------------------------------------------------
# Phase 1: SQLite migration (must run first to build node_id mapping)
# ---------------------------------------------------------------------------


def _ensure_target_db(target_file: str, source_file: str) -> None:
    """Ensure target subject_tree.db exists with the correct schema."""
    if os.path.exists(target_file):
        return
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    src_conn = sqlite3.connect(source_file)
    try:
        src_cursor = src_conn.cursor()
        src_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{SQLITE_TABLE_NAME}'")
        row = src_cursor.fetchone()
        if not row:
            return
        create_sql = row[0]
    finally:
        src_conn.close()

    tgt_conn = sqlite3.connect(target_file)
    try:
        tgt_conn.execute(create_sql)
        tgt_conn.commit()
    finally:
        tgt_conn.close()


def _ensure_standard_columns(conn: sqlite3.Connection) -> None:
    """Add datasource_id/creator_id/updator_id columns if missing."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({SQLITE_TABLE_NAME})")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "datasource_id" not in existing_cols:
        cursor.execute(f"ALTER TABLE {SQLITE_TABLE_NAME} ADD COLUMN datasource_id TEXT DEFAULT ''")
    if "creator_id" not in existing_cols:
        cursor.execute(f"ALTER TABLE {SQLITE_TABLE_NAME} ADD COLUMN creator_id TEXT DEFAULT 'datus_agent'")
    if "updator_id" not in existing_cols:
        cursor.execute(f"ALTER TABLE {SQLITE_TABLE_NAME} ADD COLUMN updator_id TEXT DEFAULT 'datus_agent'")
    conn.commit()


def migrate_sqlite_data(data_dir: str, dry_run: bool = False) -> NodeIdMapping:
    """Migrate subject_tree.db and return old→new node_id mapping per namespace.

    Returns:
        Dict of {namespace: {old_node_id: new_node_id}}
    """
    namespace_dirs = find_namespace_dirs(data_dir)
    target_path = os.path.join(data_dir, "datus_db")
    target_file = os.path.join(target_path, SQLITE_DB_NAME)
    total_rows = 0
    skipped_namespaces = 0
    node_id_mapping: NodeIdMapping = {}

    for namespace, source_path in namespace_dirs:
        source_file = os.path.join(source_path, SQLITE_DB_NAME)
        if not os.path.exists(source_file):
            continue

        # Read source rows
        src_conn = sqlite3.connect(source_file)
        try:
            src_cursor = src_conn.cursor()
            src_cursor.execute(f"PRAGMA table_info({SQLITE_TABLE_NAME})")
            src_columns = [row[1] for row in src_cursor.fetchall()]

            src_cursor.execute(f"SELECT * FROM {SQLITE_TABLE_NAME}")
            rows = src_cursor.fetchall()
        except Exception:
            src_conn.close()
            continue
        finally:
            src_conn.close()

        if not rows:
            print(f"  [SQLite] {namespace}: 0 rows (skipped)")
            continue

        if dry_run:
            print(f"  [SQLite] {namespace}: {len(rows)} rows would be migrated")
            # Build identity mapping for dry-run (LanceDB dry-run doesn't remap anyway)
            ns_mapping = {}
            for row in rows:
                row_dict = dict(zip(src_columns, row))
                old_id = row_dict.get("node_id")
                if old_id is not None:
                    ns_mapping[old_id] = old_id
            node_id_mapping[namespace] = ns_mapping
            total_rows += len(rows)
            continue

        # Ensure target DB and schema exist
        _ensure_target_db(target_file, source_file)

        tgt_conn = sqlite3.connect(target_file)
        try:
            _ensure_standard_columns(tgt_conn)
            cursor = tgt_conn.cursor()

            # Idempotent: clear existing data for this namespace before re-importing
            cursor.execute(f"SELECT COUNT(*) FROM {SQLITE_TABLE_NAME} WHERE datasource_id = ?", (namespace,))
            existing = cursor.fetchone()[0]
            if existing >= len(rows):
                print(f"  [SQLite] {namespace}: already migrated ({existing} rows), rebuilding mapping")
                skipped_namespaces += 1
                node_id_mapping[namespace] = _rebuild_mapping_from_migrated(
                    source_file, target_file, namespace, src_columns
                )
                continue
            elif existing > 0:
                cursor.execute(f"DELETE FROM {SQLITE_TABLE_NAME} WHERE datasource_id = ?", (namespace,))
                tgt_conn.commit()
                print(f"  [SQLite] {namespace}: cleared {existing} existing rows, re-migrating")

            # Build parent_id remapping: old parent_id → new parent_id
            # We need to insert in tree order (parents before children)
            ns_mapping: Dict[int, int] = {}

            # Sort rows by node_id to ensure parents are inserted before children
            row_dicts = [dict(zip(src_columns, r)) for r in rows]
            row_dicts.sort(key=lambda r: r.get("node_id", 0))

            # Get target columns
            cursor.execute(f"PRAGMA table_info({SQLITE_TABLE_NAME})")
            tgt_columns = [row[1] for row in cursor.fetchall()]

            for row_dict in row_dicts:
                old_node_id = row_dict["node_id"]
                old_parent_id = row_dict.get("parent_id")

                # Remap parent_id: -1 (root) stays -1, others use the mapping
                if old_parent_id is not None and old_parent_id != -1:
                    new_parent_id = ns_mapping.get(old_parent_id, old_parent_id)
                    row_dict["parent_id"] = new_parent_id

                row_dict["datasource_id"] = namespace
                row_dict.setdefault("creator_id", DEFAULT_CREATOR)
                row_dict.setdefault("updator_id", DEFAULT_UPDATOR)

                # Insert without node_id (auto-increment)
                insert_cols = [c for c in tgt_columns if c != "node_id" and c in row_dict]
                placeholders = ", ".join(["?"] * len(insert_cols))
                col_names = ", ".join(insert_cols)
                values = [row_dict[c] for c in insert_cols]

                cursor.execute(
                    f"INSERT INTO {SQLITE_TABLE_NAME} ({col_names}) VALUES ({placeholders})",
                    values,
                )
                new_node_id = cursor.lastrowid
                ns_mapping[old_node_id] = new_node_id

            tgt_conn.commit()
            node_id_mapping[namespace] = ns_mapping
            print(f"  [SQLite] {namespace}: migrated {len(rows)} rows, {len(ns_mapping)} node_id mappings")
            total_rows += len(rows)

        finally:
            tgt_conn.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}SQLite migration summary:")
    print(f"  Rows migrated: {total_rows}")
    if skipped_namespaces:
        print(f"  Namespaces skipped (already migrated): {skipped_namespaces}")

    return node_id_mapping


def _rebuild_mapping_from_migrated(
    source_file: str, target_file: str, namespace: str, src_columns: list
) -> Dict[int, int]:
    """Rebuild old→new node_id mapping by matching (name, depth) between source and target."""
    mapping: Dict[int, int] = {}

    # Read source: build old_node_id → (name, parent_name_chain) index
    src_conn = sqlite3.connect(source_file)
    src_cursor = src_conn.cursor()
    src_cursor.execute(f"SELECT * FROM {SQLITE_TABLE_NAME}")
    src_rows = {
        row_dict["node_id"]: row_dict for row in src_cursor.fetchall() if (row_dict := dict(zip(src_columns, row)))
    }
    src_conn.close()

    def _src_path(node_id: int) -> str:
        """Build full path string for a source node."""
        parts = []
        current = node_id
        while current in src_rows and current != -1:
            parts.append(src_rows[current]["name"])
            current = src_rows[current].get("parent_id", -1)
        return "/".join(reversed(parts))

    # Read target: same namespace rows
    tgt_conn = sqlite3.connect(target_file)
    tgt_cursor = tgt_conn.cursor()
    tgt_cursor.execute(f"PRAGMA table_info({SQLITE_TABLE_NAME})")
    tgt_columns = [row[1] for row in tgt_cursor.fetchall()]
    tgt_cursor.execute(f"SELECT * FROM {SQLITE_TABLE_NAME} WHERE datasource_id = ?", (namespace,))
    tgt_rows = {
        row_dict["node_id"]: row_dict for row in tgt_cursor.fetchall() if (row_dict := dict(zip(tgt_columns, row)))
    }
    tgt_conn.close()

    def _tgt_path(node_id: int) -> str:
        parts = []
        current = node_id
        while current in tgt_rows and current != -1:
            parts.append(tgt_rows[current]["name"])
            current = tgt_rows[current].get("parent_id", -1)
        return "/".join(reversed(parts))

    # Build path→new_node_id index
    tgt_path_to_id = {_tgt_path(nid): nid for nid in tgt_rows}

    # Match source nodes to target by path
    for old_id in src_rows:
        path = _src_path(old_id)
        if path in tgt_path_to_id:
            mapping[old_id] = tgt_path_to_id[path]

    return mapping


# ---------------------------------------------------------------------------
# Phase 2: LanceDB migration (uses node_id mapping from Phase 1)
# ---------------------------------------------------------------------------


def _remap_subject_node_ids(data: pa.Table, mapping: Dict[int, int]) -> pa.Table:
    """Remap subject_node_id column using old→new mapping."""
    if SUBJECT_NODE_ID_COLUMN not in data.column_names:
        return data
    if not mapping:
        return data

    col = data.column(SUBJECT_NODE_ID_COLUMN)
    old_values = col.to_pylist()
    new_values = [mapping.get(v, v) if v is not None else v for v in old_values]
    col_index = data.column_names.index(SUBJECT_NODE_ID_COLUMN)
    return data.set_column(col_index, data.field(SUBJECT_NODE_ID_COLUMN), pa.array(new_values, type=col.type))


def _count_existing_rows(target_db, tbl_name: str, namespace: str) -> int:
    """Count rows in target table that already have this datasource_id."""
    if tbl_name not in target_db.table_names():
        return 0
    try:
        tbl = target_db.open_table(tbl_name)
        return tbl.count_rows(f"datasource_id = '{namespace}'")
    except Exception:
        return 0


def migrate_lance_data(data_dir: str, node_id_mapping: NodeIdMapping, dry_run: bool = False) -> None:
    """Migrate LanceDB data, remapping subject_node_id where needed."""
    namespace_dirs = find_namespace_dirs(data_dir)

    if not namespace_dirs:
        print(f"No datus_db_* directories found in {data_dir}")
        return

    target_path = os.path.join(data_dir, "datus_db")
    print(f"Target: {target_path}")
    print(f"Found {len(namespace_dirs)} namespace(s) to migrate:")
    for ns, path in namespace_dirs:
        has_mapping = ns in node_id_mapping
        mapping_size = len(node_id_mapping.get(ns, {}))
        print(f"  - {ns}: {path}" + (f" ({mapping_size} node_id mappings)" if has_mapping else " (no mapping)"))
    print()

    if dry_run:
        print("[DRY RUN] Scanning tables without writing...\n")

    target_db = None
    if not dry_run:
        target_db = lancedb.connect(target_path)

    total_tables = 0
    total_rows = 0
    remapped_tables = 0
    skipped_rows = 0

    for namespace, source_path in namespace_dirs:
        print(f"--- Migrating namespace: {namespace} ---")
        ns_mapping = node_id_mapping.get(namespace, {})
        source_db = lancedb.connect(source_path)
        table_names = source_db.table_names()

        if not table_names:
            print(f"  No tables found in {source_path}")
            continue

        for tbl_name in table_names:
            if tbl_name in SKIP_TABLES:
                print(f"  {tbl_name}: skipped (not part of storage migration)")
                continue
            try:
                # Read all rows via lance directly (bypasses embedding model requirement)
                data = _read_lance_table(source_path, tbl_name)
                n_rows = data.num_rows

                if n_rows == 0:
                    print(f"  {tbl_name}: 0 rows (skipped)")
                    continue

                # Idempotent check
                if not dry_run:
                    existing = _count_existing_rows(target_db, tbl_name, namespace)
                    if existing >= n_rows:
                        print(
                            f"  {tbl_name}: already migrated ({existing} rows for datasource_id='{namespace}'), skipped"
                        )
                        skipped_rows += existing
                        continue
                    elif existing > 0:
                        # Partial data exists (e.g. from post-refactoring usage) — clean up before re-importing
                        tgt_table = target_db.open_table(tbl_name)
                        tgt_table.delete(f"datasource_id = '{namespace}'")
                        print(f"  {tbl_name}: cleared {existing} existing rows for datasource_id='{namespace}'")

                # Add standard columns
                data = add_standard_columns(data, namespace)

                # Remap subject_node_id if applicable
                needs_remap = tbl_name in TABLES_WITH_SUBJECT_NODE_ID and SUBJECT_NODE_ID_COLUMN in data.column_names
                if needs_remap and ns_mapping:
                    data = _remap_subject_node_ids(data, ns_mapping)
                    remap_label = f", {len(ns_mapping)} node_ids remapped"
                    remapped_tables += 1
                elif needs_remap and not ns_mapping:
                    remap_label = ", WARNING: no node_id mapping available (subject_tree not migrated?)"
                else:
                    remap_label = ""

                if dry_run:
                    print(f"  {tbl_name}: {n_rows} rows would be migrated{remap_label}")
                else:
                    if tbl_name in target_db.table_names():
                        target_table = target_db.open_table(tbl_name)
                        # Align source schema to target schema (handle column differences across versions)
                        tgt_schema = lance.dataset(
                            os.path.join(os.path.join(data_dir, "datus_db"), f"{tbl_name}.lance")
                        ).schema
                        data = align_schema_to_target(data, tgt_schema)
                        target_table.add(data)
                        print(f"  {tbl_name}: appended {n_rows} rows{remap_label}")
                    else:
                        target_db.create_table(tbl_name, data)
                        print(f"  {tbl_name}: created with {n_rows} rows{remap_label}")

                total_tables += 1
                total_rows += n_rows

            except Exception as e:
                print(f"  ERROR migrating {tbl_name}: {e}", file=sys.stderr)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}LanceDB migration summary:")
    print(f"  Namespaces: {len(namespace_dirs)}")
    print(f"  Tables migrated: {total_tables}")
    print(f"  Rows migrated: {total_rows}")
    if remapped_tables:
        print(f"  Tables with subject_node_id remapped: {remapped_tables}")
    if skipped_rows:
        print(f"  Rows skipped (already migrated): {skipped_rows}")


# ---------------------------------------------------------------------------
# Phase 3: Verification
# ---------------------------------------------------------------------------


def verify_migration(data_dir: str) -> bool:
    """Verify migration correctness: row counts and subject_node_id integrity.

    Checks per namespace:
      1. Row count in target matches source for each LanceDB table
      2. Row count in target SQLite matches source
      3. All subject_node_id values in LanceDB point to valid nodes in target subject_tree

    Returns:
        True if all checks pass, False otherwise.
    """
    namespace_dirs = find_namespace_dirs(data_dir)
    if not namespace_dirs:
        print("No datus_db_* source directories found. Nothing to verify.")
        return True

    target_path = os.path.join(data_dir, "datus_db")
    if not os.path.isdir(target_path):
        print(f"ERROR: Target directory {target_path} does not exist.", file=sys.stderr)
        return False

    target_db = lancedb.connect(target_path)
    target_sqlite = os.path.join(target_path, SQLITE_DB_NAME)
    all_passed = True
    total_checks = 0
    failed_checks = 0

    # Collect valid node_ids from target subject_tree (for subject_node_id validation)
    valid_node_ids: set = set()
    if os.path.exists(target_sqlite):
        conn = sqlite3.connect(target_sqlite)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT node_id FROM {SQLITE_TABLE_NAME}")
            valid_node_ids = {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()

    for namespace, source_path in namespace_dirs:
        print(f"\n--- Verifying namespace: {namespace} ---")

        # -- SQLite row count --
        source_sqlite = os.path.join(source_path, SQLITE_DB_NAME)
        if os.path.exists(source_sqlite) and os.path.exists(target_sqlite):
            src_conn = sqlite3.connect(source_sqlite)
            tgt_conn = sqlite3.connect(target_sqlite)
            try:
                src_count = src_conn.execute(f"SELECT COUNT(*) FROM {SQLITE_TABLE_NAME}").fetchone()[0]
                tgt_count = tgt_conn.execute(
                    f"SELECT COUNT(*) FROM {SQLITE_TABLE_NAME} WHERE datasource_id = ?", (namespace,)
                ).fetchone()[0]

                total_checks += 1
                if src_count == tgt_count:
                    print(f"  [OK] subject_nodes: {src_count} rows (source) == {tgt_count} rows (target)")
                else:
                    print(
                        f"  [FAIL] subject_nodes: {src_count} rows (source) != {tgt_count} rows (target)",
                        file=sys.stderr,
                    )
                    all_passed = False
                    failed_checks += 1
            finally:
                src_conn.close()
                tgt_conn.close()

        # -- LanceDB row counts --
        source_db = lancedb.connect(source_path)
        for tbl_name in source_db.table_names():
            if tbl_name in SKIP_TABLES:
                continue
            try:
                src_data = _read_lance_table(source_path, tbl_name)
                src_count = src_data.num_rows

                total_checks += 1
                if tbl_name not in target_db.table_names():
                    print(f"  [FAIL] {tbl_name}: table missing in target", file=sys.stderr)
                    all_passed = False
                    failed_checks += 1
                    continue

                tgt_table = target_db.open_table(tbl_name)
                tgt_count = tgt_table.count_rows(f"datasource_id = '{namespace}'")

                if src_count == tgt_count:
                    print(f"  [OK] {tbl_name}: {src_count} rows (source) == {tgt_count} rows (target)")
                else:
                    print(
                        f"  [FAIL] {tbl_name}: {src_count} rows (source) != {tgt_count} rows (target)",
                        file=sys.stderr,
                    )
                    all_passed = False
                    failed_checks += 1

                # -- subject_node_id integrity check --
                if tbl_name in TABLES_WITH_SUBJECT_NODE_ID and valid_node_ids:
                    total_checks += 1
                    tgt_data = tgt_table.search().where(f"datasource_id = '{namespace}'").limit(tgt_count).to_arrow()

                    if SUBJECT_NODE_ID_COLUMN in tgt_data.column_names:
                        node_ids_in_table = set(tgt_data.column(SUBJECT_NODE_ID_COLUMN).to_pylist())
                        # Filter out None values
                        node_ids_in_table = {nid for nid in node_ids_in_table if nid is not None}
                        invalid_ids = node_ids_in_table - valid_node_ids

                        if not invalid_ids:
                            print(
                                f"  [OK] {tbl_name}.subject_node_id: "
                                f"all {len(node_ids_in_table)} distinct IDs exist in subject_tree"
                            )
                        else:
                            print(
                                f"  [FAIL] {tbl_name}.subject_node_id: "
                                f"{len(invalid_ids)} IDs not found in subject_tree: "
                                f"{sorted(invalid_ids)[:10]}{'...' if len(invalid_ids) > 10 else ''}",
                                file=sys.stderr,
                            )
                            all_passed = False
                            failed_checks += 1

            except Exception as e:
                print(f"  [ERROR] {tbl_name}: {e}", file=sys.stderr)
                all_passed = False
                failed_checks += 1

    print(f"\n{'=' * 50}")
    if all_passed:
        print(f"Verification PASSED: {total_checks} checks, all OK.")
    else:
        print(f"Verification FAILED: {failed_checks}/{total_checks} checks failed.", file=sys.stderr)
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Migrate storage from datus_db_{namespace} to unified datus_db")
    parser.add_argument("--data-dir", required=True, help="Root data directory (e.g., ~/.datus/data)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--verify", action="store_true", help="Verify migration correctness (skip migration)")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Verify-only mode
    if args.verify:
        print(f"Data directory: {data_dir}")
        print("Mode: VERIFY\n")
        print("=== Migration Verification ===")
        ok = verify_migration(data_dir)
        sys.exit(0 if ok else 1)

    print(f"Data directory: {data_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    # Phase 1: SQLite first — builds node_id mapping
    print("=== Phase 1: SQLite Migration (subject_tree) ===")
    node_id_mapping = migrate_sqlite_data(data_dir, args.dry_run)

    # Phase 2: LanceDB — uses node_id mapping for subject_node_id remapping
    print("\n=== Phase 2: LanceDB Migration ===")
    migrate_lance_data(data_dir, node_id_mapping, args.dry_run)

    # Phase 3: Auto-verify after live migration
    if not args.dry_run:
        print("\n=== Phase 3: Verification ===")
        ok = verify_migration(data_dir)
        if not ok:
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
