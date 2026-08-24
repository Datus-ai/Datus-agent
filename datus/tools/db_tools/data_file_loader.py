# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Expose uploaded data files (spreadsheets, CSV, Parquet, JSON) as SQL tables.

Every supported format is surfaced the same way: a **lazy DuckDB VIEW** over the
file on disk. Nothing is copied, so there is no second source of truth to keep
in sync, and re-running a load is idempotent. That is the whole reason this
module exists instead of a materialise-and-track-staleness design: no content
hashes, no registry, no refresh command.

How current a view is, precisely, differs by format:

* CSV / Parquet / JSON — fully lazy. The view names only the path, so every
  query re-reads the file and always sees the latest bytes.
* Spreadsheets — lazy *within a pinned range*. ``read_xlsx`` demands a complete
  ``range`` and the end of it cannot be padded (see :func:`sheet_bounds`), so
  the range is fixed at load time. Edits to existing cells show up immediately;
  rows appended past the recorded end do not, until the file is loaded again.

Hence the standing advice to the model: call the load before analysing. It is
idempotent and near-free when nothing changed, which makes "just re-load" a
cheaper rule than any staleness check.

Format support, and why it differs:

* ``.csv`` / ``.tsv``  — ``read_csv_auto``, DuckDB core.
* ``.parquet``         — ``read_parquet``, statically linked into the wheel.
* ``.json`` / ``.jsonl`` / ``.ndjson`` — ``read_json_auto``, statically linked.
* ``.xlsx`` / ``.xlsm`` — ``read_xlsx`` from the autoloadable ``excel``
  extension. Reading sheet *contents* is all the extension does: it cannot list
  sheet names and does not report the used range, so openpyxl supplies both
  (see :func:`enumerate_sheets` / :func:`sheet_bounds` for why each matters).
* ``.xls``             — DuckDB has no support at all (``read_xlsx`` is strictly
  OOXML/zip). Converted once to Parquet via pandas+xlrd, then treated as
  Parquet, so everything downstream of "make it queryable" stays uniform.

Deliberately unsupported: ``.xlsb``, ``.ods``. Both would need another optional
pandas engine, and a clear "convert to .xlsx" error is more useful than a
half-working path.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

#: Datasource key the SaaS control plane registers this catalog under. Must stay
#: in sync with ``datus_backend.utils.fs_isolation.LOCAL_FILES_DATASOURCE`` —
#: ``tests/unit/test_local_files_datasource.py`` asserts the two agree.
LOCAL_FILES_DATASOURCE = "local_files"

#: Rows of the preview handed back to the model. Enough to see the shape of the
#: data and spot a misread header; small enough not to dominate the turn.
PREVIEW_ROW_LIMIT = 20

#: How far down a sheet to look for the header row. Real-world spreadsheets put
#: a title, a blank line and maybe a note above it; beyond this it is not a
#: header, it is a layout, and the caller should pass ``header_row`` explicitly.
HEADER_PROBE_ROW_LIMIT = 30

#: Columns to sample when probing for the header row.
HEADER_PROBE_COLUMN_LIMIT = 64

_CSV_SUFFIXES = frozenset({".csv", ".tsv"})
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_EXCEL_SUFFIXES = frozenset({".xlsx", ".xlsm"})
_LEGACY_EXCEL_SUFFIXES = frozenset({".xls"})
_UNSUPPORTED_SUFFIXES = frozenset({".xlsb", ".ods"})

SUPPORTED_SUFFIXES = frozenset(
    _CSV_SUFFIXES | _PARQUET_SUFFIXES | _JSON_SUFFIXES | _EXCEL_SUFFIXES | _LEGACY_EXCEL_SUFFIXES
)

#: Table functions that read arbitrary paths. Harmless when *we* emit them
#: inside a VIEW definition (the path was already validated against the
#: filesystem policy); a hole when they appear in model-authored SQL, because
#: they would read any file the process can see and bypass the path policy
#: entirely. :func:`find_file_reading_functions` is the gate.
FILE_READING_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_xlsx",
        "read_text",
        "read_blob",
        "parquet_scan",
        "csv_scan",
        "glob",
    }
)

_FUNCTION_CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)

_IDENT_SAFE_RE = re.compile(r"[^A-Za-z0-9]+")


class DataFileError(Exception):
    """A load failed for a reason the model can act on (bad format, empty sheet)."""


@dataclass
class LoadedTable:
    """One VIEW created by a load, plus what the model needs to query it."""

    table: str
    source_file: str
    sheet: Optional[str] = None
    header_row: Optional[int] = None
    used_range: Optional[str] = None
    row_count: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    preview_columns: List[str] = field(default_factory=list)
    preview_rows: List[List[Any]] = field(default_factory=list)
    materialized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "table": self.table,
            "source_file": self.source_file,
            "row_count": self.row_count,
            "columns": self.columns,
            "preview_columns": self.preview_columns,
            "preview_rows": self.preview_rows,
        }
        if self.sheet is not None:
            payload["sheet"] = self.sheet
        if self.header_row is not None:
            payload["header_row"] = self.header_row
        if self.used_range is not None:
            payload["used_range"] = self.used_range
        if self.materialized:
            payload["materialized"] = True
        return payload


@dataclass
class SkippedSheet:
    sheet: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"sheet": self.sheet, "reason": self.reason}


# ---------------------------------------------------------------- identifiers


def sanitize_identifier(text: str) -> str:
    """Reduce arbitrary text to ``[a-z0-9_]``, or ``""`` when nothing survives.

    NFKD first so that accented latin (``é`` → ``e``) and full-width digits
    survive as their ASCII equivalents instead of being stripped. CJK has no
    such decomposition and legitimately reduces to empty — callers fall back to
    a positional or hashed name rather than transliterating, because
    romanisation is ambiguous (multi-reading characters) and a plausible-looking
    but wrong name is worse than an honest opaque one. The human-readable
    mapping travels in the tool result instead.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = _IDENT_SAFE_RE.sub("_", ascii_only).strip("_").lower()
    return cleaned


def _short_hash(*parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:8]


def build_table_name(
    *,
    rel_path: str,
    sheet: Optional[str],
    sheet_index: int,
    taken: Sequence[str],
) -> str:
    """Deterministic, SQL-safe view name for one (file, sheet) pair.

    Deterministic matters: the same upload re-loaded in a later session must
    resolve to the same name, otherwise ``CREATE OR REPLACE`` stops being
    idempotent and the catalog accumulates duplicates.
    """
    stem = sanitize_identifier(Path(rel_path).stem) or f"t_{_short_hash(rel_path)}"
    if stem[0].isdigit():
        stem = f"t_{stem}"

    if sheet is None:
        candidate = stem
    else:
        sheet_part = sanitize_identifier(sheet) or f"s{sheet_index + 1}"
        if sheet_part[0].isdigit():
            sheet_part = f"s{sheet_part}"
        candidate = f"{stem}_{sheet_part}"

    if candidate not in taken:
        return candidate
    # Two distinct sources sanitizing to the same name. Suffix with a hash of
    # the real identity so the collision resolves the same way every time.
    return f"{candidate}_{_short_hash(rel_path, sheet or '')[:6]}"


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def column_letter(index: int) -> str:
    """1-based column index → spreadsheet letters (1 → A, 27 → AA)."""
    if index < 1:
        raise ValueError(f"column index must be >= 1, got {index}")
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


# ------------------------------------------------------------------ SQL guard


def find_file_reading_functions(sql: str) -> List[str]:
    """Names from :data:`FILE_READING_FUNCTIONS` that appear called in ``sql``.

    A regex rather than a parse on purpose: this runs on every statement, and
    the question is only "does a file-reading table function appear here". Over-
    matching (a column literally named ``glob(``) fails closed with a clear
    message; under-matching would be a policy bypass, so the bias is deliberate.
    """
    found: List[str] = []
    for match in _FUNCTION_CALL_RE.finditer(sql):
        name = match.group(1).lower()
        if name in FILE_READING_FUNCTIONS and name not in found:
            found.append(name)
    return found


# ------------------------------------------------------------------- openpyxl


def enumerate_sheets(path: Path) -> List[str]:
    """Sheet names of an xlsx/xlsm workbook, in workbook order.

    DuckDB cannot do this: there is no ``xlsx_sheets()`` table function, and
    ``read_xlsx(..., sheet='nope')`` reports "Sheet not found" *without* listing
    what does exist. Multi-sheet workbooks are the common case, so enumeration
    is a hard requirement, not a nicety.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataFileError(
            "Reading .xlsx requires the 'openpyxl' package, which is not installed in this environment."
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def sheet_bounds(path: Path, sheet: str) -> Tuple[int, int]:
    """``(max_row, max_column)`` of the last cell that actually holds a value.

    This is load-bearing for correctness, not an optimisation. ``read_xlsx``
    needs a complete ``range`` (``'A3'`` alone is rejected), and overshooting the
    end is silently wrong: every padding cell reads back as a NULL *row*, so a
    2-row sheet scanned as ``A1:E200`` reports 199 rows, a 99% null rate, two
    phantom columns and an extra all-NULL group in every ``GROUP BY``. The model
    cannot tell that apart from genuinely sparse data.

    Which is exactly why ``worksheet.max_row`` / ``max_column`` are not used:
    they report the sheet's *dimension*, which counts any cell carrying only
    formatting. One bolded blank cell far below the data — routine in a
    hand-maintained export — inflates the extent to that row (verified: a
    styled ``B200`` on a 2-row sheet yields exactly the 199-row reading above).
    Scanning values costs one streaming pass, against a file every query
    re-reads anyway.
    """
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet not in workbook.sheetnames:
            raise DataFileError(f"Sheet {sheet!r} not found in {path.name}. Available: {workbook.sheetnames}")
        worksheet = workbook[sheet]
        last_row = 0
        last_column = 0
        for row_index, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
            widest = 0
            for column_index, value in enumerate(row, start=1):
                if value is not None and str(value).strip() != "":
                    widest = column_index
            if widest:
                last_row = row_index
                last_column = max(last_column, widest)
        return last_row, last_column
    finally:
        workbook.close()


# --------------------------------------------------------------- header probe


def _probe_grid(connection: Any, path: Path, sheet: str, max_row: int, max_column: int) -> List[Tuple[Any, ...]]:
    """Raw upper-left cells of a sheet, with no header or type inference.

    ``all_varchar`` avoids type-inference failures while probing, and
    ``stop_at_empty=false`` is mandatory: it defaults to on, so a blank spacer
    row under the title truncates the probe to a single row and the real header
    is never seen.
    """
    rows = min(max_row, HEADER_PROBE_ROW_LIMIT)
    cols = min(max_column, HEADER_PROBE_COLUMN_LIMIT)
    if rows < 1 or cols < 1:
        return []
    probe_range = f"A1:{column_letter(cols)}{rows}"
    sql = (
        f"SELECT * FROM read_xlsx({quote_literal(str(path))}, sheet={quote_literal(sheet)}, "
        f"range={quote_literal(probe_range)}, header=false, all_varchar=true, stop_at_empty=false)"
    )
    return connection.execute(sql).fetchall()


def detect_header_row(grid: Sequence[Sequence[Any]]) -> Optional[int]:
    """1-based index of the most plausible header row in a probed grid.

    Heuristic: the first row that has at least two non-empty cells and is
    followed by a row of equal or greater width. A title row fails it (one cell)
    and a blank spacer fails it (zero cells), which covers the shape that broke
    the naive read — title, blank, header, data.
    """

    def width(row: Sequence[Any]) -> int:
        return sum(1 for cell in row if cell is not None and str(cell).strip() != "")

    for index, row in enumerate(grid):
        if width(row) < 2:
            continue
        following = grid[index + 1] if index + 1 < len(grid) else None
        if following is not None and width(following) < 2:
            continue
        return index + 1

    # Nothing looked like a header (single-column sheet, or data with no header
    # at all). Fall back to the first non-empty row so the load still produces
    # something the model can inspect and correct via ``header_row``.
    for index, row in enumerate(grid):
        if width(row) >= 1:
            return index + 1
    return None


# --------------------------------------------------------------- scan clauses


def _csv_scan(path: Path) -> str:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    return f"read_csv_auto({quote_literal(str(path))}, delim={quote_literal(delimiter)})"


def _parquet_scan(path: Path) -> str:
    return f"read_parquet({quote_literal(str(path))})"


def _json_scan(path: Path) -> str:
    return f"read_json_auto({quote_literal(str(path))})"


def _xlsx_scan(path: Path, sheet: str, header_row: int, max_row: int, max_column: int) -> Tuple[str, str]:
    """``(scan_expression, used_range)`` for one sheet, bounded to its real extent.

    ``stop_at_empty=false`` is stated rather than inferred. The range already
    ends at the last cell holding a value (:func:`sheet_bounds`), so there is no
    padding for it to protect against — and letting DuckDB infer it would
    truncate at the first interior blank row, silently dropping every group below
    a separator row, which hand-maintained exports use freely.
    """
    used_range = f"A{header_row}:{column_letter(max_column)}{max_row}"
    scan = (
        f"read_xlsx({quote_literal(str(path))}, sheet={quote_literal(sheet)}, "
        f"range={quote_literal(used_range)}, header=true, stop_at_empty=false)"
    )
    return scan, used_range


def convert_legacy_xls(path: Path, cache_dir: Path) -> Path:
    """Convert a legacy ``.xls`` to Parquet and return the Parquet path.

    DuckDB cannot read BIFF at all, so this is the only way in. The conversion
    is cached next to the pod-local catalog and re-run whenever the source is
    newer, which is the one place in this module that needs an explicit
    staleness check — every other format is read lazily and is therefore always
    current.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataFileError("Reading legacy .xls requires 'pandas' and 'xlrd'.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    cached = cache_dir / f"{path.stem}_{_short_hash(str(path))}_{int(stat.st_mtime_ns)}_{stat.st_size}.parquet"
    if cached.exists():
        return cached

    # Stale conversions of the same source (older mtime/size) are dead weight.
    for old in cache_dir.glob(f"{path.stem}_{_short_hash(str(path))}_*.parquet"):
        try:
            old.unlink()
        except OSError:  # pragma: no cover - best effort
            logger.debug("Could not remove stale xls conversion %s", old)

    try:
        frame = pd.read_excel(path, engine="xlrd")
    except ImportError as exc:
        raise DataFileError(
            "Reading legacy .xls requires the 'xlrd' package, which is not installed in this environment."
        ) from exc
    except Exception as exc:
        raise DataFileError(f"Failed to read legacy .xls file {path.name}: {exc}") from exc

    frame.to_parquet(cached, index=False)
    return cached


# ------------------------------------------------------------------ profiling


def summarize_view(connection: Any, table: str) -> List[Dict[str, Any]]:
    """Per-column profile via DuckDB's ``SUMMARIZE``.

    Cheaper and more trustworthy than hand-rolling the same statistics in
    pandas, and it works on a lazy VIEW, so no materialisation is implied.
    """
    quoted = quote_identifier(table)
    cursor = connection.execute(f"SUMMARIZE {quoted}")
    names = [description[0] for description in cursor.description]
    keep = {
        "column_name",
        "column_type",
        "min",
        "max",
        "approx_unique",
        "avg",
        "count",
        "null_percentage",
    }
    profile: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        record = dict(zip(names, row))
        profile.append({key: _jsonable(value) for key, value in record.items() if key in keep})
    return profile


def preview_view(connection: Any, table: str, limit: int = PREVIEW_ROW_LIMIT) -> Tuple[List[str], List[List[Any]]]:
    quoted = quote_identifier(table)
    cursor = connection.execute(f"SELECT * FROM {quoted} LIMIT {int(limit)}")
    names = [description[0] for description in cursor.description]
    rows = [[_jsonable(value) for value in row] for row in cursor.fetchall()]
    return names, rows


def count_view(connection: Any, table: str) -> int:
    quoted = quote_identifier(table)
    row = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()
    return int(row[0]) if row else 0


def _jsonable(value: Any) -> Any:
    """Coerce DuckDB scalars into something a tool result can carry.

    ``SUMMARIZE`` returns ``null_percentage`` as ``Decimal`` and dates as
    ``date``/``datetime``; both break naive JSON serialisation downstream.
    """
    import datetime
    import decimal

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


# ----------------------------------------------------------------- the loader


def ownership_tag(rel_path: str, sheet: Optional[str]) -> str:
    """Marker stored as the object's COMMENT, identifying what a table came from.

    The catalog is the only bookkeeping this module keeps, and this tag is why it
    is enough. Without it, a reload cannot tell "this name is mine, replace it"
    from "this name belongs to a different file, pick another" — and guessing
    either way is broken: always replacing silently clobbers an unrelated
    upload, always suffixing makes reloads non-idempotent and grows the catalog
    without bound. The tag doubles as the provenance shown by ``describe_table``.
    """
    suffix = f"|sheet={sheet}" if sheet is not None else ""
    return f"datus:source={rel_path}{suffix}"


def registered_objects(connection: Any) -> Dict[str, Optional[str]]:
    """Every name in the catalog mapped to its ownership comment (``None`` if unset)."""
    try:
        rows = connection.execute(
            "SELECT view_name, comment FROM duckdb_views() WHERE database_name != 'system' "
            "UNION ALL "
            "SELECT table_name, comment FROM duckdb_tables() WHERE database_name != 'system'"
        ).fetchall()
    except Exception as exc:  # pragma: no cover - fresh/empty catalog
        logger.debug("Could not enumerate uploads catalog: %s", exc)
        return {}
    return {row[0]: row[1] for row in rows}


def _names_owned_by_others(existing: Dict[str, Optional[str]], tag: str) -> List[str]:
    return [name for name, comment in existing.items() if comment != tag]


def _drop_existing(connection: Any, table: str) -> None:
    """Drop whatever currently owns ``table``, view or table.

    ``CREATE OR REPLACE`` only replaces an object of the *same* kind, and
    ``DROP TABLE IF EXISTS`` is not a no-op when a view holds the name — DuckDB
    raises "Existing object X is of type View, trying to drop type Table". So the
    kind has to be looked up first. This is reached whenever a reload flips
    ``materialize``, which is exactly when a user is iterating.
    """
    quoted = quote_identifier(table)
    literal = quote_literal(table)
    try:
        is_view = connection.execute(
            f"SELECT count(*) FROM duckdb_views() WHERE view_name = {literal} AND database_name != 'system'"
        ).fetchone()
        if is_view and is_view[0]:
            connection.execute(f"DROP VIEW IF EXISTS {quoted}")
            return
        is_table = connection.execute(
            f"SELECT count(*) FROM duckdb_tables() WHERE table_name = {literal} AND database_name != 'system'"
        ).fetchone()
        if is_table and is_table[0]:
            connection.execute(f"DROP TABLE IF EXISTS {quoted}")
    except Exception as exc:  # pragma: no cover - catalog probe is best effort
        logger.debug("Could not probe existing object %s: %s", table, exc)


def _create_view(connection: Any, table: str, scan: str, materialize: bool, tag: str) -> None:
    _drop_existing(connection, table)
    quoted = quote_identifier(table)
    if materialize:
        # Snapshot semantics, chosen explicitly by the caller: worth it when the
        # source is large enough that re-scanning it per query dominates, at the
        # cost of no longer tracking edits to the file.
        connection.execute(f"CREATE TABLE {quoted} AS SELECT * FROM {scan}")
        kind = "TABLE"
    else:
        connection.execute(f"CREATE VIEW {quoted} AS SELECT * FROM {scan}")
        kind = "VIEW"
    # Set after creation, not via CREATE OR REPLACE: replacing an object drops
    # its comment, so the tag has to be (re)applied on every load.
    connection.execute(f"COMMENT ON {kind} {quoted} IS {quote_literal(tag)}")


def _finish_table(
    connection: Any,
    *,
    table: str,
    source_file: str,
    sheet: Optional[str],
    header_row: Optional[int],
    used_range: Optional[str],
    materialized: bool,
) -> LoadedTable:
    preview_columns, preview_rows = preview_view(connection, table)
    return LoadedTable(
        table=table,
        source_file=source_file,
        sheet=sheet,
        header_row=header_row,
        used_range=used_range,
        row_count=count_view(connection, table),
        columns=summarize_view(connection, table),
        preview_columns=preview_columns,
        preview_rows=preview_rows,
        materialized=materialized,
    )


def inspect_file(path: Path, *, connection: Any, sheet: Optional[str] = None) -> Dict[str, Any]:
    """Describe a file without creating anything.

    The point is the *un-interpreted* grid for spreadsheets: when the header
    guess is wrong, the model needs to see the raw upper-left cells to work out
    the right ``header_row``, and a parsed preview cannot show that.
    """
    suffix = path.suffix.lower()
    result: Dict[str, Any] = {"file": path.name, "format": suffix.lstrip(".")}

    if suffix in _EXCEL_SUFFIXES:
        sheets = enumerate_sheets(path)
        result["sheets"] = sheets
        targets = [sheet] if sheet else sheets
        grids = []
        for name in targets:
            if name not in sheets:
                raise DataFileError(f"Sheet {name!r} not found in {path.name}. Available: {sheets}")
            max_row, max_column = sheet_bounds(path, name)
            if max_row < 1 or max_column < 1:
                grids.append({"sheet": name, "empty": True})
                continue
            grid = _probe_grid(connection, path, name, max_row, max_column)
            grids.append(
                {
                    "sheet": name,
                    "used_rows": max_row,
                    "used_columns": max_column,
                    "detected_header_row": detect_header_row(grid),
                    "raw_rows": [[_jsonable(cell) for cell in row] for row in grid],
                }
            )
        result["sheet_details"] = grids
        return result

    if suffix in _CSV_SUFFIXES:
        scan = _csv_scan(path)
    elif suffix in _PARQUET_SUFFIXES:
        scan = _parquet_scan(path)
    elif suffix in _JSON_SUFFIXES:
        scan = _json_scan(path)
    elif suffix in _LEGACY_EXCEL_SUFFIXES:
        result["note"] = "Legacy .xls is converted to Parquet on load; run without inspect_only to query it."
        return result
    else:
        raise DataFileError(_unsupported_message(path))

    cursor = connection.execute(f"SELECT * FROM {scan} LIMIT {PREVIEW_ROW_LIMIT}")
    result["preview_columns"] = [description[0] for description in cursor.description]
    result["preview_rows"] = [[_jsonable(value) for value in row] for row in cursor.fetchall()]
    return result


def _unsupported_message(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _UNSUPPORTED_SUFFIXES:
        return f"{suffix} files are not supported. Re-save {path.name} as .xlsx or .csv and upload that instead."
    return (
        f"Unsupported data file format {suffix or '(none)'} for {path.name}. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
    )


def load_file(
    path: Path,
    rel_path: str,
    *,
    connection: Any,
    conversion_cache_dir: Path,
    sheet: Optional[str] = None,
    header_row: Optional[int] = None,
    materialize: bool = False,
    existing_objects: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[LoadedTable], List[SkippedSheet]]:
    """Create one VIEW (or table, when ``materialize``) per loadable unit.

    A spreadsheet's unit is a sheet; every other format is a single unit. Sheets
    that cannot be read are collected into the skipped list rather than failing
    the whole load — an empty "notes" tab is normal in real workbooks and must
    not cost the user the sheets that do have data.
    """
    suffix = path.suffix.lower()
    existing = dict(existing_objects or {})
    loaded: List[LoadedTable] = []
    skipped: List[SkippedSheet] = []

    if suffix in _EXCEL_SUFFIXES:
        sheets = enumerate_sheets(path)
        if sheet is not None:
            if sheet not in sheets:
                raise DataFileError(f"Sheet {sheet!r} not found in {path.name}. Available: {sheets}")
            targets = [(sheets.index(sheet), sheet)]
        else:
            targets = list(enumerate(sheets))

        for index, name in targets:
            try:
                max_row, max_column = sheet_bounds(path, name)
                if max_row < 1 or max_column < 1:
                    skipped.append(SkippedSheet(name, "sheet is empty"))
                    continue
                resolved_header = header_row
                if resolved_header is None:
                    grid = _probe_grid(connection, path, name, max_row, max_column)
                    resolved_header = detect_header_row(grid)
                if resolved_header is None:
                    skipped.append(SkippedSheet(name, "no header row could be located"))
                    continue
                if resolved_header >= max_row + 1:
                    skipped.append(SkippedSheet(name, "header row is past the end of the sheet"))
                    continue

                tag = ownership_tag(rel_path, name)
                table = build_table_name(
                    rel_path=rel_path,
                    sheet=name,
                    sheet_index=index,
                    taken=_names_owned_by_others(existing, tag),
                )
                scan, used_range = _xlsx_scan(path, name, resolved_header, max_row, max_column)
                _create_view(connection, table, scan, materialize, tag)
                existing[table] = tag
                loaded.append(
                    _finish_table(
                        connection,
                        table=table,
                        source_file=rel_path,
                        sheet=name,
                        header_row=resolved_header,
                        used_range=used_range,
                        materialized=materialize,
                    )
                )
            except DataFileError as exc:
                skipped.append(SkippedSheet(name, str(exc)))
            except Exception as exc:
                # DuckDB raises for an empty sheet ("No rows found in xlsx
                # file") among other per-sheet problems. One bad tab must not
                # sink the workbook.
                logger.info("Skipping sheet %s of %s: %s", name, rel_path, exc)
                # An exception with an empty message yields [] from splitlines(),
                # so indexing it would make this handler the thing that fails.
                reason = (str(exc).splitlines() or [repr(exc)])[0]
                skipped.append(SkippedSheet(name, reason))

        if not loaded and skipped:
            reasons = "; ".join(f"{item.sheet}: {item.reason}" for item in skipped)
            raise DataFileError(f"No readable sheet in {path.name} ({reasons})")
        return loaded, skipped

    if suffix in _CSV_SUFFIXES:
        scan = _csv_scan(path)
    elif suffix in _PARQUET_SUFFIXES:
        scan = _parquet_scan(path)
    elif suffix in _JSON_SUFFIXES:
        scan = _json_scan(path)
    elif suffix in _LEGACY_EXCEL_SUFFIXES:
        scan = _parquet_scan(convert_legacy_xls(path, conversion_cache_dir))
    else:
        raise DataFileError(_unsupported_message(path))

    tag = ownership_tag(rel_path, None)
    table = build_table_name(
        rel_path=rel_path,
        sheet=None,
        sheet_index=0,
        taken=_names_owned_by_others(existing, tag),
    )
    _create_view(connection, table, scan, materialize, tag)
    loaded.append(
        _finish_table(
            connection,
            table=table,
            source_file=rel_path,
            sheet=None,
            header_row=None,
            used_range=None,
            materialized=materialize,
        )
    )
    return loaded, skipped


def default_conversion_cache_dir(db_path: str) -> Path:
    """Where ``.xls`` → Parquet conversions live: next to the DuckDB catalog.

    Same lifetime and same locality as the catalog itself, so a pod losing one
    loses the other and both rebuild together.
    """
    return Path(os.path.dirname(db_path) or ".") / "_conversions"
