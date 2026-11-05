"""Connection-owned metadata for optional SQL policy providers.

This module resolves physical objects. Policy matching and rewrite algorithms
belong to the provider. No private package is imported by Agent.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from datus.utils.sql_utils import table_name_field_order

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _paths(value: str) -> list[str]:
    """Split a namespace search list without splitting quoted commas."""
    result, part, quoted = [], [], False
    for char in value:
        if char == '"':
            quoted = not quoted
        if char == "," and not quoted:
            result.append("".join(part).strip())
            part = []
        else:
            part.append(char)
    if quoted:
        raise ValueError("invalid namespace search path")
    result.append("".join(part).strip())
    return [p for p in result if p]


class SqlReadContext:
    def canonical_name(self, name: str) -> str:
        if self.dialect in {"duckdb", "sqlite"}:
            return name.translate(_ASCII_LOWER)
        if self.dialect == "spark":
            if self._state["case_sensitive_names"]:
                return name
            if not name.isascii():
                raise ValueError("case-insensitive Spark Unicode identifiers require server-owned identifier binding")
        if self.dialect in {"hive", "spark", "trino"}:
            # These engines fold quoted identifiers too. Apply the same key
            # to query sources, profile targets and metadata listings.
            return name.lower()
        if self.dialect in {"mysql", "tidb"} and self._state["lower_case_table_names"]:
            if name.isascii():
                return name.translate(_ASCII_LOWER)
            if name not in self._folded_names:
                # Hex avoids dialect-dependent literal/backslash escaping.
                folded = self._query(
                    f"SELECT LOWER(CONVERT(0x{name.encode('utf-8').hex()} USING utf8mb4)) AS folded_name"
                )[0]["folded_name"]
                if not isinstance(folded, str):
                    raise ValueError("server identifier case normalization is unavailable")
                self._folded_names[name] = folded
            return self._folded_names[name]
        return name

    def fixed_object(self, parts: list[dict]) -> dict:
        """Bind a policy target, filling omitted prefixes from this live session.

        Explicit coordinates stay fixed. Defaults are captured once per bound
        policy; a namespace switch requires a fresh binding.
        """
        self.verify_context()
        specified = self._parts(parts)
        namespace = {}
        for field in self.namespace_order[:-1]:
            value = specified[field] if field in specified else self._state.get(field)
            if not value:
                raise ValueError(f"current execution {field} is unavailable; qualify the policy target explicitly")
            namespace[field] = self.canonical_name(value)
        return {
            "datasource": self.datasource,
            **{key: namespace.get(key) for key in ("catalog", "database", "schema")},
            "table": self.canonical_name(specified["table"]),
        }

    def column_name(self, identifier: dict) -> str:
        name = identifier["name"]
        if self.dialect in {"mysql", "tidb"}:
            # Column names are insensitive independently of the table setting;
            # the shared core matches them while retaining stored output names.
            return name
        if not identifier.get("quoted"):
            if self.dialect in {"postgres", "postgresql", "hologres", "gaussdb", "dws"}:
                name = name.translate(_ASCII_LOWER)
            elif self.dialect in {"snowflake", "oracle", "hana"}:
                name = name.upper()
        return self.canonical_name(name)

    def __init__(self, connector: Any, datasource: str):
        self.connector = connector
        self.datasource = datasource
        self.dialect = str(connector.dialect)
        self.namespace_order = [name.removesuffix("_name") for name in table_name_field_order(self.dialect)]
        self._folded_names: dict[str, str] = {}
        self._state = self._capture()
        # Keep the original object alive so a replacement connection cannot
        # reuse its Python id and appear to be the same SQLite session.
        self._sqlite_connection = getattr(connector, "connection", None) if self.dialect == "sqlite" else None
        self.execution_context = {
            "datasource": datasource,
            "source_dialect": self.dialect,
            **{key: self._state.get(key) or None for key in ("catalog", "database", "schema")},
            "search_path": self._state["search_path"],
            "revision": json.dumps(self._state, sort_keys=True),
        }
        if self.dialect in {"mysql", "tidb"}:
            self.execution_context["case_sensitive_table_names"] = self._state["lower_case_table_names"] == 0
        elif self.dialect == "spark":
            self.execution_context["case_sensitive_table_names"] = self._state["case_sensitive_names"]
        self._metadata: dict[tuple, dict] = {}

    def _query(self, sql: str) -> list[dict]:
        result = self.connector.execute_query(sql, result_format="arrow")
        if not getattr(result, "success", False):
            raise ValueError(f"cannot read execution metadata: {getattr(result, 'error', '')}")
        data = result.sql_return
        if hasattr(data, "to_pylist"):
            return data.to_pylist()
        if hasattr(data, "to_dict"):
            return data.to_dict(orient="records")
        if isinstance(data, list) and all(isinstance(row, dict) for row in data):
            return data
        raise ValueError("connector returned unsupported execution metadata")

    def _capture(self) -> dict:
        getter = getattr(self.connector, "get_current_context", None)
        names = (
            getter()
            if callable(getter)
            else {
                key + "_name": getattr(self.connector, key + "_name", "") for key in ("catalog", "database", "schema")
            }
        )
        state = {key: str(names.get(key + "_name") or "") for key in ("catalog", "database", "schema")}
        # Service/tenant identifies the execution connection, but is not an
        # Oracle/HANA SQL object coordinate (their objects are schema.table).
        config = getattr(self.connector, "config", None)
        if self.dialect == "oracle":
            state["database"] = str(getattr(config, "service_name", None) or state["database"])
        elif self.dialect == "hana":
            state["database"] = str(getattr(config, "database", None) or state["database"])
        if self.dialect == "sqlite":
            # Opening the temp catalog is observable in database_list. Do it
            # before taking the immutable attachment snapshot.
            self._query("SELECT COUNT(*) AS n FROM sqlite_temp_master")
            databases = self._query("PRAGMA database_list")
            state.update(
                catalog="",
                database="main",
                schema="",
                search_path=[
                    "temp",
                    "main",
                    *[row["name"] for row in databases if row["name"] not in {"temp", "main"}],
                ],
                attachments=[(row["name"], row["file"]) for row in databases],
                connection_id=id(getattr(self.connector, "connection", None)),
            )
        elif self.dialect == "duckdb":
            row = self._query(
                "SELECT current_database() AS database_name, current_schema() AS schema_name, current_setting('search_path') AS search_path"
            )[0]
            state.update(
                database=row["database_name"], schema=row["schema_name"], search_path=_paths(row["search_path"] or "")
            )
            # ATTACH/DETACH changes two-part source resolution without changing
            # current_database or search_path. Bind that topology as well.
            state["attachments"] = [
                (row["database_name"], row["path"])
                for row in self._query("SELECT database_name,path FROM duckdb_databases() ORDER BY database_name")
            ]
        elif self.dialect in {"postgres", "postgresql", "hologres", "gaussdb", "dws"}:
            row = self._query(
                "SELECT current_database() AS database_name, current_schema() AS schema_name, current_schemas(true) AS search_path"
            )[0]
            state.update(
                database=row["database_name"], schema=row["schema_name"], search_path=list(row["search_path"] or [])
            )
        else:
            state["search_path"] = [state["schema"]] if state["schema"] else []
            if self.dialect == "spark":
                getter = getattr(self.connector, "get_identifier_case_sensitive", None)
                mode = getter() if callable(getter) else None
                if not isinstance(mode, bool):
                    raise ValueError("actual Spark identifier case mode is unavailable")
                state["case_sensitive_names"] = mode
            if self.dialect in {"mysql", "tidb"}:
                mode = int(
                    self._query("SELECT @@lower_case_table_names AS lower_case_table_names")[0][
                        "lower_case_table_names"
                    ]
                )
                if mode not in {0, 1, 2}:
                    raise ValueError("unsupported server table-name case mode")
                state["lower_case_table_names"] = mode
            for field in self.namespace_order[:-1]:
                if not state.get(field):
                    raise ValueError(f"actual execution {field} is unavailable for {self.dialect}")
        return state

    def verify_context(self) -> None:
        if self._capture() != self._state or (
            self.dialect == "sqlite" and self.connector.connection is not self._sqlite_connection
        ):
            raise ValueError("SQL execution namespace changed; rebind policy and recompile")

    def verify_execution_target(self, target: str) -> None:
        self.verify_context()
        if self.dialect == "spark" and self._state["case_sensitive_names"]:
            raise ValueError("Dosi cannot reproduce Spark case-sensitive session settings")
        if self.dialect == "sqlite" and target == "duckdb":
            if any(name not in {"main", "temp"} for name, _ in self._state["attachments"]):
                raise ValueError("Dosi SQLite execution cannot reproduce attached databases from the sqlite3 session")
            if self._query("SELECT name FROM sqlite_temp_master WHERE type IN ('table','view')"):
                raise ValueError("Dosi SQLite execution cannot reproduce temporary objects from the sqlite3 session")
            if not any(name == "main" and path for name, path in self._state["attachments"]):
                raise ValueError("Dosi SQLite execution requires a file-backed database")

    def _parts(self, parts: list[dict]) -> dict:
        if not parts or len(parts) > len(self.namespace_order):
            raise ValueError("table name does not match the adapter namespace hierarchy")
        values = []
        for item in parts:
            name = item["name"]
            if not item.get("quoted"):
                if self.dialect in {"postgres", "postgresql", "hologres", "gaussdb", "dws"}:
                    name = name.translate(_ASCII_LOWER)
                elif self.dialect in {"snowflake", "oracle", "hana"}:
                    name = name.upper()
            values.append(self.canonical_name(name))
        return dict(zip(self.namespace_order[-len(values) :], values))

    def resolve_source_parts(
        self, parts: list[dict], context: dict, *, parser: Callable[[str], list[dict]], fixed: bool = False
    ) -> dict:
        self.verify_context()
        if context != self.execution_context:
            raise ValueError("metadata request does not match the bound execution context")
        specified = self._parts(parts)
        if fixed and len(parts) != len(self.namespace_order):
            raise ValueError("policy targets must include every namespace level supported by the adapter")
        key = (tuple((p["name"], p.get("quoted", False)) for p in parts), fixed)
        if key in self._metadata:
            return self._metadata[key]
        duckdb_catalog_prefix = False
        if self.dialect == "duckdb" and len(parts) == 2 and not fixed:
            prefix = self.canonical_name(specified["schema"])
            catalogs = self._query("SELECT database_name FROM duckdb_databases()")
            if any(self.canonical_name(row["database_name"]) == prefix for row in catalogs):
                schemas = self._query(
                    f"SELECT schema_name FROM duckdb_schemas() WHERE database_name={_literal(self._state['database'])}"
                )
                if any(self.canonical_name(row["schema_name"]) == prefix for row in schemas):
                    raise ValueError("ambiguous DuckDB catalog or schema; qualify database.schema.table explicitly")
                # Resolve this catalog through its search-path schemas, then
                # main. Profile-target defaulting retains right alignment.
                specified = {"database": specified["schema"], "table": specified["table"]}
                duckdb_catalog_prefix = True
        base = {
            key: self.canonical_name(specified.get(key) or self._state[key])
            for key in ("catalog", "database", "schema")
        }
        candidates = [base]
        if duckdb_catalog_prefix:
            candidates = []
            for path in self._state["search_path"]:
                namespace = self._parts(parser(path + ".__datus_probe"))
                if self.canonical_name(namespace.get("database", self._state["database"])) == self.canonical_name(
                    specified["database"]
                ):
                    candidates.append({**base, "schema": namespace["schema"]})
            candidates.append({**base, "schema": "main"})
        if self.dialect == "sqlite" and "database" not in specified:
            candidates = [{**base, "database": name} for name in self._state["search_path"]]
        if self.dialect == "duckdb" and "database" not in specified:
            # Schema qualification alone does not bypass DuckDB's temporary
            # catalog. Only an explicit database fixes the physical catalog.
            candidates = [{**base, "database": "temp", "schema": specified.get("schema", "main")}]
            if "schema" not in specified:
                for path in self._state["search_path"]:
                    namespace = self._parts(parser(path + ".__datus_probe"))
                    candidates.append({**base, **{k: v for k, v in namespace.items() if k != "table"}})
            candidates.append(base)
            if "schema" not in specified:
                # DuckDB implicitly searches the persistent main schema after
                # the configured path, even when main is absent from that path.
                candidates.append({**base, "schema": "main"})
        elif not duckdb_catalog_prefix and "schema" in self.namespace_order and "schema" not in specified:
            candidates = [{**base, "schema": name} for name in self._state["search_path"]] or [base]
        for namespace in candidates:
            table = self._lookup(namespace, specified["table"], parser)
            if table is not None:
                if not table["physical"]:
                    raise ValueError("opaque database views require an explicitly supported policy path")
                self._metadata[key] = table
                return table
        raise ValueError("physical table could not be resolved in the execution namespace")

    def _lookup(self, namespace: dict, name: str, parser: Callable[[str], list[dict]]) -> dict | None:
        # Search-path candidates can come from session state rather than _parts.
        # Metadata APIs often use case-sensitive string predicates internally.
        namespace = {field: self.canonical_name(value) for field, value in namespace.items()}
        name = self.canonical_name(name)
        if self.dialect == "sqlite":
            database = self.canonical_name(namespace["database"])
            if database not in {self.canonical_name(name) for name in self._state["search_path"]}:
                raise ValueError("SQLite database is not attached to the execution connection")
            namespace = {**namespace, "database": database}
            tables = self._query(
                f"SELECT name,type,sql FROM {_quote(database)}.sqlite_schema "
                f"WHERE name={_literal(name)} COLLATE NOCASE AND type IN ('table','view')"
            )
            if not tables:
                return None
            if len(tables) != 1:
                raise ValueError("ambiguous physical metadata")
            table = tables[0]
            kinds = self._query(
                "SELECT type FROM pragma_table_list WHERE "
                f"schema={_literal(database)} COLLATE NOCASE AND name={_literal(table['name'])} COLLATE NOCASE"
            )
            if table["type"] != "table" or len(kinds) != 1 or kinds[0]["type"] != "table":
                return {"physical": False}
            name = table["name"]
            columns = self._query(f"PRAGMA {_quote(database)}.table_xinfo({_literal(name)})")
            columns = [column for column in columns if column.get("hidden", 0) != 1]
            physical = True
        elif self.dialect == "duckdb":

            def conditions(object_field: str) -> str:
                return " AND ".join(
                    f"lower({field}_name)=lower({_literal(value)})"
                    for field, value in {
                        "database": namespace["database"],
                        "schema": namespace["schema"],
                        object_field: name,
                    }.items()
                )

            tables = self._query(
                "SELECT database_name,schema_name,table_name,true AS physical FROM duckdb_tables() WHERE "
                + conditions("table")
                + " UNION ALL SELECT database_name,schema_name,view_name AS table_name,false AS physical "
                "FROM duckdb_views() WHERE " + conditions("view")
            )
            tables = [
                row
                for row in tables
                if all(
                    self.canonical_name(row[field + "_name"]) == self.canonical_name(value)
                    for field, value in {
                        "database": namespace["database"],
                        "schema": namespace["schema"],
                        "table": name,
                    }.items()
                )
            ]
            if not tables:
                return None
            if len(tables) != 1:
                raise ValueError("ambiguous physical metadata")
            row = tables[0]
            namespace = {**namespace, "database": row["database_name"], "schema": row["schema_name"]}
            name = row["table_name"]
            columns = self._query(
                "SELECT column_name AS name,data_type AS type FROM duckdb_columns() WHERE "
                + " AND ".join(
                    f"{field}_name={_literal(value)}"
                    for field, value in {
                        "database": namespace["database"],
                        "schema": namespace["schema"],
                        "table": name,
                    }.items()
                )
                + " ORDER BY column_index"
            )
            physical = row["physical"]
        else:
            if (
                self.dialect in {"postgres", "postgresql", "hologres", "gaussdb", "dws"}
                and namespace["database"] != self._state["database"]
            ):
                raise ValueError("cross-database reads are not supported by this execution connection")
            kwargs = {key + "_name": value for key, value in namespace.items()}
            tables = self.connector.get_tables(**kwargs)
            views = self.connector.get_views(**kwargs)

            # Fully scoped getters return raw stored names, including literal
            # dots. HANA explicitly returns quoted schema.table identifiers.
            # Do not infer qualification from punctuation in other listings.
            def matches_name(reference):
                if self.dialect != "hana":
                    return self.canonical_name(str(reference)) == self.canonical_name(name)
                try:
                    stored = parser(str(reference))
                    if len(stored) != 2 or not all(part.get("quoted") for part in stored):
                        raise ValueError("HANA metadata must contain quoted schema.table identifiers")
                    parts = self._parts(stored)
                except Exception:
                    return False
                return all(
                    self.canonical_name(parts.get(key, namespace.get(key, ""))) == self.canonical_name(value)
                    for key, value in {**namespace, "table": name}.items()
                )

            # Some adapters' table listings also contain views. An explicit
            # view classification must win over the broader table listing.
            if any(matches_name(view) for view in views):
                return {"physical": False}
            matches = [table for table in tables if matches_name(table)]
            if not matches:
                return None
            columns = self.connector.get_schema(**kwargs, table_name=name)
            physical = True
        object_id = {
            "datasource": self.datasource,
            **{
                key: self.canonical_name(namespace[key]) if key in self.namespace_order and namespace.get(key) else None
                for key in ("catalog", "database", "schema")
            },
            "table": self.canonical_name(name),
        }
        quote_identifier = getattr(self.connector, "quote_identifier", _quote)
        reference = ".".join(quote_identifier(object_id[field]) for field in self.namespace_order)
        normalized = []
        for column in columns:
            column_name = column.get("name") or column.get("column_name") or column.get("Field")
            sql_type = column.get("type") or column.get("data_type") or column.get("Type")
            if not column_name or not sql_type:
                raise ValueError("column name or native SQL type is unavailable")
            text_type = str(sql_type).upper()
            while text_type.startswith(("NULLABLE(", "LOWCARDINALITY(")) and text_type.endswith(")"):
                text_type = text_type[text_type.index("(") + 1 : -1]
            normalized.append(
                {
                    "name": column_name,
                    "sql_type": sql_type,
                    "is_text": "INT" not in text_type and any(name in text_type for name in ("CHAR", "CLOB", "TEXT"))
                    if self.dialect == "sqlite"
                    else text_type.startswith(("VARCHAR", "NVARCHAR", "CHAR", "NCHAR", "TEXT", "STRING")),
                }
            )
        return {"object": object_id, "columns": normalized, "physical": physical, "sql_reference": reference}
