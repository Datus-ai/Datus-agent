# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Autocomplete module for Datus CLI.
Provides SQL keyword, table name, and column name autocompletion.
"""

import re
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Tuple, Union

import pyarrow
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from pygments.lexers.sql import SqlLexer
from pygments.styles.default import DefaultStyle
from pygments.token import Token

from datus.configuration.agent_config import AgentConfig
from datus.schemas.node_models import Metric, ReferenceSql, TableSchema
from datus.utils.constants import SYS_SUB_AGENTS, DBType
from datus.utils.loggings import get_logger
from datus.utils.path_utils import get_file_fuzzy_matches
from datus.utils.reference_paths import REFERENCE_PATH_REGEX, normalize_reference_path

logger = get_logger(__name__)

# Common SQL keywords and functions
SQL_KEYWORDS = [
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP BY",
    "HAVING",
    "ORDER BY",
    "JOIN",
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "LIMIT",
    "OFFSET",
    "UNION",
    "UNION ALL",
    "INTERSECT",
    "EXCEPT",
    "INSERT INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE FROM",
    "CREATE TABLE",
    "ALTER TABLE",
    "DROP TABLE",
    "TRUNCATE TABLE",
    "CREATE INDEX",
    "DROP INDEX",
    "CREATE VIEW",
    "DROP VIEW",
    "WITH",
    "AS",
    "ON",
    "USING",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "BETWEEN",
    "IS NULL",
    "IS NOT NULL",
    "ASC",
    "DESC",
    "DISTINCT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "EXISTS",
    "ALL",
    "ANY",
]

# Common SQL functions
SQL_FUNCTIONS = [
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "COALESCE",
    "NULLIF",
    "CURRENT_DATE",
    "CURRENT_TIME",
    "CURRENT_TIMESTAMP",
    "EXTRACT",
    "CAST",
    "CONCAT",
    "SUBSTRING",
    "UPPER",
    "LOWER",
    "TRIM",
    "LENGTH",
    "ROUND",
    "ABS",
    "RANDOM",
    "FLOOR",
    "CEILING",
    "POWER",
    "SQRT",
    "DATE_PART",
    "TO_CHAR",
    "TO_DATE",
    "TO_NUMBER",
    "NVL",
    "DECODE",
]

# SQL data types
SQL_TYPES = [
    "INT",
    "INTEGER",
    "SMALLINT",
    "BIGINT",
    "DECIMAL",
    "NUMERIC",
    "FLOAT",
    "REAL",
    "DOUBLE PRECISION",
    "BOOLEAN",
    "CHAR",
    "VARCHAR",
    "TEXT",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "INTERVAL",
    "BLOB",
    "BYTEA",
    "UUID",
    "JSON",
    "JSONB",
    "ARRAY",
    "ENUM",
]


class SQLCompleter(Completer):
    """SQL completer for prompt_toolkit."""

    def __init__(self):
        """Initialize the SQL completer."""
        self.keywords = SQL_KEYWORDS.copy()
        self.functions = SQL_FUNCTIONS.copy()
        self.types = SQL_TYPES.copy()

        # Tables and columns (populated dynamically)
        self.tables: Dict[str, List[str]] = {}  # table_name -> [column1, column2, ...]
        self.table_aliases: Dict[str, str] = {}  # alias -> original_table

        # Metadata about the database
        self.database_name = ""
        self.schema_name = ""

        # Command completions
        self.commands = self._get_command_completions()
        self.at_cmds = ["table", "metric"]

    def _get_command_completions(self) -> Dict:
        """Get a nested completer for command completions."""
        return {
            # Tool commands
            "!": None,
            # "!darun": None,
            # "!dastart": None,
            "!sl": None,
            "!schema_linking": None,
            "!sq": None,
            "!search_sql": None,
            "!sm": None,
            "!search_metrics": None,
            # "!gen": None,
            # "!run": None,
            # "!fix": None,
            # "!rf": None,
            # "!reason": None,
            "!save": None,
            "!bash": None,
            # "!daend": None,
            # Context commands
            "@catalog": None,
            "@subject": None,
            # Internal commands
            ".help": None,
            ".exit": None,
            ".quit": None,
            ".clear": None,
            ".chat_info": None,
            ".compact": None,
            ".sessions": None,
            # temporary commands for sqlite, remove after mcp server is ready
            ".databases": None,
            ".database": None,
            ".tables": None,
            ".schemas": None,
            ".schema": None,
            ".table_schema": None,
            ".indexes": None,
            # ".show": None,
            ".namespace": None,
            ".mcp": None,
            ".subagent": None,
            ".subagent list": None,
            ".subagent add": None,
            ".subagent update": None,
            ".subagent remove": None,
            ".subagent bootstrap": None,
            ".bootstrap-bi": None,
        }

    def update_tables(self, tables: Dict[str, List[str]]):
        """
        Update the tables and columns information.

        Args:
            tables: Dictionary mapping table names to column lists
        """
        self.tables = tables
        # Reset aliases when tables are updated
        self.table_aliases = {}

    def update_db_info(self, database_name: str, schema_name: str):
        """
        Update database and schema context information.

        Args:
            database_name: Name of the current database
            schema_name: Name of the current schema
        """
        self.database_name = database_name
        self.schema_name = schema_name

    def _detect_aliases(self, text: str):
        """
        Detect table aliases in the SQL query.

        Args:
            text: SQL query text
        """
        # Simple regex would work for basic cases, but we'll use a basic split approach
        lines = text.split("\n")
        for line in lines:
            # Look for FROM and JOIN clauses with aliases
            if "FROM" in line.upper() or "JOIN" in line.upper():
                parts = line.split()
                for i in range(len(parts) - 2):
                    if parts[i].upper() in ("FROM", "JOIN") and i + 2 < len(parts):
                        table_name = parts[i + 1].strip(",")
                        # Check if next token is an alias
                        if parts[i + 2] not in (
                            "ON",
                            "WHERE",
                            "GROUP",
                            "ORDER",
                            "HAVING",
                            "LIMIT",
                            "OFFSET",
                            "JOIN",
                            "LEFT",
                            "RIGHT",
                        ):
                            alias = parts[i + 2].strip(",")
                            if table_name in self.tables:
                                self.table_aliases[alias] = table_name

        logger.debug(f"Detected aliases: {self.table_aliases}")

    def get_completions(self, document: Document, complete_event=None) -> Iterable[Completion]:
        """
        Get completions for the current cursor position.

        Args:
            document: The document to complete
            complete_event: Complete event (not used)

        Returns:
            Iterable of completions
        """
        text = document.text
        if text.startswith("/"):
            return
        text = document.text_before_cursor
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        logger.debug(f"Completion for: '{word_before_cursor}', text before: '{text}'")

        # First check for command completions
        if text.lstrip().startswith(("!", "@", ".")):
            cmd_text = text.lstrip()
            for cmd in self.commands:
                if cmd.startswith(cmd_text):
                    display = cmd
                    yield Completion(cmd, start_position=-len(cmd_text), display=display, style="class:command")
            return

        # Detect aliases in the current query
        self._detect_aliases(text)

        # Check if we're after a dot (schema.table or table.column)
        if "." in word_before_cursor:
            parts = word_before_cursor.split(".")
            if len(parts) >= 2:
                prefix = parts[0]
                # If prefix is a table name or alias, suggest columns
                if prefix in self.tables or prefix in self.table_aliases:
                    table = self.tables.get(prefix) or self.tables.get(self.table_aliases.get(prefix, ""))
                    if table:
                        for col in table:
                            if col.startswith(parts[-1]) or not parts[-1]:
                                yield Completion(
                                    col,
                                    start_position=-len(parts[-1]),
                                    display=col,
                                    style="class:column",
                                )
                return

        # Check for FROM/JOIN context to suggest tables
        prev_word = self._get_previous_word(text).upper()
        if prev_word in ["FROM", "JOIN", "TABLE"]:
            for table in self.tables:
                if table.startswith(word_before_cursor) or not word_before_cursor:
                    yield Completion(
                        table,
                        start_position=-len(word_before_cursor),
                        display=table,
                        style="class:table",
                    )
            return

        # Suggest columns in SELECT, WHERE, GROUP BY, etc. contexts
        if prev_word in ["SELECT", "WHERE", "ON", "BY", "HAVING", "ORDER", "SET", "UPDATE"]:
            # First suggest all column names from all tables and aliases
            for table, columns in self.tables.items():
                for col in columns:
                    if col.startswith(word_before_cursor) or not word_before_cursor:
                        yield Completion(
                            col,
                            start_position=-len(word_before_cursor),
                            display=f"{col} [{table}]",
                            style="class:column",
                        )

            # Then suggest qualified column names for tables and aliases
            for table in self.tables:
                if table.startswith(word_before_cursor) or not word_before_cursor:
                    yield Completion(
                        f"{table}.",
                        start_position=-len(word_before_cursor),
                        display=f"{table}.",
                        style="class:table",
                    )
            for alias, table in self.table_aliases.items():
                if alias.startswith(word_before_cursor) or not word_before_cursor:
                    yield Completion(
                        f"{alias}.",
                        start_position=-len(word_before_cursor),
                        display=f"{alias}. → {table}",
                        style="class:table",
                    )
            return

        # Suggest keywords and functions for other contexts
        if word_before_cursor:
            for keyword in self.keywords:
                if keyword.startswith(word_before_cursor.upper()):
                    yield Completion(
                        keyword,
                        start_position=-len(word_before_cursor),
                        display=keyword,
                        style="class:keyword",
                    )

            for func in self.functions:
                if func.startswith(word_before_cursor.upper()):
                    yield Completion(
                        f"{func}(",
                        start_position=-len(word_before_cursor),
                        display=f"{func}()",
                        style="class:function",
                    )

    def _get_previous_word(self, text: str) -> str:
        """
        Get the previous word in the text.

        Args:
            text: Text to analyze

        Returns:
            Previous word
        """
        text = text.strip()
        if not text:
            return ""

        words = text.split()
        if len(words) < 2:
            return ""

        return words[-2]


class CustomSqlLexer(SqlLexer):
    """Custom lexer extending SqlLexer for @references with space separator."""

    tokens = {
        "root": [
            (rf"@Table(?:\s+{REFERENCE_PATH_REGEX})?", Token.AtTables),
            (rf"@Metrics(?:\s+{REFERENCE_PATH_REGEX})?", Token.AtMetrics),
            (rf"@Sql(?:\s+{REFERENCE_PATH_REGEX})?", Token.AtReferenceSql),
            (r"@File(?:\s+[^\r\n@]+)?", Token.AtFiles),
        ]
        + SqlLexer.tokens["root"],
    }


class CustomPygmentsStyle(DefaultStyle):
    """Custom style for coloring the @ references."""

    styles = {
        Token.AtTables: "#00CED1 bold",  # Pink
        Token.AtMetrics: "#FFD700 bold",  # Gold
        Token.AtReferenceSql: "#32CD32 bold",  # Green
        Token.AtFiles: "ansiblue bold",  # Blue
    }


class DynamicAtReferenceCompleter(Completer):
    def __init__(self, max_completions=10, quote_leaf=False):
        self._data: Union[Dict[str, Any], List[str]] = {}
        self.flatten_data: Dict[str, Any] = {}
        self.max_level = 0
        self.max_completions = max_completions
        self.quote_leaf = quote_leaf

    def clear(self):
        self._data = {}
        self.max_level = 0

    def fuzzy_match(self, text: str) -> List[str]:
        text = text.strip().lower()
        if not text:
            return []
        result = []
        for k in self.flatten_data.keys():
            if text in k.lower():
                result.append(k)
                if len(result) == 5:
                    break
        return result

    @abstractmethod
    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        raise NotImplementedError()

    def reload_data(self):
        self._data = self.load_data()

    def get_data(self):
        if not self._data:
            self._data = self.load_data()
        return self._data

    def _format_leaf_for_completion(self, leaf: str) -> str:
        """Wrap final component in quotes when required."""
        if not self.quote_leaf or not leaf:
            return leaf
        trimmed = leaf.strip()
        if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2:
            return trimmed
        return f'"{trimmed}"'

    def format_path_for_completion(self, path: str) -> str:
        """Format full path when presenting completions."""
        if not self.quote_leaf or not path:
            return path
        segments = path.split(".")
        if not segments:
            return path
        segments[-1] = self._format_leaf_for_completion(segments[-1])
        return ".".join(segments)

    def get_completions(self, document, complete_event):
        """Provide completions for specified type

        Args:
            document: Current document object
            complete_event: Completion event
        """
        data = self.get_data()
        rest = document.text
        separator = "."
        levels = rest.split(separator)
        ends_with_sep = rest.endswith(separator)

        if ends_with_sep:
            prev_levels = levels[:-1]
            prefix = ""
            current_level = len(prev_levels) + 1
        else:
            prev_levels = levels[:-1]
            prefix = levels[-1] if levels else ""
            current_level = len(levels)
        if current_level > self.max_level:
            return
        current_dict = data
        for lvl in prev_levels:
            current_dict = current_dict.get(lvl, {})
            if not isinstance(current_dict, (dict, list)):
                return
        # Handle case where last level is a list
        prefix_for_match = prefix.strip().lower()
        if self.quote_leaf:
            prefix_for_match = prefix_for_match.lstrip('"')
        suggestions = [k for k in current_dict if k.lower().startswith(prefix_for_match)]
        # Smart filtering: show more items when user types more characters
        if len(prefix) >= 3:
            # User typed enough characters, can show more options
            effective_limit = min(self.max_completions + 5, len(suggestions))
        else:
            # User typed few characters, limit to avoid overwhelming
            effective_limit = self.max_completions

        is_last_level = current_level == self.max_level
        suggestions = sorted(suggestions)[:effective_limit]
        for s in suggestions:
            completion_text = s

            # The display text (what user sees in menu)
            display_text = s
            if not is_last_level:
                display_text = f"{s}."
            else:
                completion_text = self._format_leaf_for_completion(s)
                display_text = completion_text

            if is_last_level and isinstance(current_dict, dict) and s in current_dict and current_dict[s]:
                display_text = f"{display_text}: {current_dict[s]}"
                if len(display_text) > 30:
                    display_text = f"{display_text[:30]}..."

            yield Completion(completion_text, display=display_text, start_position=-len(prefix))


def insert_into_dict(data: Dict, keys: List[str], value: str) -> None:
    """Helper function to insert values into a nested dictionary based on keys."""
    temp = data
    for key in keys[:-1]:
        temp = temp.setdefault(key, {})
    temp.setdefault(keys[-1], []).append(value)


class TableCompleter(DynamicAtReferenceCompleter):
    """Dynamic completer specifically for tables and metrics"""

    def __init__(self, agent_config: AgentConfig, sqlite_show_db: bool = False):
        super().__init__()
        self.agent_config = agent_config
        self.sqlite_show_db = sqlite_show_db

    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        from datus.storage.schema_metadata.store import SchemaWithValueRAG

        storage = SchemaWithValueRAG(self.agent_config)
        try:
            schema_table = storage.search_all_schemas(
                # database_name=self.agent_config.current_database,
                select_fields=[
                    "catalog_name",
                    "database_name",
                    "schema_name",
                    "table_name",
                    "table_type",
                    "definition",
                    "identifier",
                ],
            )
        except Exception as e:
            logger.warning(f"Failed to load table data: {e}")
            schema_table = pyarrow.table([])
        logger.debug(f"Load table data for completer: {len(schema_table)}")
        if schema_table is None or schema_table.num_rows == 0:
            return []

        # Process schema table directly using pyarrow (no conversion to pylist)
        table_column = schema_table["table_name"]

        if self.agent_config.db_type == DBType.SQLITE and not self.sqlite_show_db:
            self.max_level = 1
            for table, definition, table_type in zip(
                table_column, schema_table["definition"], schema_table["table_type"]
            ):
                self.flatten_data[table.as_py()] = {
                    "table_name": table.as_py(),
                    "table_type": table_type.as_py(),
                    "definition": definition.as_py(),
                }
            return table_column.to_pylist()

        catalog_column = schema_table["catalog_name"]
        database_column = schema_table["database_name"]
        schema_column = schema_table["schema_name"]
        identifier_column = schema_table["identifier"]

        data: Dict[str, Any] = {}

        if DBType.support_catalog(self.agent_config.db_type) and catalog_column[0].as_py():
            if DBType.support_database(self.agent_config.db_type):
                if DBType.support_schema(self.agent_config.db_type):
                    # catalog -> database -> schema -> table
                    self.max_level = 4
                    # Catalog -> Database -> Schema -> Table structure
                    for catalog, database, schema, table, definition, table_type, identifier in zip(
                        catalog_column,
                        database_column,
                        schema_column,
                        table_column,
                        schema_table["definition"],
                        schema_table["table_type"],
                        identifier_column,
                    ):
                        insert_into_dict(data, [catalog.as_py(), database.as_py(), schema.as_py()], table.as_py())
                        self.flatten_data[f"{catalog}.{database}.{schema}.{table}"] = {
                            "identifier": identifier.as_py(),
                            "catalog_name": catalog.as_py(),
                            "database_name": database.as_py(),
                            "schema_name": schema.as_py(),
                            "table_name": table.as_py(),
                            "table_type": table_type.as_py(),
                            "definition": definition.as_py(),
                        }
                    return data
                else:
                    # catalog -> database -> table
                    self.max_level = 3
                    for catalog, database, table, definition, table_type, identifier in zip(
                        catalog_column,
                        database_column,
                        table_column,
                        schema_table["definition"],
                        schema_table["table_type"],
                        identifier_column,
                    ):
                        insert_into_dict(data, [catalog.as_py(), database.as_py()], table.as_py())
                        self.flatten_data[f"{catalog}.{database}.{table}"] = {
                            "identifier": identifier.as_py(),
                            "catalog_name": catalog.as_py(),
                            "database_name": database.as_py(),
                            "table_name": table.as_py(),
                            "table_type": table_type.as_py(),
                            "definition": definition.as_py(),
                        }
                    return data
            elif DBType.support_schema(self.agent_config.db_type):
                self.max_level = 3
                # catalog -> schema -> table
                for catalog, schema, table, definition, table_type, identifier in zip(
                    catalog_column,
                    schema_column,
                    table_column,
                    schema_table["definition"],
                    schema_table["table_type"],
                    identifier_column,
                ):
                    insert_into_dict(data, [catalog.as_py(), schema.as_py()], table.as_py())
                    self.flatten_data[f"{catalog}.{schema}.{table}"] = {
                        "identifier": identifier.as_py(),
                        "catalog_name": catalog.as_py(),
                        "schema_name": schema.as_py(),
                        "table_name": table.as_py(),
                        "table_type": table_type.as_py(),
                        "definition": definition.as_py(),
                    }

        if (DBType.support_database(self.agent_config.db_type) or self.sqlite_show_db) and database_column[0].as_py():
            if DBType.support_schema(self.agent_config.db_type) and schema_column[0].as_py():
                self.max_level = 3
                # Database -> Schema -> Table structure
                for database, schema, table, definition, table_type, identifier in zip(
                    database_column,
                    schema_column,
                    table_column,
                    schema_table["definition"],
                    schema_table["definition"],
                    identifier_column,
                ):
                    insert_into_dict(data, [database.as_py(), schema.as_py()], table.as_py())
                    self.flatten_data[f"{database}.{schema}.{table}"] = {
                        "identifier": identifier.as_py(),
                        "database_name": database.as_py(),
                        "schema_name": schema.as_py(),
                        "table_name": table.as_py(),
                        "table_type": table_type.as_py(),
                        "definition": definition.as_py(),
                    }
            else:
                self.max_level = 2
                # Database -> Table structure
                for database, table, definition, table_type, identifier in zip(
                    database_column,
                    table_column,
                    schema_table["definition"],
                    schema_table["table_type"],
                    identifier_column,
                ):
                    insert_into_dict(data, [database.as_py()], table.as_py())
                    self.flatten_data[f"{database}.{table}"] = {
                        "identifier": identifier.as_py(),
                        "database_name": database.as_py(),
                        "table_name": table.as_py(),
                        "table_type": table_type.as_py(),
                        "definition": definition.as_py(),
                    }
            return data

        if DBType.support_schema(self.agent_config.db_type):
            self.max_level = 2
            # schema -> table
            for schema, table, definition, table_type, identifier in zip(
                schema_column, table_column, schema_table["definition"], schema_table["table_type"], identifier_column
            ):
                insert_into_dict(data, [schema.as_py()], table.as_py())
                self.flatten_data[f"{schema}.{table}"] = {
                    "identifier": identifier.as_py(),
                    "schema_name": schema.as_py(),
                    "table_name": table.as_py(),
                    "table_type": table_type.as_py(),
                    "definition": definition.as_py(),
                }

        return data


def insert_into_dict_with_dict(data: Dict, keys: List[str], leaf_key: str, value: str) -> None:
    """Helper function to insert values into a nested dictionary based on keys."""
    temp = data
    for key in keys[:-1]:
        temp = temp.setdefault(key, {})
    temp.setdefault(keys[-1], {})[leaf_key] = value


class MetricsCompleter(DynamicAtReferenceCompleter):
    """Dynamic completer specifically for tables and metrics"""

    def __init__(self, agent_config: AgentConfig):
        super().__init__(quote_leaf=True)
        self.agent_config = agent_config
        self.max_level = 4

    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        from datus.storage.cache import get_storage_cache_instance

        storage = get_storage_cache_instance(self.agent_config).metric_storage()
        data = storage.search_all_metrics()

        result = {}
        for metric in data:
            subject_path = metric.get("subject_path", [])
            name = metric.get("name", "unknown")
            description = metric.get("description", "")

            # Build nested dict using subject_path
            if subject_path:
                insert_into_dict_with_dict(result, subject_path, name, description)

            # Flatten key uses "/" separator
            flatten_key = "/".join(subject_path + [name]) if subject_path else name
            self.flatten_data[flatten_key] = {
                "name": name,
                "description": description,
            }
        return result


class ReferenceSqlCompleter(DynamicAtReferenceCompleter):
    def __init__(self, agent_config: AgentConfig):
        super().__init__(quote_leaf=True)
        self.agent_config = agent_config

    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        self.max_level = 4

        from datus.storage.reference_sql.store import ReferenceSqlRAG

        storage = ReferenceSqlRAG(self.agent_config)
        search_data = storage.search_all_reference_sql()
        result = {}
        for item in search_data:
            subject_path = item.get("subject_path", [])
            name = item["name"]

            # Build nested dict using subject_path
            if subject_path:
                insert_into_dict_with_dict(result, subject_path, name, item["summary"])

            # Flatten key uses "/" separator
            flatten_key = "/".join(subject_path + [name]) if subject_path else name
            self.flatten_data[flatten_key] = {
                "name": name,
                "comment": item["comment"],
                "summary": item["summary"],
                "tags": item["tags"],
                "sql": item["sql"],
            }
        return result


class AtReferenceCompleter(Completer):
    """Router completer: dispatch to different completers based on type"""

    def __init__(self, agent_config: AgentConfig):
        # Initialize specialized completers
        self.parser = AtReferenceParser()
        self.table_completer = TableCompleter(agent_config)
        self.metric_completer = MetricsCompleter(agent_config)
        self.sql_completer = ReferenceSqlCompleter(agent_config)

        # Get workspace_root from chat node configuration or storage configuration
        workspace_root = None
        if hasattr(agent_config, "nodes") and "chat" in agent_config.nodes:
            chat_node = agent_config.nodes["chat"]
            if hasattr(chat_node, "input") and chat_node.input and hasattr(chat_node.input, "workspace_root"):
                workspace_root = chat_node.input.workspace_root

        # Also check storage configuration for workspace_root
        if not workspace_root and hasattr(agent_config, "workspace_root"):
            workspace_root = agent_config.workspace_root

        if not workspace_root:
            workspace_root = "."
        self.workspace_root = workspace_root

        def get_search_paths():
            paths = []
            # import os
            # paths = [os.getcwd()]
            if workspace_root:
                paths.insert(0, workspace_root)
            return paths

        self.file_completer = PathCompleter(get_paths=get_search_paths)

        self.completer_dict = {
            "Table": self.table_completer,
            "Metrics": self.metric_completer,
            "Sql": self.sql_completer,
            "File": self.file_completer,
        }
        self.type_options = {
            "Table": "📊 Table",
            "Metrics": "📈 Metrics",
            "Sql": "💻 Sql",
            "File": "📁 File",
        }

        self.at_parser = AtReferenceParser()

    def reload_data(self):
        self.table_completer.reload_data()
        self.metric_completer.reload_data()
        self.sql_completer.reload_data()

    def parse_at_context(self, user_input: str) -> Tuple[List[TableSchema], List[Metric], List[ReferenceSql]]:
        user_input = user_input.strip()
        if not user_input:
            return ([], [], [])
        parse_result = self.at_parser.parse_input(user_input)
        tables = []
        metrics = []
        sqls = []
        if parse_result["tables"]:
            for key in parse_result["tables"]:
                if key in self.table_completer.flatten_data:
                    tables.append(TableSchema.from_dict(self.table_completer.flatten_data[key]))

        if parse_result["metrics"]:
            for key in parse_result["metrics"]:
                if key in self.metric_completer.flatten_data:
                    metrics.append(Metric.from_dict(self.metric_completer.flatten_data[key]))
        if parse_result["sqls"]:
            for key in parse_result["sqls"]:
                if key in self.sql_completer.flatten_data:
                    sqls.append(ReferenceSql.from_dict(self.sql_completer.flatten_data[key]))
        return (tables, metrics, sqls)

    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        if not document.text.startswith("/"):
            return
        text = document.text_before_cursor
        at_pos = text.rfind("@")

        if at_pos == -1:
            return

        prefix = text[at_pos:]

        if " " not in prefix[1:]:
            # User is typing after @ without space, do fuzzy matching
            type_prefix = prefix[1:]

            if type_prefix:  # Only do fuzzy matching if there's text after @
                # Get fuzzy matches from each completer (max 5 each)
                table_matches = self.table_completer.fuzzy_match(type_prefix)
                metric_matches = self.metric_completer.fuzzy_match(type_prefix)
                sql_matches = self.sql_completer.fuzzy_match(type_prefix)
                file_matches = get_file_fuzzy_matches(type_prefix, path=self.workspace_root, max_matches=5)
                # Yield fuzzy match results first
                for match in table_matches[:5]:
                    # Extract the actual path from the match string
                    formatted = self.table_completer.format_path_for_completion(match)
                    display = f"📊 {formatted}"
                    yield Completion(
                        f"@Table {formatted}",  # Remove the @ from completion
                        start_position=-len(prefix),
                        display=display,
                        style="class:fuzzy",
                    )

                for match in metric_matches[:5]:
                    formatted = self.metric_completer.format_path_for_completion(match)
                    display = f"📈 {formatted}"
                    yield Completion(
                        f"@Metrics {formatted}", start_position=-len(prefix), display=display, style="class:fuzzy"
                    )

                for match in sql_matches[:5]:
                    formatted = self.sql_completer.format_path_for_completion(match)
                    display = f"💻 {formatted}"
                    yield Completion(
                        f"@Sql {formatted}", start_position=-len(prefix), display=display, style="class:fuzzy"
                    )

                for file_path in file_matches:
                    yield Completion(
                        f"@File {file_path}",  # Remove @ from completion
                        start_position=-len(prefix),
                        display=f"📁 {file_path}",
                        style="class:fuzzy",
                    )

            # Then yield type options that match
            type_prefix_lower = type_prefix.lower()
            for opt_text, opt_display in self.type_options.items():
                if opt_text.lower().startswith(type_prefix_lower):
                    yield Completion(
                        opt_text, start_position=-len(type_prefix), display=opt_display, style="class:type"
                    )
            return

        # Parse type and path
        type_part, rest = prefix[1:].split(" ", 1) if " " in prefix[1:] else (prefix[1:], "")
        type_ = type_part.strip()
        if type_ not in self.completer_dict:
            return

        # Create path document object
        from prompt_toolkit.document import Document

        path_document = Document(rest, len(rest))
        # Route to different completers based on type
        yield from self.completer_dict[type_].get_completions(path_document, complete_event)


class SubagentCompleter(Completer):
    """Completer for /subagent commands."""

    def __init__(self, agent_config: AgentConfig):
        """Initialize with agent configuration."""
        self.agent_config = agent_config
        self._available_subagents = self._load_subagents()

    def _load_subagents(self) -> List[str]:
        """Load available subagents from configuration and include built-in subagents."""
        subagents = list(SYS_SUB_AGENTS)
        if hasattr(self.agent_config, "agentic_nodes") and self.agent_config.agentic_nodes:
            for name in self.agent_config.agentic_nodes.keys():
                if name != "chat" and name not in SYS_SUB_AGENTS:  # Exclude default chat and avoid duplicates
                    subagents.append(name)
        return subagents

    def get_completions(self, document: Document, complete_event=None) -> Iterable[Completion]:
        """
        Get completions for subagent commands.

        Args:
            document: The document to complete
            complete_event: Complete event (not used)

        Returns:
            Iterable of completions
        """
        text = document.text_before_cursor

        # Only provide completions for slash commands
        if not text.startswith("/"):
            return

        # Get the text after the slash
        slash_content = text[1:]

        # If there's already a space, don't provide subagent completions
        if " " in slash_content:
            return

        # Generate completions for available subagents
        for subagent_name in self._available_subagents:
            if subagent_name.lower().startswith(slash_content.lower()) or not slash_content:
                # Choose emoji based on subagent name/type
                # We can add more if gen_metrics gen_table coder_revier added
                emoji = "🤖"
                if "chat" in subagent_name.lower():
                    emoji = "💬"
                elif "bot" in subagent_name.lower():
                    emoji = "🤖"

                display_text = f"{emoji} {subagent_name}"
                completion_text = f"{subagent_name} "  # Add space after subagent name

                yield Completion(
                    completion_text,
                    start_position=-len(slash_content),
                    display=display_text,
                    style="class:subagent",
                )


class AtReferenceParser:
    """
    Independent parser for extracting @Table, @Metrics, and @Sql references from text.
    This parser only extracts the reference paths, not the actual data.
    """

    def __init__(self):
        """Initialize the parser with regex patterns."""
        # Regular expressions for matching different types of references
        self.patterns = {
            "Table": re.compile(rf"@Table\s+({REFERENCE_PATH_REGEX})", re.IGNORECASE),
            "Metrics": re.compile(rf"@Metrics\s+({REFERENCE_PATH_REGEX})", re.IGNORECASE),
            "Sqls": re.compile(rf"@Sql\s+({REFERENCE_PATH_REGEX})", re.IGNORECASE),
        }

    def parse_input(self, text: str) -> Dict[str, List[str]]:
        """
        Parse text and extract all @reference paths.

        Args:
            text: Input text containing @references

        Returns:
            Dictionary with keys 'tables', 'metrics', 'reference_sql', 'files',
            each containing a list of extracted paths
        """
        results = {"tables": [], "metrics": [], "sqls": []}

        # Extract Table references
        for match in self.patterns["Table"].finditer(text):
            path = normalize_reference_path(match.group(1))
            if path:
                results["tables"].append(path)

        # Extract Metric references
        for match in self.patterns["Metrics"].finditer(text):
            path = normalize_reference_path(match.group(1))
            if path:
                results["metrics"].append(path)

        # Extract ReferenceSql references
        for match in self.patterns["Sqls"].finditer(text):
            path = normalize_reference_path(match.group(1))
            if path:
                results["sqls"].append(path)

        return results
