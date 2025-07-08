"""
Autocomplete module for Datus CLI.
Provides SQL keyword, table name, and column name autocompletion.
"""

from typing import Dict, Iterable, List

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from datus.utils.loggings import get_logger

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

    def _get_command_completions(self) -> Dict:
        """Get a nested completer for command completions."""
        return {
            # Tool commands
            "!": None,
            "!darun": None,
            "!dastart": None,
            "!sl": None,
            "!gen": None,
            "!run": None,
            "!fix": None,
            "!rf": None,
            "!reason": None,
            "!save2file": None,
            "!savesql": None,
            "!bash": None,
            "!daend": None,
            # Context commands
            "@": None,
            "@catalogs": None,
            "@tables": None,
            "@metrics": None,
            # Internal commands
            ".help": None,
            ".exit": None,
            ".quit": None,
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
