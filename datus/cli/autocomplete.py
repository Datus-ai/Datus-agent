"""
Autocomplete module for Datus CLI.
Provides SQL keyword, table name, and column name autocompletion.
"""
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Union

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from pygments.lexers.sql import SqlLexer
from pygments.styles.default import DefaultStyle
from pygments.token import Token

from datus.configuration.agent_config import AgentConfig
from datus.utils.constants import DBType
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
        self.at_cmds = ["table", "metric"]

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
            (r"@Table(?:\s[^ \t\n]+)?", Token.AtTables),
            (r"@Metric(?:\s[^ \t\n]+)?", Token.AtMetrics),
            (r"@File(?:\s[^ \t\n]+)?", Token.AtFiles),
        ]
        + SqlLexer.tokens["root"],
    }


class CustomPygmentsStyle(DefaultStyle):
    """Custom style for coloring the @ references."""

    styles = {
        Token.AtTables: "#00CED1 bold",  # pink
        Token.AtMetrics: "#FFD700 bold",  # Green
        Token.AtFiles: "ansiblue bold",  # Blue
    }


class DynamicAtReferenceCompleter(Completer):
    def __init__(self, max_completions=10):
        self._data: Union[Dict[str, Any], List[str]] = {}
        self.max_level = 0
        self.max_completions = max_completions

    def clear(self):
        self._data = {}
        self.max_level = 0

    @abstractmethod
    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        raise NotImplementedError

    def get_data(self):
        if not self._data:
            self._data = self.load_data()
        return self._data

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
        prefix_lower = prefix.lower()
        suggestions = [k for k in current_dict if k.lower().startswith(prefix_lower)]
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

            if is_last_level and isinstance(current_dict, dict) and s in current_dict and current_dict[s]:
                display_text = f"{display_text}: {current_dict[s]}"
                if len(display_text) > 20:
                    display_text = f"{display_text[:20]}..."

            yield Completion(completion_text, display=display_text, start_position=-len(prefix))


class TableCompleter(DynamicAtReferenceCompleter):
    """Dynamic completer specifically for tables and metrics"""

    def __init__(self, agent_config: AgentConfig):
        super().__init__()
        self.agent_config = agent_config

    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        from datus.storage.schema_metadata.store import rag_by_configuration

        storage = rag_by_configuration(self.agent_config)
        schema_table = storage.search_all_schemas(
            database_name=self.agent_config.current_database,
            select_fields=["catalog_name", "database_name", "schema_name", "table_name"],
        )
        if schema_table is None or schema_table.num_rows == 0:
            return []

        schema_data = schema_table.to_pylist()

        if self.agent_config.db_type == DBType.SQLITE:
            self.max_level = 1
            return [item["table_name"] for item in schema_data]

        data: Dict[str, Any] = {}

        if DBType.support_catalog(self.agent_config.db_type) and schema_data[0]["catalog_name"]:
            if DBType.support_database(self.agent_config.db_type):
                if DBType.support_schema(self.agent_config.db_type):
                    # catalog -> database -> schema -> table
                    self.max_level = 4
                    for item in schema_data:
                        catalog_name = item["catalog_name"]
                        if catalog_name not in data:
                            data[catalog_name] = {}
                        if item["database_name"] not in data[catalog_name]:
                            data[catalog_name][item["database_name"]] = {}
                        if item["schema_name"] not in data[catalog_name][item["database_name"]]:
                            data[catalog_name][item["database_name"]][item["schema_name"]] = []
                        data[catalog_name][item["database_name"]][item["schema_name"]].append(item["table_name"])
                else:
                    # catalog -> database -> table
                    self.max_level = 3
                    for item in schema_data:
                        catalog_name = item["catalog_name"]
                        if catalog_name not in data:
                            data[catalog_name] = {}
                        if item["database_name"] not in data[catalog_name]:
                            data[catalog_name][item["database_name"]] = []
                        data[catalog_name][item["database_name"]].append(item["table_name"])
            elif DBType.support_schema(self.agent_config.db_type):
                self.max_level = 3
                # catalog -> schema -> table
                for item in schema_data:
                    catalog_name = item["catalog_name"]
                    if catalog_name not in data:
                        data[catalog_name] = {}
                    if item["schema_name"] not in data[catalog_name]:
                        data[catalog_name][item["schema_name"]] = []
                    data[catalog_name][item["schema_name"]].append(item["table_name"])

        if DBType.support_database(self.agent_config.db_type) and schema_data[0]["database_name"]:
            if DBType.support_schema(self.agent_config.db_type) and schema_data[0]["schema_name"]:
                self.max_level = 3
                for item in schema_data:
                    if item["database_name"] not in data:
                        data[item["database_name"]] = {}
                    if item["schema_name"] not in data[item["database_name"]]:
                        data[item["database_name"]][item["schema_name"]] = []
                    data[item["database_name"]][item["schema_name"]].append(item["table_name"])
            else:
                self.max_level = 2
                for item in schema_data:
                    if item["database_name"] not in data:
                        data[item["database_name"]] = []
                    data[item["database_name"]].append(item["table_name"])
            return data

        if DBType.support_schema(self.agent_config.db_type):
            self.max_level = 2
            for item in schema_data:
                if item["schema_name"] not in data:
                    data[item["schema_name"]] = []
                data[item["schema_name"]].append(item["table_name"])

        return data


class MetricsCompleter(DynamicAtReferenceCompleter):
    """Dynamic completer specifically for tables and metrics"""

    def __init__(self, agent_config: AgentConfig):
        super().__init__()
        self.agent_config = agent_config
        self.max_level = 4

    def load_data(self) -> Union[List[str], Dict[str, Any]]:
        from datus.storage.metric.store import rag_by_configuration

        storage = rag_by_configuration(self.agent_config).metric_storage
        data = storage.search_all(select_fields=["domain", "layer1", "layer2", "name", "description"])
        from collections import defaultdict

        result = defaultdict(dict)
        for i in range(data.num_rows):
            (
                result.get(data["domain"][i])
                .get(data["layer1"][i])
                .get(data["layer2"][i])
                .put(data["name"][i], data["description"][i])
            )
        return result


class AtReferenceCompleter(Completer):
    """Router completer: dispatch to different completers based on type"""

    def __init__(self, agent_config: AgentConfig):
        # Initialize specialized completers
        self.table_completer = TableCompleter(agent_config)
        self.metric_completer = MetricsCompleter(agent_config)
        from prompt_toolkit.completion import PathCompleter

        self.completer_dict = {
            "Table": self.table_completer,
            "Metric": self.metric_completer,
            "File": PathCompleter(),
        }
        self.type_options = {
            "Table ": "📊 Table",
            "Metric ": "📈 Metric",
            "File ": "📁 File",
        }

    def reload_data(self):
        self.table_completer.clear()
        self.table_completer.load_data()
        self.metric_completer.clear()
        self.metric_completer.load_data()

    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        if not document.text.startswith("/"):
            return
        text = document.text_before_cursor
        at_pos = text.rfind("@")

        if at_pos == -1:
            return

        prefix = text[at_pos:]

        # Limit number of spaces
        if prefix.count(" ") > 1:
            return

        if " " not in prefix[1:]:
            # Complete type
            type_prefix = prefix[1:]
            type_prefix = type_prefix.lower()
            for opt_text, opt_display in self.type_options.items():
                if opt_text.lower().startswith(type_prefix):
                    yield Completion(opt_text, start_position=-len(type_prefix), display=opt_display)
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
