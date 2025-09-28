from enum import Enum


class DBType(str, Enum):
    """SQL database dialect types supported by Datus."""

    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    POSTGRES = "postgres"  # Alternative name for PostgreSQL
    SNOWFLAKE = "snowflake"
    CLICKHOUSE = "clickhouse"
    BIGQUERY = "bigquery"
    STARROCKS = "starrocks"
    SQLSERVER = "sqlserver"  # same as mssql
    MSSQL = "mssql"  # same as sqlserver
    ORACLE = "oracle"

    @classmethod
    def support_catalog(cls, db_type: str) -> bool:
        # bigquery support project as catalog
        return db_type in SUPPORT_CATALOG_DIALECTS

    @classmethod
    def support_database(cls, db_type: str) -> bool:
        return db_type in SUPPORT_DATABASE_DIALECTS

    @classmethod
    def support_schema(cls, db_type: str) -> bool:
        return db_type in SUPPORT_SCHEMA_DIALECTS


SUPPORT_CATALOG_DIALECTS = {DBType.STARROCKS, DBType.SNOWFLAKE, DBType.BIGQUERY}
SUPPORT_DATABASE_DIALECTS = {
    DBType.STARROCKS,
    DBType.SNOWFLAKE,
    DBType.BIGQUERY,
    DBType.MYSQL,
    DBType.MSSQL,
    DBType.SQLSERVER,
    DBType.ORACLE,
    DBType.POSTGRES,
    DBType.POSTGRESQL,
    DBType.DUCKDB,
}
SUPPORT_SCHEMA_DIALECTS = {
    DBType.SNOWFLAKE,
    DBType.BIGQUERY,
    DBType.MSSQL,
    DBType.SQLSERVER,
    DBType.ORACLE,
    DBType.DUCKDB,
    DBType.POSTGRES,
    DBType.POSTGRESQL,
}


class LLMProvider(str, Enum):
    """Large Language Model provider types supported by Datus."""

    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    ANTHROPIC = "anthropic"  # Alternative name for Claude
    GEMINI = "gemini"
    LLAMA = "llama"
    GPT = "gpt"  # Alternative name for OpenAI


class EmbeddingProvider(str, Enum):
    """Embedding model provider types supported by Datus."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    HUGGINGFACE = "huggingface"


class SQLType(str, Enum):
    """SQL statement types."""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    DDL = "ddl"
    METADATA_SHOW = "metadata"
    EXPLAIN = "explain"
    CONTENT_SET = "context_set"
    UNKNOWN = "unknown"
