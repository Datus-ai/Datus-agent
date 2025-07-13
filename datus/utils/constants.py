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
    SQLSERVER = "sqlserver"
    MSSQL = "mssql"
    ORACLE = "oracle"


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
