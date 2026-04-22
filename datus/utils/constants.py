# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from enum import Enum


class DBType(str, Enum):
    """Built-in database dialect types (zero or minimal dependencies).

    External dialects (mysql, postgresql, snowflake, etc.) are registered
    dynamically via adapter packages and the ConnectorRegistry.
    """

    SQLITE = "sqlite"
    DUCKDB = "duckdb"


class LLMProvider(str, Enum):
    """Large Language Model provider types supported by Datus."""

    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    KIMI = "kimi"  # Moonshot Kimi models
    ANTHROPIC = "anthropic"  # Alternative name for Claude
    GEMINI = "gemini"
    MINIMAX = "minimax"
    GLM = "glm"
    LLAMA = "llama"
    GPT = "gpt"  # Alternative name for OpenAI
    CODEX = "codex"  # OpenAI Codex (ChatGPT subscription, OAuth authentication)
    OPENROUTER = "openrouter"  # OpenRouter unified AI gateway


class EmbeddingProvider(str, Enum):
    """Embedding model provider types supported by Datus."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    FASTEMBED = "fastembed"
    HUGGINGFACE = "huggingface"


# System sub-agents that are built-in and not user-configurable
SYS_SUB_AGENTS = {
    "gen_semantic_model",
    "gen_metrics",
    "gen_sql_summary",
    "gen_ext_knowledge",
    "gen_sql",
    "gen_report",
    "gen_table",
    "gen_job",
    "gen_skill",
    "gen_dashboard",
    "scheduler",
    "feedback",
}

# Subset of SYS_SUB_AGENTS that stays invokable via "/<name> ..." but is hidden
# from user-facing listings (autocomplete, ".agent" selector). Meta/internal
# agents that the user shouldn't see as a routing choice belong here.
HIDDEN_SYS_SUB_AGENTS = {"feedback"}


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
