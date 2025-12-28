# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import sys
import traceback
from enum import Enum
from typing import Any, Optional

from datus.utils.loggings import get_log_manager, get_logger

logger = get_logger(__name__)


class ErrorCode(Enum):
    """Error codes with descriptions for Datus exceptions."""

    # Common errors
    COMMON_UNKNOWN = ("1000000", "Unknown error occurred")
    COMMON_FIELD_INVALID = (
        "1000001",
        "Unexcepted value of {field_name}, excepted value: {except_values}, your value: {your_value}",
    )
    COMMON_FILE_NOT_FOUND = ("100002", "{config_name} file not found: {file_name}")
    COMMON_FIELD_REQUIRED = ("100003", "Missing required field: {field_name}")
    COMMON_UNSUPPORTED = ("100004", "Unsupported value `{your_value}` for field `{field_name}`")
    COMMON_ENV = ("100005", "The environment variable {env_var} is not set")
    COMMON_CONFIG_ERROR = ("100006", "Configuration error: {config_error}")
    COMMON_MISSING_DEPENDENCY = ("100007", "Missing node dependency")
    COMMON_VALIDATION_FAILED = ("100008", "Data validation failed")
    COMMON_JSON_PARSE_ERROR = ("100009", "JSON parsing error in file '{file_path}': {error_detail}")
    COMMON_TEMPLATE_NOT_FOUND = ("100010", "Template not found: '{template_name}' with version '{version}'")

    # Node execution errors
    NODE_EXECUTION_FAILED = ("200001", "Node execution failed")
    NODE_NO_SQL_CONTEXT = ("200002", "No SQL context available. Please run a SQL generation node first.")

    # Model errors
    MODEL_REQUEST_FAILED = ("300001", "LLM request failed")
    MODEL_INVALID_RESPONSE = ("300002", "Invalid request format, content, or model response (HTTP 400)")
    MODEL_TIMEOUT = ("300003", "Model request timeout")

    # API errors following Anthropic/OpenAI standards
    MODEL_AUTHENTICATION_ERROR = ("300011", "Authentication failed - check your API key (HTTP 401)")
    MODEL_PERMISSION_ERROR = ("300012", "API key lacks required permissions (HTTP 403)")
    MODEL_NOT_FOUND = ("300013", "Requested resource not found (HTTP 404)")
    MODEL_REQUEST_TOO_LARGE = ("300014", "Request exceeds size limit (HTTP 413)")
    MODEL_RATE_LIMIT = ("300015", "Rate limit exceeded - please wait before retrying (HTTP 429)")
    MODEL_API_ERROR = ("300016", "Unexpected API internal error (HTTP 500)")
    MODEL_OVERLOADED = ("300017", "API temporarily overloaded - please try again later (HTTP 529)")
    MODEL_CONNECTION_ERROR = ("300018", "Connection error - check your network connection")
    MODEL_EMBEDDING_ERROR = ("300019", "Embedding Model error")
    MODEL_QUOTA_EXCEEDED = ("300020", "Usage quota exceeded - please check your billing plan")
    MODEL_TIMEOUT_ERROR = ("300021", "Request timeout - the API took too long to respond")
    MODEL_MAX_TURNS_EXCEEDED = ("300022", "Maximum turns ({max_turns}) exceeded - agent execution stopped")
    MODEL_ILLEGAL_FORMAT_RESPONSE = (
        "300023",
        "Model returned response in illegal format. Response: '{response_preview}' (length: {response_length})",
    )
    # Tool errors
    TOOL_EXECUTION_FAILED = ("400001", "Tool execution failed")
    TOOL_INVALID_INPUT = ("400002", "Invalid tool input")

    # Storage errors - Vector Database Operations
    STORAGE_FAILED = ("410000", "Vector database operation failed: {error_message}")
    STORAGE_CONNECTION_FAILED = ("410001", "Failed to connect to vector database at path: {storage_path}")
    STORAGE_TABLE_OPERATION_FAILED = (
        "410002",
        "Vector database table operation failed: {operation} on collection '{table_name}' failed with {error_message}",
    )
    STORAGE_SAVE_FAILED = ("410003", "Failed to save data to vector database: {error_message}")
    STORAGE_SEARCH_FAILED = (
        "410004",
        (
            "Vector database search operation failed: {error_message}. "
            "Query vector: '{query}', Filter: '{where_clause}', Limit: {top_n}"
        ),
    )
    STORAGE_INDEX_FAILED = ("410005", "Vector database index operation failed: {error_message}")

    # Database errors
    DB_FAILED = ("500000", "Database operation failed. Error details: {error_message}")
    # Database errors - Connection (common SQLAlchemy exceptions)
    DB_CONNECTION_FAILED = ("500001", "Failed to establish connection to database. Error details: {error_message}")
    DB_CONNECTION_TIMEOUT = ("500002", "Connection to database timed out. Error details: {error_message}")
    DB_AUTHENTICATION_FAILED = (
        "500003",
        "Authentication failed for database. Please check your credentials. Error details: {error_message}",
    )
    DB_PERMISSION_DENIED = (
        "500004",
        "Permission denied when performing '{operation}' on database. Error details: {error_message}",
    )

    # Database errors - Query Execution (SQLAlchemy + Snowflake specific)
    DB_EXECUTION_SYNTAX_ERROR = (
        "500005",
        "Invalid SQL syntax in query. Error details: {error_message}",
    )
    DB_EXECUTION_ERROR = (
        "500006",
        "Failed to execute query on database. Error details: {error_message}",
    )
    DB_EXECUTION_TIMEOUT = (
        "500007",
        "Query execution timed out on database. Error details: {error_message}",
    )
    DB_QUERY_METADATA_FAILED = (
        "500008",
        "Failed to retrieve metadata for query. Error details: {error_message}",
    )

    # Database errors - Constraints (SQLAlchemy IntegrityError)
    DB_CONSTRAINT_VIOLATION = (
        "500008",
        "Database constraint violation occurred. Error details: {error_message}",
    )

    # Database errors - Transaction (SQLAlchemy transaction issues)
    DB_TRANSACTION_FAILED = ("500009", "Database transaction failed. Error details: {error_message}")

    DB_TABLE_NOT_EXISTS = ("500010", "Table {table_name} does not exist.")

    # Semantic adapter errors
    SEMANTIC_ADAPTER_NOT_FOUND = ("600001", "Semantic adapter '{adapter_type}' not found or not installed")
    SEMANTIC_ADAPTER_ERROR = ("600002", "Semantic adapter operation failed: {error_message}")
    SEMANTIC_ADAPTER_CONFIG_ERROR = ("600003", "Semantic adapter configuration error: {error_message}")
    SEMANTIC_ADAPTER_SYNC_FAILED = ("600004", "Failed to sync from semantic adapter: {error_message}")

    def __init__(self, code: str, desc: str):
        self.code = code
        self.desc = desc


class DatusException(Exception):
    """Datus custom exceptions for standardized printing

    Args:
        code: ErrorCode - The error code enum that defines the type and category of the exception
        message: Optional[str] - Custom error message. If not provided, uses the default message from ErrorCode
        message_args: Optional[dict[str, Any]] - Arguments to format the error message template from ErrorCode
        *args: object - Additional arguments passed to the base Exception class
        Exception (_type_): _description_
    """

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str] = None,
        message_args: Optional[dict[str, Any]] = None,
        *args: object,
    ):
        self.code = code
        self.message_args = message_args or {}
        self.message = self.build_msg(message, message_args)
        super().__init__(self.message, *args)

    def __str__(self):
        return self.message

    def build_msg(self, message: Optional[str] = None, message_args: Optional[dict[str, Any]] = None) -> str:
        if message:
            final_message = message
        elif message_args:
            final_message = self.code.desc.format(**message_args)
        else:
            final_message = self.code.desc
        return f"error_code={self.code.code}, error_message={final_message}"


def setup_exception_handler(console_logger=None, prefix_wrap_func=None):
    """Setup global exception handler for Datus

    Args:
        console_logger (function, optional): If provided, print exception message to console.
    """

    def global_exception_handler(type, value, tb):
        if issubclass(type, (SystemExit, KeyboardInterrupt, GeneratorExit)):
            # Do not catch these exceptions, let the program exit or respond to the interrupt
            sys.__excepthook__(type, value, traceback)
            return

        # Print exception
        format_ex = "\n".join(traceback.format_exception(type, value, tb))
        log_prefix = (
            "Execution failed" if type == DatusException or issubclass(type, DatusException) else "Unexpected failed"
        )
        log_manager = get_log_manager()
        if log_manager.debug:
            logger.error(f"{log_prefix}: {format_ex}")
            if console_logger:
                console_log(console_logger, log_prefix, format_ex, prefix_wrap_func)
        else:
            if console_logger:
                # print exception trace to file
                logger.error(f"{log_prefix}: {format_ex}")
                console_log(
                    console_logger,
                    log_prefix,
                    str(value) if not hasattr(value, "message") else value.message,
                    prefix_wrap_func,
                )
            else:
                # print exception trace to file
                with log_manager.temporary_output("file"):
                    logger.error(f"{log_prefix}: {format_ex}")
                # print exception message to console
                with log_manager.temporary_output("console"):
                    message = str(value) if not hasattr(value, "message") else value.message
                    logger.error(f"{log_prefix}: {message}")

    def console_log(console_logger, log_prefix, error_msg: str, prefix_wrap_func=None):
        if prefix_wrap_func:
            console_logger(f"{prefix_wrap_func(log_prefix)}: {error_msg}")
        else:
            console_logger(f"{log_prefix}: {error_msg}")

    sys.excepthook = global_exception_handler
