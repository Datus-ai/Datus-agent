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
    COMMON_UNSUPPORTED = ("100004", "Unsupported value {your_value} for {field_name}")
    COMMON_ENV = ("100005", "The environment variable {env_var} is not set")
    COMMON_CONFIG_ERROR = ("100006", "Configuration error: {config_error}")
    COMMON_MISSING_DEPENDENCY = ("100007", "Missing node dependency")

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
    # ...
    # Tool errors
    TOOL_EXECUTION_FAILED = ("400001", "Tool execution failed")
    TOOL_INVALID_INPUT = ("400002", "Invalid tool input")
    TOOL_STORE_FAILED = ("400003", "Store failed")
    TOOL_DB_FAILED = ("400004", "Database {operation} failed, uri={uri}, error_message={error_message}")
    TOOL_DB_EXECUTE_QUERY_FAILED = (
        "400005",
        'Database execute failed, uri={uri}, sql="{sql}", error_message={error_message}',
    )
    # ...

    # Validation errors
    VALIDATION_FAILED = ("500001", "Data validation failed")
    # ...

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
        self.message = self.build_msg(message, message_args)
        super().__init__(self.message, *args)

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
            "Execution failed" if type == DatusException or issubclass(type, DatusException) else "Unexcepted failed"
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
                console_log(console_logger, log_prefix, value.message, prefix_wrap_func)
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
