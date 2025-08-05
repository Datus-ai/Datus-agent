from enum import Enum


class McpType(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAM_HTTP = "http"  # streamable http
