# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""SSE content formatter for IM output with verbose-level control."""

from typing import Callable, Dict, Optional

from datus.gateway.models import Verbose

# Maximum length for any single value in full-mode output
_MAX_VALUE_LEN = 2000

# SQL-like keys whose values should be wrapped in code blocks
_SQL_KEYS = {"sql", "query", "definition"}


class ToolOutputFormatter:
    """Format SSE tool call/result payloads for IM output."""

    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._register_builtins()

    def _register_builtins(self):
        self._registry["read_query"] = self._format_read_query_result
        self._registry["describe_table"] = self._format_describe_table_result
        self._registry["search_table"] = self._format_search_table_result

    def format_tool_complete(
        self,
        call_payload: dict,
        result_payload: dict,
        verbose: Verbose,
    ) -> Optional[str]:
        """Merge call-tool + call-tool-result into a single IM text.

        call_payload:   {"callToolId": ..., "toolName": ..., "toolParams": {...}}
        result_payload: {"callToolId": ..., "toolName": ..., "duration": ..., "shortDesc": ..., "result": ...}

        Returns None if verbose is OFF.
        """
        if verbose == Verbose.OFF:
            return None

        tool_name = result_payload.get("toolName", call_payload.get("toolName", "unknown"))
        duration = result_payload.get("duration", 0)

        line = f"\U0001f527 **{tool_name}** \u2705 ({duration:.1f}s)"

        if verbose == Verbose.FULL:
            params = call_payload.get("toolParams", {})
            if params:
                line += "\n" + _format_params(params)

            result = result_payload.get("result")
            if result is not None:
                formatted = self._format_result(result, tool_name)
                if formatted:
                    line += "\n" + formatted

        return line

    def _format_result(self, result, tool_name: str) -> str:
        """Format a tool result, using a registered formatter if available."""
        formatter = self._registry.get(tool_name)
        if formatter:
            return formatter(result)
        return _format_result_default(result)

    @staticmethod
    def _format_read_query_result(result) -> str:
        if not isinstance(result, dict):
            return _format_result_default(result)
        rows = result.get("rows", result.get("data", []))
        row_count = len(rows) if isinstance(rows, list) else 0
        columns = result.get("columns", [])
        parts = []
        if columns:
            parts.append(f"> columns: {len(columns)}")
        parts.append(f"> rows: {row_count}")
        if isinstance(rows, list) and rows:
            preview = _truncate(str(rows[:3]), _MAX_VALUE_LEN)
            parts.append(f"> preview: {preview}")
        return "\n".join(parts) if parts else _format_result_default(result)

    @staticmethod
    def _format_describe_table_result(result) -> str:
        if not isinstance(result, dict):
            return _format_result_default(result)
        columns = result.get("columns", result.get("fields", []))
        col_count = len(columns) if isinstance(columns, list) else 0
        return f"> columns: {col_count}"

    @staticmethod
    def _format_search_table_result(result) -> str:
        if not isinstance(result, dict):
            return _format_result_default(result)
        metadata = result.get("metadata", [])
        sample_data = result.get("sample_data", result.get("samples", []))
        meta_count = len(metadata) if isinstance(metadata, list) else 0
        sample_count = len(sample_data) if isinstance(sample_data, list) else 0
        return f"> metadata: {meta_count}, sample rows: {sample_count}"


def _format_params(params: dict) -> str:
    """Format tool parameters for full-mode display."""
    lines = []
    for key, value in params.items():
        str_value = _truncate(str(value), _MAX_VALUE_LEN)
        if key in _SQL_KEYS:
            lines.append(f"> {key}:\n```\n{str_value}\n```")
        else:
            lines.append(f"> {key}: {str_value}")
    return "\n".join(lines)


def _format_result_default(result) -> str:
    """Default formatter for tool results."""
    if isinstance(result, dict):
        lines = []
        for key, value in result.items():
            if isinstance(value, list):
                lines.append(f"> {key}: [{len(value)} items]")
            elif isinstance(value, dict):
                lines.append(f"> {key}: " + _truncate(str(value), _MAX_VALUE_LEN))
            else:
                lines.append(f"> {key}: " + _truncate(str(value), _MAX_VALUE_LEN))
        return "\n".join(lines)
    if isinstance(result, list):
        return f"> [{len(result)} items]"
    return _truncate(str(result), _MAX_VALUE_LEN)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
