# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Read bounded pages from this session's already-executed tool results."""

from typing import Callable, Optional

from datus.tools.func_tool.base import FuncToolResult, trans_to_function_tool
from datus.utils.tool_result_store import ToolResultStore


class ToolResultFuncTool:
    permission_category = "tool_result_tools"

    def __init__(self, get_store: Callable[[], Optional[ToolResultStore]]):
        self._get_store = get_store

    def available_tools(self):
        return [trans_to_function_tool(self.read_tool_result)]

    def read_tool_result(
        self, result_id: str, section: str = "", cursor: Optional[str] = None, limit: int = 100
    ) -> FuncToolResult:
        """Read the next bounded page of a saved tool result without rerunning it.

        Copy result_id, section and next_cursor from the previous page. A
        record_ref section reads one oversized record in text segments. An
        empty section selects the default; section="raw" reads the original
        serialized output. has_more only covers captured records;
        source_truncated/source_has_more/source_complete describe the source.
        This tool cannot read another session's results or arbitrary paths.
        """
        store = self._get_store()
        if store is None:
            return FuncToolResult(success=0, error="No result archive is available for this session")
        try:
            return FuncToolResult(**store.read(result_id, section, cursor, limit))
        except (OSError, ValueError, TypeError) as exc:
            return FuncToolResult(success=0, error=str(exc))
