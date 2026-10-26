# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Model-only copies of tool outputs; raw results and action events stay intact."""

from typing import Any, Callable

from datus.models.request_capacity import ContextCapacityError
from datus.utils.image_content import contains_images
from datus.utils.tool_result_store import ToolResultStore, json_text


def project_tool_results(
    items: list[Any],
    store: ToolResultStore | None,
    *,
    force: bool = False,
    counter: Callable[[Any], int] | None = None,
) -> list[Any]:
    def project(call_id: str, raw: Any) -> Any:
        if store is not None:
            return store.project(call_id, raw, force=force)
        if contains_images(raw):
            return raw
        size = len(json_text(raw).encode("utf-8"))
        if force or size > 16384 or (counter is not None and counter(raw) > 4096):
            raise ContextCapacityError("RESULT_ARCHIVE_FAILED", "No writable session archive is available.")
        return raw

    result = []
    for item in items:
        replacement = item
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            raw = item.get("output")
            if isinstance(raw, (str, dict)):
                value = project(item.get("call_id", ""), raw)
                if value is not raw:
                    replacement = {**item, "output": json_text(value)}
            elif isinstance(raw, list):
                blocks = []
                for index, block in enumerate(raw):
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
                        text = block.get("text", "")
                        value = project(f"{item.get('call_id', '')}:{index}", text)
                        if value is not text:
                            block = {**block, "text": json_text(value)}
                    blocks.append(block)
                if any(new is not old for new, old in zip(blocks, raw, strict=True)):
                    replacement = {**item, "output": blocks}
                if store is not None:
                    store.release(item.get("call_id", ""))
        elif isinstance(item, dict) and isinstance(item.get("content"), list):
            blocks = []
            changed = False
            for block in item["content"]:
                updated = block
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    raw = block.get("content")
                    if isinstance(raw, str):
                        value = project(block.get("tool_use_id", ""), raw)
                        if value is not raw:
                            updated = {**block, "content": json_text(value)}
                    elif isinstance(raw, list):
                        content = []
                        for index, part in enumerate(raw):
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                value = project(f"{block.get('tool_use_id', '')}:{index}", text)
                                if value is not text:
                                    part = {**part, "text": value if isinstance(value, str) else json_text(value)}
                            content.append(part)
                        if any(new is not old for new, old in zip(content, raw, strict=True)):
                            updated = {**block, "content": content}
                        if store is not None:
                            store.release(block.get("tool_use_id", ""))
                blocks.append(updated)
                changed |= updated is not block
            if changed:
                replacement = {**item, "content": blocks}
        result.append(replacement)
    return result if any(new is not old for new, old in zip(result, items, strict=True)) else items
