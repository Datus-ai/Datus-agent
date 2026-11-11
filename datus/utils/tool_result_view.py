# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Recover pre-pagination archive references without changing current tool outputs."""

from typing import Any

from datus.utils.tool_result_store import ToolResultStore, json_text


def restore_legacy_tool_results(items: list[Any], store: ToolResultStore | None) -> list[Any]:
    """Resolve old saved references while preserving raw history and media blocks."""
    if store is None:
        return items
    recover = store.recover_legacy
    result = []
    for item in items:
        replacement = item
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            raw = item.get("output")
            if isinstance(raw, (str, dict)):
                value = recover(raw)
                if value is not raw:
                    replacement = {**item, "output": json_text(value)}
            elif isinstance(raw, list):
                blocks = []
                for block in raw:
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
                        text = block.get("text", "")
                        value = recover(text)
                        if value is not text:
                            block = {**block, "text": json_text(value)}
                    blocks.append(block)
                if any(new is not old for new, old in zip(blocks, raw, strict=True)):
                    replacement = {**item, "output": blocks}
        elif isinstance(item, dict) and isinstance(item.get("content"), list):
            blocks = []
            changed = False
            for block in item["content"]:
                updated = block
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    raw = block.get("content")
                    if isinstance(raw, str):
                        value = recover(raw)
                        if value is not raw:
                            updated = {**block, "content": json_text(value)}
                    elif isinstance(raw, list):
                        content = []
                        for part in raw:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                value = recover(text)
                                if value is not text:
                                    part = {**part, "text": value if isinstance(value, str) else json_text(value)}
                            content.append(part)
                        if any(new is not old for new, old in zip(content, raw, strict=True)):
                            updated = {**block, "content": content}
                blocks.append(updated)
                changed |= updated is not block
            if changed:
                replacement = {**item, "content": blocks}
        result.append(replacement)
    return result if any(new is not old for new, old in zip(result, items, strict=True)) else items
