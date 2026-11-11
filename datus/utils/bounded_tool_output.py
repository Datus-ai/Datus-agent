"""Small wire contracts for tools whose data is delivered outside the model."""

import json
import re

DATA_TOOLS = frozenset({"export_query", "transfer_query_result"})
MAX_ERROR_BYTES = 2048
MAX_RESULT_BYTES = 16384


def bounded_text(value, limit=MAX_ERROR_BYTES):
    text = str(value)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return raw.decode("utf-8")
    return raw[: limit - 16].decode("utf-8", errors="ignore") + " [truncated]"


def error_summary(value):
    # Driver errors can embed statements/parameters even with bound writes.
    text = re.split(
        r"\[SQL:|\bSQL\s*[:=]|\[parameters:|\bparameters\s*:|\nLINE \d+:|\bINSERT\s+INTO\b",
        str(value),
        maxsplit=1,
        flags=re.I,
    )[0]
    return bounded_text(text.strip() or "Database operation failed; inspect the diagnostic file")


def bound_data_tool_result(result):
    """Keep valid JSON and essential status; never slice serialized JSON."""
    if not isinstance(result, dict):
        return {"success": 0, "error": error_summary(result), "result": None}
    result = dict(result)
    if result.get("error") is not None:
        result["error"] = error_summary(result["error"])
    raw = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
    if len(raw) <= MAX_RESULT_BYTES:
        return result
    body = result.get("result")
    if isinstance(body, dict):
        keep = {
            "file",
            "format",
            "row_count",
            "byte_count",
            "complete",
            "sql_ref",
            "sql_sha256",
            "source_sql_sha256",
            "source_datasource",
            "target_datasource",
            "target_table",
            "mode",
            "rows_transferred",
            "transferred_row_count",
            "source_row_count",
            "source_row_count_verified",
            "target_table_created",
            "batch_size",
            "deliverable_target",
            "scope",
            "query_rewritten",
            "executed_sql_sha256",
            "diagnostic_ref",
            "diagnostic_encoding",
            "error_code",
            "rows_written",
            "partial_write",
            "write_outcome",
            "retryable",
        }
        reduced = {key: value for key, value in body.items() if key in keep}
        reduced["metadata_truncated"] = True
        policies = body.get("applied_policies")
        reduced["applied_policy_count"] = len(policies) if isinstance(policies, (list, tuple)) else 0
        compact = {"success": result.get("success", 0), "error": result.get("error"), "result": reduced}
        if len(json.dumps(compact, ensure_ascii=False, default=str).encode()) <= MAX_RESULT_BYTES:
            return compact
    return {
        "success": 0,
        "error": "Tool metadata exceeded the output budget; inspect project diagnostic files before retrying",
        "result": {"error_code": "TOOL_OUTPUT_TOO_LARGE", "execution_outcome": "unknown", "retryable": False},
    }
