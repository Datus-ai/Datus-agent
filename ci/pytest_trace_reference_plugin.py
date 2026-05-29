"""Record nightly pytest case trace references.

This plugin is intentionally inert unless ``DATUS_NIGHTLY_TRACE_REFERENCES_FILE``
is set. It is loaded by the nightly runner so trace references can be joined
back to pytest nodeids after the run.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    item.stash[_trace_before_key] = _current_trace_metadata()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    if os.environ.get("DATUS_TEST_LAYER") != "nightly":
        return

    output_file = os.environ.get("DATUS_NIGHTLY_TRACE_REFERENCES_FILE", "")
    if not output_file:
        return

    current = _current_trace_metadata()
    before = item.stash.get(_trace_before_key, None)
    has_new_trace = bool(current and current != before)
    trace_expected = _truthy(os.environ.get("DATUS_NIGHTLY_TRACE_EXPECTED", ""))

    if not has_new_trace and not trace_expected:
        return

    row: dict[str, Any] = {
        "suite": os.environ.get("DATUS_NIGHTLY_SUITE_NAME", ""),
        "nodeid": item.nodeid,
        "outcome": report.outcome,
        "duration_seconds": getattr(report, "duration", None),
        "trace_expected": trace_expected,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if has_new_trace and current:
        row.update(current)

    _append_jsonl(Path(output_file), row)


_trace_before_key: pytest.StashKey[dict[str, str] | None] = pytest.StashKey()


def _current_trace_metadata() -> dict[str, str] | None:
    try:
        from datus.utils.traceable_utils import get_trace_reference

        ref = get_trace_reference()
    except Exception:
        return None
    if ref is None:
        return None
    return ref.to_metadata()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        warnings.warn(
            f"Failed to write nightly trace reference to {path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
