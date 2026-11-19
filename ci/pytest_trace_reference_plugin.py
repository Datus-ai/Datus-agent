"""Record nightly pytest case trace references.

This plugin is intentionally inert unless ``DATUS_NIGHTLY_TRACE_REFERENCES_FILE``
is set. It is loaded by the nightly runner so trace references can be joined
back to pytest nodeids after the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    case = item.stash.get(_bi_case_key, {})
    if case:
        from ci.bi_diagnostics import trace_rows

        case["outcome"] = report.outcome
        path = _write_bi_case(case)
        captured = trace_rows(case)
        row["trace_expected"] = trace_expected and bool(case.get("runs"))
        if captured:
            for reference in captured:
                _append_jsonl(Path(output_file), {**row, **reference, "evidence_file": str(path)})
            return
        row["evidence_file"] = str(path)
    if has_new_trace and current:
        row.update(current)
    _append_jsonl(Path(output_file), row)


_bi_case_key: pytest.StashKey[dict] = pytest.StashKey()


@pytest.fixture(autouse=True)
def _capture_bi_execution(request, monkeypatch):
    if not os.environ.get("DATUS_NIGHTLY_BI_EVIDENCE_DIR") or not any(
        name in request.node.nodeid for name in ("test_gen_dashboard_agentic.py", "test_bi_chart_lifecycle.py")
    ):
        return
    from ci.bi_diagnostics import install_capture

    case = {
        "nodeid": request.node.nodeid,
        "suite": os.environ.get("DATUS_NIGHTLY_SUITE_NAME", ""),
        "outcome": "running",
    }
    request.node.stash[_bi_case_key] = case
    install_capture(monkeypatch, case, checkpoint=lambda: _write_bi_case(case))
    _write_bi_case(case)


def _write_bi_case(case: dict) -> Path:
    directory = Path(os.environ["DATUS_NIGHTLY_BI_EVIDENCE_DIR"])
    path = directory / (hashlib.sha256(case["nodeid"].encode()).hexdigest()[:16] + ".json")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(case, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix=path.stem, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(data)
        temporary.replace(path)
    except (OSError, RuntimeError) as exc:
        warnings.warn(f"Failed to write BI execution evidence: {exc}", RuntimeWarning, stacklevel=2)
    return path


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
