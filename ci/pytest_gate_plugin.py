"""Record collected and executed nodeids, including xdist skips and crashes."""

import json
import os
from pathlib import Path

import pytest

_collected: set[str] = set()
_executed: set[str] = set()
_outcomes: dict[str, str] = {}
_excluded: dict[str, str] = {}
_config = None


def pytest_addoption(parser):
    parser.addoption("--gate-report", help="Write machine-readable collection/execution evidence")
    parser.addoption(
        "--pr-gate", action="store_true", help="Keep deterministic integration and all affected unit tests"
    )


def pytest_sessionstart(session):
    global _config
    _config = session.config
    _collected.clear()
    _executed.clear()
    _outcomes.clear()
    _excluded.clear()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--pr-gate"):
        return
    from ci.harness.pr_test_plan import nightly_pr_contracts
    from ci.harness.test_plan import covers

    contracts = nightly_pr_contracts()
    selected, deselected = [], []
    for item in items:
        marks = {mark.name for mark in item.iter_markers()}
        integration = item.nodeid.startswith("tests/integration/")
        deterministic = bool(marks & {"acceptance", "component", "llm_harness"})
        nightly = "nightly" in marks and not any(covers(ref, item.nodeid) for ref in contracts)
        if "quarantine" in marks or nightly or (integration and not deterministic):
            deselected.append(item)
            _excluded[item.nodeid] = (
                "quarantine" if "quarantine" in marks else "nightly" if nightly else "integration-not-deterministic"
            )
        else:
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_deselected(items):
    # pytest's -m filter runs before our final selection hook.
    for item in items:
        if item.config.getoption("--pr-gate") and item.get_closest_marker("quarantine"):
            _excluded[item.nodeid] = "quarantine"


def pytest_collection_finish(session):
    _collected.update(item.nodeid for item in session.items)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    _collected.update(ids)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    _excluded.update(getattr(node, "workeroutput", {}).get("gate_excluded", {}))


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    if report.when == "call":
        _executed.add(report.nodeid)
    # pytest-rerunfailures changes a failed report's outcome to "rerun".
    # Treat it as the original failure, even if a diagnostic retry later passes.
    failed = report.failed or report.outcome == "rerun"
    if failed:
        _outcomes[report.nodeid] = "failed"
    elif report.when == "call" or report.skipped:
        previous = _outcomes.get(report.nodeid)
        if previous not in {"failed", "skipped"}:
            _outcomes[report.nodeid] = "xfailed" if getattr(report, "wasxfail", None) else report.outcome
    if failed and _config is not None and not hasattr(_config, "workerinput"):
        reporter = _config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", f"Immediate failure: {report.nodeid} [{report.when}]")
            # Pytest already formatted longrepr with the selected traceback and
            # locals options, including reports transported back from xdist.
            report.toterminal(reporter._tw)
            reporter._tw.flush()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["gate_excluded"] = _excluded
    target = session.config.getoption("--gate-report")
    if target and not hasattr(session.config, "workerinput"):
        if session.exitstatus == pytest.ExitCode.OK and "failed" in _outcomes.values():
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        identity = {"source_id": os.environ["DATUS_GATE_SOURCE_ID"]} if os.environ.get("DATUS_GATE_SOURCE_ID") else {}
        Path(target).write_text(
            json.dumps(
                {
                    "collected": sorted(_collected),
                    "executed": sorted(_executed),
                    "outcomes": _outcomes,
                    "excluded": _excluded,
                    "exit_code": int(session.exitstatus),
                    **identity,
                },
                indent=2,
            )
            + "\n"
        )
