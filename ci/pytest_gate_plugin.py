"""Record collected and executed nodeids, including xdist skips and crashes."""

import json
import os
from pathlib import Path

import pytest

_collected: set[str] = set()
_outcomes: dict[str, str] = {}


def pytest_addoption(parser):
    parser.addoption("--gate-report", help="Write machine-readable collection/execution evidence")
    parser.addoption(
        "--pr-gate", action="store_true", help="Keep deterministic integration and all affected unit tests"
    )


def pytest_sessionstart(session):
    _collected.clear()
    _outcomes.clear()


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
        else:
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session):
    _collected.update(item.nodeid for item in session.items)


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    _collected.update(ids)


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed or report.skipped:
        previous = _outcomes.get(report.nodeid)
        if previous not in {"failed", "skipped"}:
            _outcomes[report.nodeid] = "xfailed" if getattr(report, "wasxfail", None) else report.outcome


def pytest_sessionfinish(session, exitstatus):
    target = session.config.getoption("--gate-report")
    if target and not hasattr(session.config, "workerinput"):
        identity = {"source_id": os.environ["DATUS_GATE_SOURCE_ID"]} if os.environ.get("DATUS_GATE_SOURCE_ID") else {}
        Path(target).write_text(
            json.dumps(
                {"collected": sorted(_collected), "outcomes": _outcomes, "exit_code": int(exitstatus), **identity},
                indent=2,
            )
            + "\n"
        )
