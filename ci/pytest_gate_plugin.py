"""Record collected and executed nodeids, including xdist skips and crashes."""

import json
from pathlib import Path

import pytest

_collected: set[str] = set()
_outcomes: dict[str, str] = {}


def pytest_addoption(parser):
    parser.addoption("--gate-report", help="Write machine-readable collection/execution evidence")


def pytest_sessionstart(session):
    _collected.clear()
    _outcomes.clear()


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
        Path(target).write_text(
            json.dumps({"collected": sorted(_collected), "outcomes": _outcomes, "exit_code": int(exitstatus)}, indent=2)
            + "\n"
        )
