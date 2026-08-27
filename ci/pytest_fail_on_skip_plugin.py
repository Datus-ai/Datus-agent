"""Make an explicitly strict pytest suite fail when any test is skipped."""

from __future__ import annotations

from pytest import ExitCode


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail the pytest session when one or more selected tests are skipped.",
    )


def pytest_sessionfinish(session, exitstatus) -> None:
    if not session.config.getoption("--fail-on-skip"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter is not None else []
    if not skipped:
        return
    if reporter is not None:
        reporter.write_sep("=", f"FAIL-ON-SKIP: {len(skipped)} selected test(s) skipped")
    if exitstatus == ExitCode.OK:
        session.exitstatus = ExitCode.TESTS_FAILED
