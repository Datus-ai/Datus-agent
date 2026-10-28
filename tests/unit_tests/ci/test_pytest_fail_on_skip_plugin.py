from types import SimpleNamespace

from pytest import ExitCode

from ci.pytest_fail_on_skip_plugin import pytest_sessionfinish


class _Reporter:
    def __init__(self, skipped):
        self.stats = {"skipped": skipped}
        self.messages = []

    def write_sep(self, separator, message):
        self.messages.append((separator, message))


def _session(*, enabled: bool, skipped: list) -> SimpleNamespace:
    reporter = _Reporter(skipped)
    pluginmanager = SimpleNamespace(get_plugin=lambda _name: reporter)
    config = SimpleNamespace(
        getoption=lambda _name: enabled,
        pluginmanager=pluginmanager,
    )
    return SimpleNamespace(config=config, exitstatus=ExitCode.OK, reporter=reporter)


def test_fail_on_skip_changes_successful_session_to_failure():
    session = _session(enabled=True, skipped=[object()])

    pytest_sessionfinish(session, ExitCode.OK)

    assert session.exitstatus == ExitCode.TESTS_FAILED
    assert session.reporter.messages == [("=", "FAIL-ON-SKIP: 1 selected test(s) skipped")]


def test_fail_on_skip_is_opt_in():
    session = _session(enabled=False, skipped=[object()])

    pytest_sessionfinish(session, ExitCode.OK)

    assert session.exitstatus == ExitCode.OK
    assert session.reporter.messages == []
