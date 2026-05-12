from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "audit_tests.py"


def _load_audit_tests():
    module_spec = importlib.util.spec_from_file_location("audit_tests", MODULE_PATH)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError(f"Unable to load audit_tests from {MODULE_PATH}")
    audit_tests = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = audit_tests
    module_spec.loader.exec_module(audit_tests)
    return audit_tests


def test_audit_flags_asyncio_run_in_integration_test(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "integration" / "test_asyncio_run.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
import asyncio

async def do_work():
    return 1

def test_nested_loop_smell():
    result = asyncio.run(do_work())
    assert result == 1
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert any(issue.check == "asyncio_run_in_integration" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_flags_nightly_marker_in_unit_test(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_marker.py"
        test_file.parent.mkdir(parents=True)
        nightly_marker = "@pytest.mark." + "nightly"
        test_file.write_text(
            f"""
import pytest

{nightly_marker}
def test_component_case():
    assert 1 == 1
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert any(issue.check == "nightly_marker_in_unit" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_flags_multiline_pytestmark_nightly_in_unit_test(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_multiline_marker.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
import pytest

pytestmark = [
    pytest.mark.component,
    pytest.mark.nightly,
]

def test_component_case():
    assert 1 == 1
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        nightly_issues = [issue for issue in issues if issue.check == "nightly_marker_in_unit"]
        assert len(nightly_issues) == 1
        assert nightly_issues[0].line == 6
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_does_not_flag_large_unit_test_file(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_large_file.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            "def test_large_file_still_scans():\n    assert 1 == 1\n" + "\n".join("# filler" for _ in range(1800)),
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert all(issue.check != "file_size_budget" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)
