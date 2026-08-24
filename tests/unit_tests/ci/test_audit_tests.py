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


def test_audit_flags_pseudo_regex_alternation_in_not_in(tmp_path):
    """`"A|B|C" not in s` is literal containment, so it holds for almost any s."""
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_pseudo_regex.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
def test_markers_stay_out():
    notes = load_notes()
    assert "WITH S3|HDFS|BROKER" not in notes
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        flagged = [issue for issue in issues if issue.check == "regex_literal_containment"]
        assert len(flagged) == 1
        assert flagged[0].severity == "P1"
        assert flagged[0].line == 4
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_flags_explicit_regex_tokens_in_not_in(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_regex_tokens.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            r"""
def test_no_word_boundary_needle():
    rendered = render()
    assert "\bDROP\b" not in rendered
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        flagged = [issue for issue in issues if issue.check == "regex_literal_containment"]
        assert len(flagged) == 1
        assert flagged[0].severity == "P1"
        assert flagged[0].line == 4
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_flags_multi_escape_and_group_regex_needles(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_regex_shapes.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            r"""
def test_regex_shaped_needles():
    rendered = render()
    assert r"DROP\s+TABLE\s+users" not in rendered
    assert "suffix(?:_v2)" not in rendered
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        flagged = [issue for issue in issues if issue.check == "regex_literal_containment"]
        assert [issue.line for issue in flagged] == [4, 5]
        assert {issue.severity for issue in flagged} == {"P1"}
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_does_not_flag_windows_path_needles(tmp_path):
    """Backslash sequences in path literals are not regex intent."""
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_windows_paths.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            r"""
def test_path_needles_are_literal():
    output = render()
    assert r"C:\build" not in output
    assert r"D:/state\dump" not in output
    assert r"\\server\share\logs" not in output
    assert r"src\build" not in output
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert all(issue.check != "regex_literal_containment" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_does_not_flag_legitimate_pipe_needles(tmp_path):
    """Markdown tables, spaced pipes, SQL concat, and `in` direction stay clean."""
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_legit_pipes.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
def test_pipe_shapes_that_are_really_literals():
    output = render()
    assert "| Option |" not in output
    assert "left | right" not in output
    assert "a || b" not in output
    assert "csv|field" in output
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert all(issue.check != "regex_literal_containment" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_flags_case_contradictory_containment(tmp_path):
    """A lowercase needle can never appear in an `.upper()` container."""
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_case_blind.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
def test_marker_stays_out():
    notes = load_notes()
    assert "broker load" not in notes.upper()
    assert "LOAD LABEL" in notes.lower()
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        flagged = [issue for issue in issues if issue.check == "case_contradictory_containment"]
        assert [issue.line for issue in flagged] == [4, 5]
        assert {issue.severity for issue in flagged} == {"P0"}
    finally:
        audit_tests.configure_repo_root(original_root)


def test_audit_does_not_flag_case_consistent_containment(tmp_path):
    audit_tests = _load_audit_tests()
    original_root = audit_tests.REPO_ROOT
    try:
        test_file = tmp_path / "tests" / "unit_tests" / "test_case_consistent.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """
def test_marker_stays_out():
    notes = load_notes()
    assert "BROKER LOAD" not in notes.upper()
    assert "load label" not in notes.lower()
    assert "Broker Load" not in notes
""",
            encoding="utf-8",
        )
        audit_tests.configure_repo_root(tmp_path)

        issues = audit_tests.scan_file(test_file, required_packages=set())

        assert all(issue.check != "case_contradictory_containment" for issue in issues)
    finally:
        audit_tests.configure_repo_root(original_root)
