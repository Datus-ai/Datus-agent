"""Guard the alignment between harness marks and the PR acceptance target list.

The PR harness runs two suites with complementary mark filters: the acceptance
suite runs only files in ``PR_ACCEPTANCE_TARGETS`` with harness marks, while the
impacted suite excludes harness-marked tests. A harness-marked test in a file
outside the target list therefore never runs in PR or merge-queue CI. These
tests fail the acceptance suite on the offending PR itself.

Detection is AST-based (decorators, ``pytestmark``, ``pytest.param`` marks), so
marks applied dynamically from conftest hooks are not seen; keep harness marks
literal in the test files.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "ci" / "run-pr-tests.py"
MODULE_SPEC = importlib.util.spec_from_file_location("run_pr_tests_for_alignment", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise AssertionError(f"Unable to load run-pr-tests from {MODULE_PATH}")
run_pr_tests = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(run_pr_tests)

HARNESS_MARKS = ("acceptance", "component", "llm_harness")


def _harness_mark_names(path: Path) -> set[str]:
    """Return harness mark names used as ``pytest.mark.<name>`` expressions in the file.

    String occurrences (e.g. mark names inside test fixtures' source snippets) do
    not count — only real ``pytest.mark.X`` attribute accesses in the AST.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not any(mark in text for mark in HARNESS_MARKS):
        return set()

    found: set[str] = set()
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if not isinstance(node, ast.Attribute) or node.attr not in HARNESS_MARKS:
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "mark"
            and isinstance(value.value, ast.Name)
            and value.value.id == "pytest"
        ):
            found.add(node.attr)
    return found


def _all_test_files() -> list[Path]:
    return sorted((REPO_ROOT / "tests").rglob("test_*.py"))


def test_acceptance_targets_exist():
    missing = [target for target in run_pr_tests.PR_ACCEPTANCE_TARGETS if not (REPO_ROOT / target).exists()]
    assert missing == [], (
        f"Stale PR_ACCEPTANCE_TARGETS entries (file deleted or renamed): {missing}. "
        "Update the list in ci/run-pr-tests.py."
    )


def test_harness_marked_files_are_acceptance_targets():
    targets = set(run_pr_tests.PR_ACCEPTANCE_TARGETS)
    orphaned = [
        str(path.relative_to(REPO_ROOT))
        for path in _all_test_files()
        if _harness_mark_names(path) and str(path.relative_to(REPO_ROOT)) not in targets
    ]
    assert orphaned == [], (
        f"Files with acceptance/component/llm_harness marks missing from PR_ACCEPTANCE_TARGETS: {orphaned}. "
        "The acceptance suite skips them (not listed) and the impacted suite excludes them (marked), "
        "so they never run in PR or merge-queue CI. Add them to the list in ci/run-pr-tests.py."
    )


def test_acceptance_targets_collect_harness_marked_tests():
    dead_weight = [
        target for target in run_pr_tests.PR_ACCEPTANCE_TARGETS if not _harness_mark_names(REPO_ROOT / target)
    ]
    assert dead_weight == [], (
        f"PR_ACCEPTANCE_TARGETS entries without any harness mark: {dead_weight}. "
        "The acceptance suite collects zero tests from them; mark the acceptance-worthy "
        "tests or drop the entry from ci/run-pr-tests.py."
    )
