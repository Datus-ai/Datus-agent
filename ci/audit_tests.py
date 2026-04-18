#!/usr/bin/env python3
"""Static test-quality audit for tests/unit_tests/** and tests/integration/**.

Enforces industry-standard test engineering practices via AST + regex checks.
Rules derive from:
  - xUnit Test Patterns (Meszaros) — test smell catalog
  - Clean Code F.I.R.S.T. (Uncle Bob) — Fast / Independent / Repeatable / Self-validating / Timely
  - Google Testing Blog — Hermetic tests, flakiness policy
  - Working Effectively with Unit Tests (Khorikov) — behavior-over-execution, mock-at-boundary

Tier-aware: unit-tests (`tests/unit_tests/`) enforce hermetic rules strictly;
integration-tests (`tests/integration/`) relax hermeticity but add boundary-
safety rules (no real `rmtree`, no `patch('builtins.open')`, no hardcoded
localhost ports, no sleep-based readiness, etc).

Usage:
    python ci/audit_tests.py --diff-only origin/main    # incremental: P0+P1 hard fail
    python ci/audit_tests.py --all                      # full scan: P0 hard fail, P1 warn
    python ci/audit_tests.py --paths tests/unit_tests/tools/skill_tools/
    python ci/audit_tests.py --json ci/audit-report.json
    python ci/audit_tests.py --tier unit --all          # only audit unit tier

Exit codes:
    0 - no blocking issues
    1 - blocking issues found (P0 always blocks; P1 blocks only in --diff-only / --paths mode)

Outputs (when --github-output and $GITHUB_OUTPUT is set):
    audit_outcome  - "success" or "failure"
    issue_total    - total issue count
    p0_count       - P0 issue count
    p1_count       - P1 issue count
    scan_mode      - "diff" or "full" or "paths"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "datus"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Tier roots — each maps to a distinct rule set.
TIER_ROOTS: dict[str, Path] = {
    "unit": REPO_ROOT / "tests" / "unit_tests",
    "integration": REPO_ROOT / "tests" / "integration",
}
UNIT_ROOT = TIER_ROOTS["unit"]
INTEGRATION_ROOT = TIER_ROOTS["integration"]
# Back-compat: TESTS_ROOT used by missing_test_for_module / build_test_class_index.
# These are unit-tier-specific checks.
TESTS_ROOT = UNIT_ROOT


def configure_repo_root(new_root: Path) -> None:
    """Override the repo root used by all scans (--repo-root CLI flag).

    The script itself lives under `base/ci/` in the workflow's trusted
    checkout. When invoked with `--repo-root .` from the untrusted PR
    checkout, REPO_ROOT / TIER_ROOTS / etc must point at the PR's tree so
    git diff and file reads actually see the PR's code. This keeps the
    trust boundary clean: the _code_ runs from `base/` (never imports from
    `pr/`), while the _data_ scanned lives at the given root.
    """
    global REPO_ROOT, SOURCE_ROOT, PYPROJECT, TIER_ROOTS, UNIT_ROOT, INTEGRATION_ROOT, TESTS_ROOT
    REPO_ROOT = new_root.resolve()
    SOURCE_ROOT = REPO_ROOT / "datus"
    PYPROJECT = REPO_ROOT / "pyproject.toml"
    TIER_ROOTS = {
        "unit": REPO_ROOT / "tests" / "unit_tests",
        "integration": REPO_ROOT / "tests" / "integration",
    }
    UNIT_ROOT = TIER_ROOTS["unit"]
    INTEGRATION_ROOT = TIER_ROOTS["integration"]
    TESTS_ROOT = UNIT_ROOT


FILE_SIZE_LIMIT = 1500
AUDIT_NOQA = "audit-noqa"


# ---------------------------------------------------------------------------
# Rule → tier applicability matrix.
#
# Every check name used in _emit(Issue(check=...)) must be registered here.
# _emit silently drops an issue whose check is not applicable to the file's
# tier. This lets us enforce e.g. hermetic rules only on unit tests, and
# boundary-safety rules only on integration tests.
# ---------------------------------------------------------------------------
RULE_TIERS: dict[str, set[str]] = {
    # Universal rules (behavior verification; apply to every test tier).
    "conditional_assert": {"unit", "integration"},
    "tautology": {"unit", "integration"},
    "zero_assert_test": {"unit", "integration"},
    "module_level_sys_modules": {"unit", "integration"},
    "debug_leftover": {"unit", "integration"},
    "importorskip_on_required": {"unit", "integration"},
    "try_except_skip": {"unit", "integration"},
    "weak_assert": {"unit", "integration"},
    "or_assert": {"unit", "integration"},
    "lambda_throw": {"unit", "integration"},
    "duplicate_test_files": {"unit", "integration"},
    # Unit-tier-only (strict hermeticity / tight size budget).
    "real_external_io_path": {"unit"},
    "file_size_budget": {"unit"},
    "missing_test_for_module": {"unit"},
    # Integration-tier-only (boundary safety + flakiness prevention).
    "rmtree_outside_tmp": {"integration"},
    "patch_builtins_open": {"integration"},
    "hardcoded_localhost": {"integration"},
    "sleep_based_wait": {"integration"},
    "try_except_pass_in_test": {"integration"},
    "empty_test_subdir": {"integration"},
}


def rule_applies(check: str, tier: str | None) -> bool:
    """Return True if a rule applies to a file of the given tier."""
    if tier is None:
        return False
    return tier in RULE_TIERS.get(check, set())


def detect_tier(path: Path) -> str | None:
    """Return 'unit' / 'integration' / None based on path location."""
    try:
        abs_path = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    except OSError:
        return None
    for tier, root in TIER_ROOTS.items():
        try:
            abs_path.relative_to(root)
            return tier
        except ValueError:
            continue
    return None


def log(msg: str) -> None:
    print(f"[audit] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    file: str
    line: int
    severity: str  # "P0" or "P1"
    check: str
    message: str
    quote: str = ""
    suggestion: str = ""
    tier: str = ""  # "unit" / "integration" / "" (for cross-file findings)


# ---------------------------------------------------------------------------
# AST-based checks
# ---------------------------------------------------------------------------


class _AstChecker(ast.NodeVisitor):
    """Walks a test file once, emitting issues for every AST-based check
    that applies to the file's tier."""

    def __init__(self, path: Path, source: str, tier: str):
        self.path = str(path.relative_to(REPO_ROOT))
        self.source_lines = source.splitlines()
        self.tier = tier
        self.issues: list[Issue] = []
        # Track enclosing function for zero_assert check
        self._func_stack: list[ast.AST] = []
        # Track if we're inside a test function or pytest fixture body
        self._in_test_or_fixture: list[bool] = [False]
        # If nodes already classified as part of an elif chain at the head —
        # prevents double-reporting when generic_visit descends into them.
        self._handled_elif_ifs: set[int] = set()

    # -- helpers

    def _quote(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()[:120]
        return ""

    def _has_noqa(self, lineno: int, rule: str) -> bool:
        text = self._quote(lineno)
        return f"# {AUDIT_NOQA}: {rule}" in text or f"# {AUDIT_NOQA}" in text

    def _emit(self, issue: Issue) -> None:
        # Tier filter — only emit if rule applies to this file's tier.
        if not rule_applies(issue.check, self.tier):
            return
        if self._has_noqa(issue.line, issue.check):
            return
        # Fill in tier if not set.
        if not issue.tier:
            issue = Issue(**{**asdict(issue), "tier": self.tier})
        self.issues.append(issue)

    # -- conditional_assert: if X: assert Y   (without a symmetric else-branch assertion)
    # -- try_except_skip: try ... except: pytest.skip()
    # -- try_except_pass_in_test: try ... except: pass (integration only)

    def visit_If(self, node: ast.If) -> None:
        # conditional_assert (top-level-branch heuristic, no recursion):
        # flag iff the if/elif/else chain has at least one branch that
        # directly verifies AND at least one branch that is empty (missing
        # else, or Pass/...-only), AND no branches are "opaque" (complex
        # structured code we deliberately don't evaluate). This keeps the
        # rule short and predictable: it catches the classic smells and
        # stays silent on anything needing real control-flow analysis.
        if id(node) in self._handled_elif_ifs:
            self.generic_visit(node)
            return

        branches, has_else, chained = _flatten_if_chain(node)
        self._handled_elif_ifs.update(id(n) for n in chained)

        labels = [_classify_branch(body) for body in branches]
        if not has_else:
            # Implicit "no branch matched" fall-through path is unverified.
            labels.append("empty")

        has_verified = "verified" in labels
        has_empty = "empty" in labels
        has_opaque = "opaque" in labels

        if has_verified and has_empty and not has_opaque:
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="conditional_assert",
                    message="Conditional branches verify behavior only on some execution paths (Meszaros: Conditional Test Logic)",
                    quote=self._quote(node.lineno),
                    suggestion="Mirror verification across all branches, or restructure so the assertion runs unconditionally.",
                )
            )

        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            for stmt in handler.body:
                # try/except + pytest.skip()
                call = _maybe_skip_call(stmt)
                if call is not None:
                    self._emit(
                        Issue(
                            file=self.path,
                            line=call.lineno,
                            severity="P0",
                            check="try_except_skip",
                            message="try/except + pytest.skip silently turns failures into skips (Meszaros: Lying Test)",
                            quote=self._quote(call.lineno),
                            suggestion="Let the exception propagate, or use pytest.raises / pytest.mark.skipif with an explicit condition.",
                        )
                    )
            # try/except + pass (integration only — swallows cleanup / teardown errors)
            if (
                self._in_test_or_fixture[-1]
                and len(handler.body) == 1
                and isinstance(handler.body[0], (ast.Pass, ast.Expr))
                and (
                    isinstance(handler.body[0], ast.Pass)
                    or (
                        isinstance(handler.body[0], ast.Expr)
                        and isinstance(handler.body[0].value, ast.Constant)
                        and handler.body[0].value.value is Ellipsis
                    )
                )
            ):
                self._emit(
                    Issue(
                        file=self.path,
                        line=handler.lineno,
                        severity="P1",
                        check="try_except_pass_in_test",
                        message="try/except: pass inside test/fixture swallows diagnostic info — failures become invisible",
                        quote=self._quote(handler.lineno),
                        suggestion="Let the exception propagate, or catch a specific type and log the details.",
                    )
                )
        self.generic_visit(node)

    # -- tautology / or_assert / weak_assert

    def visit_Assert(self, node: ast.Assert) -> None:
        if _is_tautological(node.test):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="tautology",
                    message="Assertion is always true — does not verify any behavior (Meszaros: Lying Test)",
                    quote=self._quote(node.lineno),
                    suggestion="Replace with a specific expected-value assertion.",
                )
            )
        elif _is_or_assert(node.test):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P1",
                    check="or_assert",
                    message="`assert A or B` accepts two outcomes — contract is ambiguous (Meszaros: Obscure Test)",
                    quote=self._quote(node.lineno),
                    suggestion="Pick exactly one expected value, or split into two separate tests.",
                )
            )
        elif _is_weak_assert(node.test):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P1",
                    check="weak_assert",
                    message="Weak assertion (`is not None`, `len > 0`, bare truthiness) — passes for almost any value",
                    quote=self._quote(node.lineno),
                    suggestion="Assert the exact expected value, e.g. `== 'foo'` or `== 3`.",
                )
            )
        self.generic_visit(node)

    # -- zero_assert_test / fixture / test function tracking

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)
        self.generic_visit(node)
        self._exit_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_function(node)
        self.generic_visit(node)
        self._exit_function()

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_test_function(node)
        is_test_or_fixture = node.name.startswith("test_") or _has_pytest_fixture_decorator(node)
        self._in_test_or_fixture.append(is_test_or_fixture or self._in_test_or_fixture[-1])
        self._func_stack.append(node)

    def _exit_function(self) -> None:
        self._func_stack.pop()
        self._in_test_or_fixture.pop()

    def _check_test_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not node.name.startswith("test_"):
            return
        # Skip @pytest.fixture-decorated functions whose names happen to start
        # with "test_" (e.g. `def test_data(self): ...` used as a fixture).
        if _has_pytest_fixture_decorator(node):
            return
        # Only check top-level and class-level test methods, not nested.
        if any(isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) for f in self._func_stack):
            return
        if not _function_has_assert(node):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="zero_assert_test",
                    message=f"Test function `{node.name}` contains zero assertions — tests execution, not behavior (Khorikov)",
                    quote=self._quote(node.lineno),
                    suggestion="Add at least one `assert`, `self.assert*`, or `pytest.raises` that verifies behavior.",
                )
            )

    # -- rmtree_outside_tmp: shutil.rmtree(X) where X isn't rooted in tmp_path.
    # -- sleep_based_wait: time.sleep(N) / asyncio.sleep(N) with N >= 1, used as readiness wait.
    # -- debug_leftover (call form): print(...) / breakpoint(...) / pdb.set_trace() in ACTUAL code.
    #    Using AST instead of regex avoids false positives on print() inside triple-quoted
    #    strings (e.g. test fixtures that ship Python scripts as string literals).

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_rmtree_call(node) and not _arg_is_tmp_rooted(node.args[0] if node.args else None):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="rmtree_outside_tmp",
                    message="shutil.rmtree on a path not rooted under tmp_path — risk of deleting real user data",
                    quote=self._quote(node.lineno),
                    suggestion="Always rmtree under tmp_path / Path(tempfile.mkdtemp()); never touch production storage paths.",
                )
            )
        sleep_n = _is_long_sleep(node)
        if sleep_n is not None and self._in_test_or_fixture[-1]:
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P1",
                    check="sleep_based_wait",
                    message=f"sleep({sleep_n}s) used as readiness probe — flaky on slow machines / CI",
                    quote=self._quote(node.lineno),
                    suggestion="Use a readiness probe: poll a health endpoint / wait on asyncio.Event / watch stdout for a 'ready' marker.",
                )
            )
        debug_name = _debug_call_name(node)
        if debug_name is not None:
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="debug_leftover",
                    message=f"Debug artifact `{debug_name}` left in test — not allowed in committed tests",
                    quote=self._quote(node.lineno),
                    suggestion="Remove the debug call, or use `caplog` / logger for intentional output.",
                )
            )
        # time.sleep with literal < 1s in unit tier is still a debug_leftover concern
        # (unit tests must be deterministic); integration tier uses sleep_based_wait above.
        if self.tier == "unit" and _is_time_sleep_call(node):
            self._emit(
                Issue(
                    file=self.path,
                    line=node.lineno,
                    severity="P0",
                    check="debug_leftover",
                    message="Debug artifact `time.sleep()` in unit test — unit tests must be deterministic and fast",
                    quote=self._quote(node.lineno),
                    suggestion="Remove the sleep; use a deterministic clock or mock the timing dependency.",
                )
            )
        self.generic_visit(node)

    @staticmethod
    def _is_rmtree_call(node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "rmtree":
            return True
        if isinstance(func, ast.Name) and func.id == "rmtree":
            return True
        return False


def _has_pytest_fixture_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _arg_is_tmp_rooted(arg: ast.AST | None) -> bool:
    """Approximate: is this expression rooted in a tmp_path-ish variable?"""
    if arg is None:
        return True  # no arg — don't flag (will error at runtime)
    if isinstance(arg, ast.Name):
        n = arg.id.lower()
        return any(k in n for k in ("tmp", "temp"))
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        v = arg.value.lower()
        # Literal strings: /tmp/... or relative paths under tmp directory.
        return v.startswith("/tmp/") or v.startswith("/var/folders/") or "tmp_path" in v
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div):
        return _arg_is_tmp_rooted(arg.left)
    if isinstance(arg, ast.Call) and arg.args:
        # Path(x) / str(x) / os.fspath(x) / os.path.join(tmp_path, "...")
        return _arg_is_tmp_rooted(arg.args[0])
    if isinstance(arg, ast.Attribute):
        # tmp_path.parent / tmp_path.name etc
        return _arg_is_tmp_rooted(arg.value)
    if isinstance(arg, ast.Subscript):
        # Args pulled from dicts/sequences — uncertain; don't flag.
        return True
    return False


def _debug_call_name(node: ast.Call) -> str | None:
    """Return a display name if this call is a debug artifact (print, breakpoint, pdb.*)."""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id == "print":
            return "print()"
        if func.id == "breakpoint":
            return "breakpoint()"
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "pdb":
            if func.attr in {"set_trace", "run"}:
                return f"pdb.{func.attr}()"
    return None


def _is_time_sleep_call(node: ast.Call) -> bool:
    """True if this call is `time.sleep(...)` specifically."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "sleep":
        if isinstance(func.value, ast.Name) and func.value.id == "time":
            return True
    return False


def _is_long_sleep(node: ast.Call) -> float | None:
    """Return sleep seconds if this is time.sleep(N) or asyncio.sleep(N)
    with a literal numeric N >= 1.0; else None."""
    func = node.func
    name = None
    if isinstance(func, ast.Attribute) and func.attr == "sleep":
        if isinstance(func.value, ast.Name):
            name = func.value.id
        elif isinstance(func.value, ast.Attribute):
            name = func.value.attr
    elif isinstance(func, ast.Name) and func.id == "sleep":
        name = "sleep"
    if name not in {"time", "asyncio", "anyio", "sleep"}:
        return None
    if not node.args:
        return None
    first = node.args[0]
    if not (
        isinstance(first, ast.Constant) and isinstance(first.value, (int, float)) and not isinstance(first.value, bool)
    ):
        return None
    if first.value >= 1:
        return float(first.value)
    return None


def _maybe_skip_call(stmt: ast.AST) -> ast.Call | None:
    """Return the pytest.skip Call node if stmt is `pytest.skip(...)`, else None."""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
        call = stmt.exc
    else:
        return None
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "skip":
        if isinstance(func.value, ast.Name) and func.value.id == "pytest":
            return call
    return None


def _is_tautological(expr: ast.AST) -> bool:
    # len(x) >= 0
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        left, op, right = expr.left, expr.ops[0], expr.comparators[0]
        if (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "len"
            and isinstance(op, ast.GtE)
            and isinstance(right, ast.Constant)
            and right.value == 0
        ):
            return True
    # isinstance(x, object)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "isinstance"
        and len(expr.args) == 2
        and isinstance(expr.args[1], ast.Name)
        and expr.args[1].id == "object"
    ):
        return True
    # A or not A / x in s or x not in s
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or) and len(expr.values) == 2:
        a, b = expr.values
        if _is_negation_of(a, b) or _is_negation_of(b, a):
            return True
        if _inverted_contains(a, b):
            return True
    return False


def _is_negation_of(a: ast.AST, b: ast.AST) -> bool:
    if isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.Not):
        return ast.dump(b.operand) == ast.dump(a)
    return False


def _inverted_contains(a: ast.AST, b: ast.AST) -> bool:
    if not (isinstance(a, ast.Compare) and isinstance(b, ast.Compare)):
        return False
    if len(a.ops) != 1 or len(b.ops) != 1:
        return False
    if ast.dump(a.left) != ast.dump(b.left):
        return False
    if ast.dump(a.comparators[0]) != ast.dump(b.comparators[0]):
        return False
    ops = (type(a.ops[0]), type(b.ops[0]))
    return ops == (ast.In, ast.NotIn) or ops == (ast.NotIn, ast.In)


def _is_or_assert(expr: ast.AST) -> bool:
    if not (isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or)):
        return False
    if _is_tautological(expr):
        return False
    return all(isinstance(v, (ast.Compare, ast.Call)) for v in expr.values)


_WEAK_NAMES = {"result", "output", "response", "ret", "data", "value", "x", "y", "obj"}


def _is_weak_assert(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        op = expr.ops[0]
        right = expr.comparators[0]
        if isinstance(op, ast.IsNot) and isinstance(right, ast.Constant) and right.value is None:
            return True
        if (
            isinstance(op, ast.Gt)
            and isinstance(right, ast.Constant)
            and right.value == 0
            and isinstance(expr.left, ast.Call)
            and isinstance(expr.left.func, ast.Name)
            and expr.left.func.id == "len"
        ):
            return True
    if isinstance(expr, ast.Name) and expr.id in _WEAK_NAMES:
        return True
    if isinstance(expr, ast.Attribute) and expr.attr in {"success", "ok"}:
        return True
    return False


def _function_has_assert(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.withitem) and isinstance(child.context_expr, ast.Call):
            func = child.context_expr.func
            if isinstance(func, ast.Attribute) and func.attr in {"raises", "warns", "deprecated_call"}:
                if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                    return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            name = child.func.attr
            if name.startswith("assert") or name.startswith("assert_"):
                return True
    return False


def _is_direct_verifier_stmt(stmt: ast.stmt) -> bool:
    """True iff this SINGLE top-level statement itself verifies behavior.

    Deliberately shallow — no recursion into nested blocks. Recognized forms:
      - `assert ...`
      - `with pytest.raises(...) / pytest.warns(...) / pytest.deprecated_call(...)`
      - expression statement `mock.assert_called_*() / self.assertXxx(...)`

    Everything else (for / while / try / match, nested if, def / class, bare
    calls, assignments) returns False. Higher-level callers treat statements
    in this "not a direct verifier" bucket as UNKNOWN rather than
    "definitely does not verify", so complex branches are not flagged.
    """
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            ce = item.context_expr
            if isinstance(ce, ast.Call) and isinstance(ce.func, ast.Attribute):
                if isinstance(ce.func.value, ast.Name) and ce.func.value.id == "pytest":
                    if ce.func.attr in {"raises", "warns", "deprecated_call"}:
                        return True
        return False
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if isinstance(call.func, ast.Attribute):
            name = call.func.attr
            if name.startswith("assert") or name.startswith("assert_"):
                return True
    return False


def _classify_branch(body: list[ast.stmt]) -> str:
    """Classify an if-chain branch body as one of:

    - 'verified': at least one top-level statement is a direct verifier
    - 'empty':    empty, or only Pass / Ellipsis / a leading docstring
    - 'opaque':   contains structured / non-verifier code (for, while, try,
                  match, nested if, def, class, bare calls, assignments).
                  Not evaluated; kept silent to avoid false positives.
    """
    # Strip a leading docstring if present.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if any(_is_direct_verifier_stmt(s) for s in body):
        return "verified"

    def _is_filler(s: ast.stmt) -> bool:
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis:
            return True
        return False

    if not body or all(_is_filler(s) for s in body):
        return "empty"
    return "opaque"


def _flatten_if_chain(node: ast.If) -> tuple[list[list[ast.stmt]], bool, list[ast.If]]:
    """Walk an `if / elif / .../ else` chain head.

    Returns:
      - branches: list of stmt-lists in source order, one per `if`/`elif` and
                  (if present) the final `else`.
      - has_else: True iff the chain ends with an explicit `else:` clause.
      - chained:  inner elif-If nodes (excluding the head) so the caller can
                  mark them as handled and avoid double-reporting when the
                  AST walker later visits them individually.
    """
    branches: list[list[ast.stmt]] = [node.body]
    chained: list[ast.If] = []
    cur = node
    while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
        cur = cur.orelse[0]
        branches.append(cur.body)
        chained.append(cur)
    has_else = bool(cur.orelse) and not (len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If))
    if has_else:
        branches.append(cur.orelse)
    return branches, has_else, chained


# ---------------------------------------------------------------------------
# Module-level statement check (sys.modules mutation)
# ---------------------------------------------------------------------------


def check_module_level_sys_modules(path: Path, tree: ast.AST, tier: str) -> list[Issue]:
    if not rule_applies("module_level_sys_modules", tier):
        return []
    rel = str(path.relative_to(REPO_ROOT))
    issues: list[Issue] = []

    def _emit_if_sys_modules(stmt: ast.stmt) -> None:
        if not isinstance(stmt, ast.Assign):
            return
        for target in stmt.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "modules"
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "sys"
            ):
                quote = _quote_line(path, stmt.lineno)
                # Honor audit-noqa the same way AST / regex checks do.
                if AUDIT_NOQA in quote:
                    return
                issues.append(
                    Issue(
                        file=rel,
                        line=stmt.lineno,
                        severity="P0",
                        check="module_level_sys_modules",
                        message="Module-level `sys.modules[...] = ...` breaks test isolation — bleeds into other tests (F.I.R.S.T. Independent)",
                        quote=quote,
                        suggestion="Move inside a fixture with yield + cleanup, or use monkeypatch.setitem(sys.modules, ...).",
                        tier=tier,
                    )
                )
                return

    def _walk_module_scope(nodes: list[ast.stmt]) -> None:
        for stmt in nodes:
            _emit_if_sys_modules(stmt)
            if isinstance(stmt, ast.If):
                _walk_module_scope(stmt.body)
                _walk_module_scope(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                _walk_module_scope(stmt.body)
                for h in stmt.handlers:
                    _walk_module_scope(h.body)
                _walk_module_scope(stmt.orelse)
                _walk_module_scope(stmt.finalbody)
            elif isinstance(stmt, ast.With):
                _walk_module_scope(stmt.body)

    _walk_module_scope(getattr(tree, "body", []))
    return issues


def _quote_line(path: Path, lineno: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1].strip()[:120]
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Regex-based checks
# ---------------------------------------------------------------------------


# Only statement-level patterns that can't be expressed as ast.Call. Call-form
# debug artifacts (print / breakpoint / pdb.set_trace / time.sleep) are detected
# via AST in _AstChecker.visit_Call — that avoids false positives on code-in-string
# test fixtures.
_DEBUG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*import\s+pdb\b"), "import pdb"),
]

_REAL_IO_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[\"']([^\"']*/)?sample_data/[^\"']*\.(db|duckdb|sqlite)[^\"']*[\"']"),
    re.compile(r"[\"']tests/data/[^\"']*\.(db|duckdb|sqlite)[^\"']*[\"']"),
]

_LAMBDA_THROW = re.compile(r"lambda\s+[^:]*:\s*\(\s*_\s+for\s+_\s+in\s+\(\s*\)\s*\)\s*\.\s*throw")

_PATCH_BUILTINS_OPEN = re.compile(r"""patch(?:\.object)?\s*\(\s*["'](?:builtins|__builtins__)\.open["']""")

# Match hardcoded localhost connections in three flavors:
#  1. URL / host:port literal:   "localhost:15432" / "postgresql://127.0.0.1:6379/"
#  2. Kwarg host=:               host="localhost"  / hostname='127.0.0.1'  / server="localhost"
#  3. Kwarg port= with literal:  port=15432 / PORT=9030   (pairs with kwarg host only in the
#     kwarg case; bare `port=<int>` alone is too generic to flag)
_HARDCODED_LOCALHOST_URL = re.compile(
    r"""["'](?:(?:https?|tcp|postgres(?:ql)?|mysql|mongodb|redis)://)?(?:localhost|127\.0\.0\.1)[:/]\d{2,5}\b"""
)
_HARDCODED_LOCALHOST_KWARG = re.compile(
    r"""\b(?:host|hostname|server|HOST|HOSTNAME|SERVER|DB_HOST)\s*=\s*["'](?:localhost|127\.0\.0\.1)["']"""
)


def regex_scan(path: Path, required_packages: set[str], tier: str) -> list[Issue]:
    rel = str(path.relative_to(REPO_ROOT))
    issues: list[Issue] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return issues

    importorskip_pattern = re.compile(r"""importorskip\s*\(\s*['"]([^'"]+)['"]""")

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        noqa_match = f"# {AUDIT_NOQA}" in line

        # debug_leftover (excluding time.sleep — handled by sleep_based_wait for integration / debug_leftover for unit)
        for pat, name in _DEBUG_PATTERNS:
            if pat.search(line) and not noqa_match and rule_applies("debug_leftover", tier):
                issues.append(
                    Issue(
                        file=rel,
                        line=lineno,
                        severity="P0",
                        check="debug_leftover",
                        message=f"Debug artifact `{name}` left in test — not allowed in committed tests",
                        quote=stripped[:120],
                        suggestion="Remove the debug call, or use `caplog` / logger for intentional output.",
                        tier=tier,
                    )
                )
                break

        for pat in _REAL_IO_PATTERNS:
            if pat.search(line) and not noqa_match and rule_applies("real_external_io_path", tier):
                issues.append(
                    Issue(
                        file=rel,
                        line=lineno,
                        severity="P0",
                        check="real_external_io_path",
                        message="Hardcoded pre-built DB file — unit tests must be hermetic (Google Hermetic Tests)",
                        quote=stripped[:120],
                        suggestion='Use `":memory:"` or create a temp DB under `tmp_path` in a fixture.',
                        tier=tier,
                    )
                )
                break

        m = importorskip_pattern.search(line)
        if m and not noqa_match and rule_applies("importorskip_on_required", tier):
            pkg = m.group(1).replace("-", "_").split(".")[0]
            if pkg in required_packages:
                issues.append(
                    Issue(
                        file=rel,
                        line=lineno,
                        severity="P0",
                        check="importorskip_on_required",
                        message=f"`importorskip('{m.group(1)}')` on required dependency — creates test coverage hiding",
                        quote=stripped[:120],
                        suggestion=f"`{m.group(1)}` is in [project.dependencies]; import it directly.",
                        tier=tier,
                    )
                )

        if _LAMBDA_THROW.search(line) and not noqa_match and rule_applies("lambda_throw", tier):
            issues.append(
                Issue(
                    file=rel,
                    line=lineno,
                    severity="P1",
                    check="lambda_throw",
                    message="`lambda: (...).throw(Exception)` is unreadable — use `mock.side_effect = ExceptionClass(...)`",
                    quote=stripped[:120],
                    suggestion="Replace with `mock.side_effect = RuntimeError('...')`.",
                    tier=tier,
                )
            )

        # patch_builtins_open (integration only)
        if _PATCH_BUILTINS_OPEN.search(line) and not noqa_match and rule_applies("patch_builtins_open", tier):
            issues.append(
                Issue(
                    file=rel,
                    line=lineno,
                    severity="P0",
                    check="patch_builtins_open",
                    message="`patch('builtins.open')` globally intercepts every open() in the with-block — flaky + may hide production bugs",
                    quote=stripped[:120],
                    suggestion="Patch the specific module path, e.g. `patch('datus.module.path.open')`, not builtins.",
                    tier=tier,
                )
            )

        # hardcoded_localhost (integration only) — two flavors
        if rule_applies("hardcoded_localhost", tier) and not noqa_match:
            uses_env_override = "os.environ" in line or "os.getenv" in line or "getenv(" in line
            if not uses_env_override and (
                _HARDCODED_LOCALHOST_URL.search(line) or _HARDCODED_LOCALHOST_KWARG.search(line)
            ):
                issues.append(
                    Issue(
                        file=rel,
                        line=lineno,
                        severity="P1",
                        check="hardcoded_localhost",
                        message="Hardcoded `localhost` host — prevents env-based override (CI vs local vs docker) and parallel runs",
                        quote=stripped[:120],
                        suggestion="Read host/port from env: `os.getenv('TEST_DB_HOST', 'localhost')`, or use a pytest fixture.",
                        tier=tier,
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def check_file_size_budget(path: Path, tier: str) -> list[Issue]:
    if not rule_applies("file_size_budget", tier):
        return []
    try:
        line_count = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
    except OSError:
        return []
    if line_count > FILE_SIZE_LIMIT:
        return [
            Issue(
                file=str(path.relative_to(REPO_ROOT)),
                line=1,
                severity="P1",
                check="file_size_budget",
                message=f"Test file has {line_count} lines (> {FILE_SIZE_LIMIT}) — split by class / feature",
                quote=f"{line_count} lines total",
                suggestion="Split into multiple files, one per source class being tested.",
                tier=tier,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Cross-file checks
# ---------------------------------------------------------------------------


def collect_source_classes() -> dict[str, list[Path]]:
    """Return {ClassName: [source_files defining it]}."""
    index: dict[str, list[Path]] = defaultdict(list)
    for py in SOURCE_ROOT.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index[node.name].append(py)
    return index


def build_test_class_index(roots: list[Path]) -> dict[str, list[Path]]:
    """Scan given test roots, return {source_class_name: [test_files_with_TestClassName]}."""
    index: dict[str, list[Path]] = defaultdict(list)
    class_def = re.compile(r"^class\s+Test([A-Z][A-Za-z0-9_]+)\b", re.MULTILINE)
    for root in roots:
        for py in root.rglob("test_*.py"):
            try:
                text = py.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in class_def.finditer(text):
                index[m.group(1)].append(py)
    return index


def check_duplicate_test_files(target_files: set[Path], tiers_in_scope: set[str]) -> list[Issue]:
    """Detect redundant test files within each tier separately.

    Cross-tier duplication (same class covered by a unit test with mocks AND
    an integration test with real services) is a legitimate pattern
    (Khorikov / Google Test Pyramid). Therefore detection runs per-tier:
    class `TestFoo` appearing once in `tests/unit_tests/` and once in
    `tests/integration/` is NOT a duplicate.
    """
    issues: list[Issue] = []
    source_classes = collect_source_classes()
    reported_pairs: set[tuple[str, ...]] = set()
    _NUISANCE_AFFIXES = ("_tools", "_module", "_unit", "_v2", "_ext", "_new", "_v1")

    for tier in tiers_in_scope:
        root = TIER_ROOTS[tier]
        test_classes = build_test_class_index([root])

        # Signal 1: same source class has TestX class in multiple test files (within this tier).
        for cls, test_files in test_classes.items():
            if cls not in source_classes:
                continue
            if cls in {"Meta", "Config", "Error", "Base"}:
                continue
            if len(test_files) < 2:
                continue
            touched = [t for t in test_files if t in target_files]
            if not touched:
                continue
            rel_tests = ", ".join(str(t.relative_to(REPO_ROOT)) for t in test_files)
            key = tuple(sorted(str(t) for t in test_files))
            reported_pairs.add(key)
            for t in touched:
                issues.append(
                    Issue(
                        file=str(t.relative_to(REPO_ROOT)),
                        line=1,
                        severity="P1",
                        check="duplicate_test_files",
                        message=f"Class `{cls}` has `class Test{cls}` in multiple {tier}-tier test files: {rel_tests}",
                        quote=f"defined in {source_classes[cls][0].relative_to(REPO_ROOT)}",
                        suggestion="Merge into a single test file — each source class should have exactly ONE test file per tier (Meszaros: Test Code Duplication).",
                        tier=tier,
                    )
                )

        # Signal 2: near-duplicate filenames in same directory (within this tier).
        files_by_dir: dict[Path, list[Path]] = defaultdict(list)
        for p in root.rglob("test_*.py"):
            files_by_dir[p.parent].append(p)
        for _dir_path, files in files_by_dir.items():
            stems = {p.stem: p for p in files}
            for stem, p in stems.items():
                for aff in _NUISANCE_AFFIXES:
                    if stem.endswith(aff) and stem[: -len(aff)] in stems:
                        other = stems[stem[: -len(aff)]]
                        pair = tuple(sorted([str(p), str(other)]))
                        if pair in reported_pairs:
                            continue
                        reported_pairs.add(pair)
                        touched = [f for f in (p, other) if f in target_files]
                        if not touched:
                            continue
                        for t in touched:
                            issues.append(
                                Issue(
                                    file=str(t.relative_to(REPO_ROOT)),
                                    line=1,
                                    severity="P1",
                                    check="duplicate_test_files",
                                    message=f"Near-duplicate filenames in same {tier}-tier directory: `{p.name}` and `{other.name}` (differ only by `{aff}` suffix)",
                                    suggestion=f"Merge into one file — same-directory test files differing only by `{aff}` almost always indicate forked duplicates.",
                                    tier=tier,
                                )
                            )
                        break
                if stem + "s" in stems:
                    other = stems[stem + "s"]
                    pair = tuple(sorted([str(p), str(other)]))
                    if pair in reported_pairs:
                        continue
                    reported_pairs.add(pair)
                    touched = [f for f in (p, other) if f in target_files]
                    if not touched:
                        continue
                    for t in touched:
                        issues.append(
                            Issue(
                                file=str(t.relative_to(REPO_ROOT)),
                                line=1,
                                severity="P1",
                                check="duplicate_test_files",
                                message=f"Near-duplicate filenames in same {tier}-tier directory: `{p.name}` and `{other.name}` (singular/plural)",
                                suggestion="Merge into one file — singular/plural pairs almost always indicate forked duplicates.",
                                tier=tier,
                            )
                        )
    return issues


def check_missing_test_for_module() -> list[Issue]:
    """Unit-tier only: datus/X/Y.py exists but no test file imports it."""
    imported_modules: set[str] = set()
    import_pattern = re.compile(r"^\s*(?:from|import)\s+(datus(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
    for py in TESTS_ROOT.rglob("*.py"):
        try:
            for line in py.read_text(encoding="utf-8", errors="replace").splitlines():
                m = import_pattern.match(line)
                if m:
                    imported_modules.add(m.group(1))
        except OSError:
            continue

    issues: list[Issue] = []
    for py in SOURCE_ROOT.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        mod_parts = ("datus",) + py.relative_to(SOURCE_ROOT).with_suffix("").parts
        mod_name = ".".join(mod_parts)
        if mod_name in imported_modules:
            continue
        parent = ".".join(mod_parts[:-1])
        symbol = mod_parts[-1]
        if parent in imported_modules and _test_imports_symbol(parent, symbol):
            continue
        if _is_trivial_module(py):
            continue
        issues.append(
            Issue(
                file=str(py.relative_to(REPO_ROOT)),
                line=1,
                severity="P1",
                check="missing_test_for_module",
                message=f"Source module `{mod_name}` has no test file importing it anywhere under tests/",
                suggestion=f"Create a test file that imports from `{mod_name}` (mirror path preferred: tests/unit_tests/{py.relative_to(SOURCE_ROOT).parent}/test_{py.stem}.py).",
                tier="unit",
            )
        )
    return issues


def _test_imports_symbol(parent_module: str, symbol: str) -> bool:
    pattern = re.compile(rf"from\s+{re.escape(parent_module)}\s+import\s+[^\n]*\b{re.escape(symbol)}\b")
    for py in TESTS_ROOT.rglob("*.py"):
        try:
            if pattern.search(py.read_text(encoding="utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False


def _is_trivial_module(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return False
    return True


def check_empty_test_subdir(tiers_in_scope: set[str]) -> list[Issue]:
    """Integration-tier P2: directories under test roots that contain only
    __pycache__ / __init__.py / nothing — stale after refactor."""
    if "integration" not in tiers_in_scope:
        return []
    issues: list[Issue] = []
    root = INTEGRATION_ROOT
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        if sub.name in {"__pycache__"}:
            continue
        entries = [p for p in sub.iterdir() if p.name != "__pycache__"]
        # Directory with only __init__.py or completely empty (excluding __pycache__)
        if not entries:
            pass  # truly empty
        elif len(entries) == 1 and entries[0].name == "__init__.py":
            pass
        else:
            continue
        issues.append(
            Issue(
                file=str(sub.relative_to(REPO_ROOT)),
                line=1,
                severity="P1",
                check="empty_test_subdir",
                message="Empty test directory (only __pycache__ / __init__.py) — likely stale after a refactor",
                suggestion="Delete the directory or commit the test files that belong here.",
                tier="integration",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Required-package loader
# ---------------------------------------------------------------------------


def load_required_packages() -> set[str]:
    if not PYPROJECT.exists():
        return set()
    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    deps = data.get("project", {}).get("dependencies", []) or []
    required: set[str] = set()
    for dep in deps:
        name = re.split(r"[<>=!~\[;]", dep, maxsplit=1)[0].strip()
        if name:
            required.add(name.replace("-", "_"))
    return required


# ---------------------------------------------------------------------------
# Target file discovery (tier-aware)
# ---------------------------------------------------------------------------


def _ensure_test_file(p: Path, allowed_tiers: set[str]) -> bool:
    try:
        p_abs = p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    except OSError:
        return False
    if not (p_abs.is_file() and p_abs.suffix == ".py" and p_abs.name != "__init__.py"):
        return False
    tier = detect_tier(p_abs)
    return tier in allowed_tiers


class AuditGitDiffError(RuntimeError):
    """Raised when `git diff` cannot enumerate changed files.

    Represents an INDETERMINATE audit state (ref lookup failure, missing
    merge base, force-push breakage, subprocess timeout). The caller must
    FAIL CLOSED — returning an empty list would let a PR merge with the
    audit silently disabled.
    """


def git_diff_files(base: str, allowed_tiers: set[str]) -> list[Path]:
    # --diff-filter=AMR covers Added, Modified, and Renamed. `-M` forces rename
    # detection even when the user has `diff.renames=false`; without it a
    # renamed (or rename+modified) test file would bypass the audit entirely.
    cmd = ["git", "diff", "-M", "--name-only", "--diff-filter=AMR", f"{base}...HEAD"]
    try:
        out = subprocess.check_output(cmd, text=True, cwd=REPO_ROOT, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Fail closed: propagate so main() can mark the audit as failed rather
        # than silently treating "can't enumerate" the same as "0 files changed".
        raise AuditGitDiffError(
            f"git diff failed (ref={base!r}, cwd={REPO_ROOT!r}): {e}. "
            "Cannot determine changed files — failing audit closed."
        ) from e
    paths: list[Path] = []
    for line in out.splitlines():
        p = REPO_ROOT / line.strip()
        if _ensure_test_file(p, allowed_tiers):
            paths.append(p)
    return paths


def all_test_files(allowed_tiers: set[str]) -> list[Path]:
    out: list[Path] = []
    for tier in allowed_tiers:
        root = TIER_ROOTS[tier]
        out.extend(p for p in root.rglob("test_*.py") if _ensure_test_file(p, {tier}))
    return out


def paths_to_files(paths: list[str], allowed_tiers: set[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.is_file() and _ensure_test_file(p, allowed_tiers):
            out.append(p)
        elif p.is_dir():
            out.extend(q for q in p.rglob("test_*.py") if _ensure_test_file(q, allowed_tiers))
    return out


# ---------------------------------------------------------------------------
# Main scan orchestration
# ---------------------------------------------------------------------------


def scan_file(path: Path, required_packages: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    tier = detect_tier(path)
    if tier is None:
        return issues
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"cannot read {path}: {e}")
        return issues
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        log(f"syntax error in {path}:{e.lineno} — skipping AST checks")
        issues.extend(regex_scan(path, required_packages, tier))
        issues.extend(check_file_size_budget(path, tier))
        return issues
    checker = _AstChecker(path, source, tier)
    checker.visit(tree)
    issues.extend(checker.issues)
    issues.extend(check_module_level_sys_modules(path, tree, tier))
    issues.extend(regex_scan(path, required_packages, tier))
    issues.extend(check_file_size_budget(path, tier))
    return issues


def emit_report(
    issues: list[Issue],
    mode: str,
    target_files: list[Path],
    json_path: str | None,
) -> tuple[int, int]:
    issues.sort(key=lambda i: (i.file, i.line, i.severity, i.check))
    p0 = sum(1 for i in issues if i.severity == "P0")
    p1 = sum(1 for i in issues if i.severity == "P1")

    log(f"mode={mode} scanned={len(target_files)} issues={len(issues)} (P0={p0}, P1={p1})")

    by_file: dict[str, list[Issue]] = defaultdict(list)
    for i in issues:
        by_file[i.file].append(i)
    for f, lst in sorted(by_file.items()):
        log(f"  {f}:")
        for i in lst:
            tag = f" [{i.tier}]" if i.tier else ""
            log(f"    [{i.severity}]{tag} line {i.line}: {i.check} — {i.message}")

    if json_path:
        by_tier = defaultdict(lambda: {"p0": 0, "p1": 0})
        for i in issues:
            key = i.tier or "unknown"
            by_tier[key]["p0" if i.severity == "P0" else "p1"] += 1
        out = {
            "mode": mode,
            "scanned": [str(p.relative_to(REPO_ROOT)) for p in target_files],
            "summary": {
                "total": len(issues),
                "p0": p0,
                "p1": p1,
                "by_tier": dict(by_tier),
            },
            "issues": [asdict(i) for i in issues],
        }
        Path(json_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log(f"wrote JSON report: {json_path}")

    return p0, p1


def write_github_output(mode: str, p0: int, p1: int, outcome: str) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    pairs = {
        "audit_outcome": outcome,
        "issue_total": str(p0 + p1),
        "p0_count": str(p0),
        "p1_count": str(p1),
        "scan_mode": mode,
    }
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            for k, v in pairs.items():
                fh.write(f"{k}={v}\n")
    else:
        for k, v in pairs.items():
            log(f"output: {k}={v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static test-quality audit (unit + integration tiers)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--diff-only", metavar="BASE", help="Scan only files changed vs BASE (e.g. origin/main). P1 issues hard-fail."
    )
    mode.add_argument("--all", action="store_true", help="Scan all tests under selected tiers. P1 issues warn-only.")
    mode.add_argument("--paths", nargs="+", help="Scan explicit paths (files or dirs). P1 issues hard-fail.")
    p.add_argument(
        "--tier",
        choices=["all", "unit", "integration"],
        default="all",
        help="Which test tier(s) to audit. 'all' (default) = unit + integration.",
    )
    p.add_argument("--json", metavar="PATH", help="Write JSON report to PATH")
    p.add_argument("--github-output", action="store_true", help="Append key=value pairs to $GITHUB_OUTPUT")
    p.add_argument(
        "--repo-root",
        metavar="PATH",
        help=(
            "Override the repo root used for git diff and file reads. "
            "Required when the auditor is executed from a different checkout than the "
            "repo being audited (e.g. pull_request_target workflow running base/ci/audit_tests.py "
            "against pr/ content)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.repo_root:
        configure_repo_root(Path(args.repo_root))

    if args.tier == "all":
        allowed_tiers = {"unit", "integration"}
    else:
        allowed_tiers = {args.tier}

    if args.diff_only:
        mode = "diff"
        try:
            target_files = git_diff_files(args.diff_only, allowed_tiers)
        except AuditGitDiffError as e:
            # Fail closed: don't let a git breakage silently disable the audit.
            log(f"ERROR: {e}")
            if args.github_output:
                write_github_output(mode, 0, 0, "failure")
            return 1
    elif args.all:
        mode = "full"
        target_files = all_test_files(allowed_tiers)
    else:
        mode = "paths"
        target_files = paths_to_files(args.paths, allowed_tiers)

    if not target_files and not (mode == "full" and "integration" in allowed_tiers):
        # Edge case: full scan wants to still run empty_test_subdir even if 0 files
        log("no test files to scan — done")
        if args.github_output:
            write_github_output(mode, 0, 0, "success")
        return 0

    required_packages = load_required_packages()
    log(f"required_packages loaded: {len(required_packages)} entries")
    log(f"tier scope: {sorted(allowed_tiers)}")

    issues: list[Issue] = []
    target_set = set(target_files)
    for path in target_files:
        issues.extend(scan_file(path, required_packages))

    # Cross-file checks
    issues.extend(check_duplicate_test_files(target_set, allowed_tiers))
    if mode == "full":
        if "unit" in allowed_tiers:
            issues.extend(check_missing_test_for_module())
        issues.extend(check_empty_test_subdir(allowed_tiers))

    p0, p1 = emit_report(issues, mode, target_files, args.json)

    if mode == "full":
        failed = p0 > 0
    else:
        failed = (p0 + p1) > 0
    outcome = "failure" if failed else "success"

    if args.github_output:
        write_github_output(mode, p0, p1, outcome)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
