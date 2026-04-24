# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Layer A built-in invariants for ValidationHook.

Deterministic, LLM-free checks that run on every mutating tool call:

- ``table_exists`` — ``describe_table`` returns a non-empty column list
- ``transfer_row_count_parity`` — tool-reported source and target row counts
  agree

Always enforced, regardless of ``agent.validation.skill_validators_enabled``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from datus.utils.loggings import get_logger
from datus.validation.report import (
    CheckResult,
    DeliverableTarget,
    SessionTarget,
    TableTarget,
    TransferTarget,
    ValidationReport,
    describe_target,
)

if TYPE_CHECKING:
    from datus.tools.func_tool.database import DBFuncTool

logger = get_logger(__name__)


async def run_builtin_checks(
    target: DeliverableTarget,
    db_func_tool: Optional["DBFuncTool"] = None,
) -> ValidationReport:
    """Run Layer A invariants for a single target.

    Args:
        target: The deliverable produced by a mutating tool call
        db_func_tool: The tool instance to dispatch ``describe_table`` /
            ``read_query`` through. When ``None`` (tests / harness without a
            live DB) the check list is empty.

    Returns:
        :class:`ValidationReport` with one or more :class:`CheckResult`
        entries; never raises.
    """
    report = ValidationReport(target=target, checks=[])
    if db_func_tool is None:
        logger.info("Layer A skipped for %s: no connector available", _target_descriptor(target))
        return report

    if isinstance(target, TableTarget):
        await _check_table(target, db_func_tool, report)
    elif isinstance(target, TransferTarget):
        await _check_transfer(target, db_func_tool, report)
    elif isinstance(target, SessionTarget):
        for inner in target.targets:
            nested = await run_builtin_checks(inner, db_func_tool=db_func_tool)
            inner_tag = describe_target(inner)
            for check in nested.checks:
                observed = dict(check.observed) if check.observed else {}
                observed["_target"] = inner_tag
                check.observed = observed
            report.checks.extend(nested.checks)
            report.warnings.extend(nested.warnings)

    passed = sum(1 for c in report.checks if c.passed)
    failed = len(report.checks) - passed
    summary = f"{passed} passed, {failed} failed" if failed else f"{passed} passed"
    logger.info("Layer A checks for %s: %s", _target_descriptor(target), summary)
    for c in report.checks:
        if c.passed:
            logger.debug("  ✓ %s observed=%s", c.name, c.observed)
        else:
            logger.warning("  ✗ %s observed=%s expected=%s error=%s", c.name, c.observed, c.expected, c.error)
    return report


def _target_descriptor(target: DeliverableTarget) -> str:
    if isinstance(target, TableTarget):
        return f"table {target.database}.{target.fqn}"
    if isinstance(target, TransferTarget):
        return f"transfer {target.source.name} → {target.target.database}.{target.target.fqn}"
    if isinstance(target, SessionTarget):
        return f"session[{len(target.targets)} target(s)]"
    return repr(target)


async def run_session_builtin_checks(
    session: SessionTarget,
    db_func_tool: Optional["DBFuncTool"] = None,
) -> ValidationReport:
    """``on_end`` entrypoint — wrap each accumulated target in its own checks.

    Identical to calling :func:`run_builtin_checks` with the ``SessionTarget``
    directly; kept as a named alias so ``ValidationHook.on_end`` reads cleanly.
    """
    return await run_builtin_checks(session, db_func_tool=db_func_tool)


async def _check_table(
    target: TableTarget,
    db_func_tool: "DBFuncTool",
    report: ValidationReport,
) -> None:
    report.checks.append(_run_describe_table(target, db_func_tool))


async def _check_transfer(
    target: TransferTarget,
    db_func_tool: "DBFuncTool",
    report: ValidationReport,
) -> None:
    """Run invariants for a ``TransferTarget`` — target exists + row count parity."""
    # (1) Target table exists
    tgt_exists = _run_describe_table(target.target, db_func_tool)
    report.checks.append(tgt_exists)

    # (2) Row count parity — source vs target counts reported by the tool.
    # We do NOT re-run source queries here (dangerous, may have side effects).
    parity = _run_row_count_parity(target)
    if parity is not None:
        report.checks.append(parity)


def _run_describe_table(target: TableTarget, db_func_tool: "DBFuncTool") -> CheckResult:
    """Check that the target table exists and has at least one column.

    Route to the datasource the tool wrote through. ``target.datasource``
    carries the connector key (e.g. "ch_prod"); without it a cross-datasource
    write would be validated against the node's default connector and
    misreport the table as missing. See ``report.TableTarget.datasource``.
    """
    try:
        result = db_func_tool.describe_table(
            table_name=target.table,
            catalog=target.catalog or "",
            database=target.database,
            schema_name=target.db_schema or "",
            datasource=target.datasource or "",
        )
    except Exception as e:
        return CheckResult(
            name="table_exists",
            passed=False,
            severity="blocking",
            source="builtin",
            error=f"describe_table raised: {e}",
        )

    if not getattr(result, "success", False):
        return CheckResult(
            name="table_exists",
            passed=False,
            severity="blocking",
            source="builtin",
            error=getattr(result, "error", "describe_table failed"),
        )

    payload = getattr(result, "result", None) or {}
    columns = payload.get("columns", []) if isinstance(payload, dict) else []
    passed = bool(columns)
    return CheckResult(
        name="table_exists",
        passed=passed,
        severity="blocking",
        source="builtin",
        observed={"column_count": len(columns)},
        expected={"column_count_gte": 1},
        error=None if passed else "describe_table returned zero columns",
    )


def _run_row_count_parity(target: TransferTarget) -> Optional[CheckResult]:
    src = target.source_row_count
    tgt = target.transferred_row_count
    if src is None or tgt is None:
        # Tool did not report counts — skip this check (don't block, just no data).
        return None
    return CheckResult(
        name="transfer_row_count_parity",
        passed=src == tgt,
        severity="blocking",
        source="builtin",
        observed={"source_row_count": src, "transferred_row_count": tgt},
        expected={"source_row_count_eq_transferred_row_count": True},
        error=None if src == tgt else f"source rows {src} != transferred rows {tgt}",
    )
