# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Data models for validation reports and deliverable targets.

These types are filled by mutating tools (via ``FuncToolResult.result[
"deliverable_target"]``), consumed by :class:`ValidationHook`, and surfaced in
``NodeResult.validation_report`` for downstream observability.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class DBRef(BaseModel):
    """Lightweight database reference used in :class:`TransferTarget`."""

    name: str = Field(..., description="Database name / connector key")

    model_config = ConfigDict(frozen=True)


class TableTarget(BaseModel):
    """Deliverable target: a single physical table written by a DDL/DML tool."""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    type: Literal["table"] = "table"
    database: str = Field(..., description="Database / connector key")
    db_schema: Optional[str] = Field(
        default=None,
        description="Schema name; may be None for flat-namespace engines",
        alias="schema",
    )
    table: str = Field(..., description="Table name (unqualified)")
    rows_affected: Optional[int] = Field(
        default=None,
        description="Row count reported by the tool (CTAS row count or INSERT affected rows)",
    )

    @property
    def fqn(self) -> str:
        """Fully qualified name (schema.table or just table)."""
        if self.db_schema:
            return f"{self.db_schema}.{self.table}"
        return self.table


class TransferTarget(BaseModel):
    """Deliverable target: a cross-database transfer.

    The tool is required to report authoritative source / target row counts so
    reconciliation does not need to re-run the source query.
    """

    type: Literal["transfer"] = "transfer"
    source: DBRef = Field(..., description="Source database reference")
    target: TableTarget = Field(..., description="Target table where data was written")
    source_row_count: Optional[int] = Field(
        default=None, description="Row count of the source query (tool-reported, not re-computed)"
    )
    transferred_row_count: Optional[int] = Field(
        default=None, description="Row count actually written to the target (tool-reported)"
    )

    @property
    def database(self) -> str:
        """Database of the write target — lets hook/builtin checks treat this uniformly."""
        return self.target.database


# Discriminated union used by tools to report the deliverable produced by a
# single mutating tool call. ``DeliverableTarget.model_validate(dict)`` will
# pick the right subclass based on the ``type`` discriminator.
DeliverableTarget = Union[TableTarget, TransferTarget]


class SessionTarget(BaseModel):
    """Aggregated targets accumulated across a whole agent run.

    Passed to ``on_end`` validators so they can reason over the complete run
    (e.g. reconciliation across multiple transferred tables).
    """

    type: Literal["session"] = "session"
    targets: List[Union[TableTarget, TransferTarget]] = Field(default_factory=list)

    @property
    def database(self) -> Optional[str]:
        """Convenience: the database of the first target, if any."""
        if not self.targets:
            return None
        return self.targets[0].database


class TargetFilter(BaseModel):
    """Filter spec declared in a validator skill's frontmatter.

    All set fields must match for the filter to apply; any unset (``None``)
    field is a wildcard. A skill with an empty ``targets: []`` matches every
    target.
    """

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    type: Optional[Literal["table", "transfer"]] = None
    database: Optional[str] = None
    db_schema: Optional[str] = Field(default=None, alias="schema")
    table: Optional[str] = None
    table_pattern: Optional[str] = Field(default=None, description="fnmatch glob pattern matched against target.table")


class CheckResult(BaseModel):
    """Single check outcome inside a :class:`ValidationReport`."""

    name: str = Field(..., description="Human-readable check name")
    passed: bool
    severity: Literal["blocking", "advisory"] = "blocking"
    source: str = Field(..., description="'builtin' or 'skill:<name>'")
    observed: Optional[Dict[str, Any]] = Field(default=None)
    expected: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None, description="Error message when the check itself failed to run")


class ValidationReport(BaseModel):
    """Aggregated validation outcome surfaced into ``NodeResult``."""

    target: Optional[Union[TableTarget, TransferTarget, SessionTarget]] = Field(
        default=None, description="The deliverable this report concerns"
    )
    checks: List[CheckResult] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Non-blocking issues the user should see (e.g. validator_skill_malformed). "
            "CLI layer should surface these alongside checks."
        ),
    )

    @classmethod
    def empty(cls, target: Optional[Union[TableTarget, TransferTarget, SessionTarget]] = None) -> "ValidationReport":
        return cls(target=target, checks=[], warnings=[])

    def has_blocking_failure(self) -> bool:
        """Return True if any check failed at blocking severity."""
        return any((not c.passed) and c.severity == "blocking" for c in self.checks)

    def merge(
        self,
        other: "ValidationReport",
        source: Optional[str] = None,
        severity_override: Optional[Literal["blocking", "advisory", "off"]] = None,
    ) -> "ValidationReport":
        """Merge another report into this one.

        Args:
            other: Report to merge in
            source: If set, override the ``source`` field on merged checks (used
                to tag checks with the originating skill name)
            severity_override: If set to ``"advisory"`` / ``"off"``, downgrade
                the merged checks' severity. ``"off"`` drops the checks entirely
                (the validator skill is declared off; its results are noise).
        """
        if severity_override == "off":
            return self
        for check in other.checks:
            new_check = check.model_copy()
            if source:
                new_check.source = source
            if severity_override == "advisory":
                new_check.severity = "advisory"
            self.checks.append(new_check)
        self.warnings.extend(other.warnings)
        return self

    def add_warning(self, warning: Dict[str, Any]) -> None:
        self.warnings.append(warning)

    def to_markdown(self) -> str:
        """Render the report as Markdown for injection back into the agent loop.

        Kept intentionally compact so retry prompts don't balloon in size.
        """
        lines: List[str] = []
        if self.target is not None:
            tgt = self.target
            if isinstance(tgt, TableTarget):
                lines.append(f"**Target:** table `{tgt.fqn}` on `{tgt.database}`")
            elif isinstance(tgt, TransferTarget):
                lines.append(f"**Target:** transfer `{tgt.source.name}` → `{tgt.target.database}.{tgt.target.fqn}`")
            elif isinstance(tgt, SessionTarget):
                lines.append(f"**Target:** session with {len(tgt.targets)} deliverable(s)")

        failed = [c for c in self.checks if not c.passed]
        passed = [c for c in self.checks if c.passed]

        if failed:
            lines.append("")
            lines.append(f"**Failing checks ({len(failed)}):**")
            for c in failed:
                sev = c.severity.upper()
                line = f"- [{sev}] {c.name} (source: {c.source})"
                if c.observed is not None:
                    line += f" — observed: {c.observed}"
                if c.expected is not None:
                    line += f"; expected: {c.expected}"
                if c.error:
                    line += f"; error: {c.error}"
                lines.append(line)

        if passed and not failed:
            lines.append("")
            lines.append(f"All {len(passed)} checks passed.")

        if self.warnings:
            lines.append("")
            lines.append("**Warnings:**")
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines) if lines else "(empty validation report)"


def skill_matches_target(
    targets: List[TargetFilter],
    target: Union[TableTarget, TransferTarget, SessionTarget],
) -> bool:
    """Decide whether a skill with the given ``targets`` frontmatter applies.

    Args:
        targets: The ``targets`` list from the skill's frontmatter (empty means
            match everything).
        target: The current deliverable (single target for ``on_tool_end`` or
            ``SessionTarget`` for ``on_end``).

    Returns:
        True when any filter matches, or when the filter list is empty. For
        :class:`SessionTarget` the skill matches if **any** contained target
        matches — that way ``on_end`` validators fire whenever relevant targets
        exist in the session.
    """
    if not targets:
        return True

    if isinstance(target, SessionTarget):
        return any(_filter_any_match(targets, t) for t in target.targets)

    return _filter_any_match(targets, target)


def _filter_any_match(
    filters: List[TargetFilter],
    target: Union[TableTarget, TransferTarget],
) -> bool:
    for flt in filters:
        if _filter_matches(flt, target):
            return True
    return False


def _filter_matches(flt: TargetFilter, target: Union[TableTarget, TransferTarget]) -> bool:
    """Single filter vs single target match. All set fields must match."""
    if flt.type and flt.type != target.type:
        return False
    if flt.database and flt.database != target.database:
        return False
    table_name: Optional[str] = None
    schema_name: Optional[str] = None
    if isinstance(target, TableTarget):
        table_name = target.table
        schema_name = target.db_schema
    elif isinstance(target, TransferTarget):
        table_name = target.target.table
        schema_name = target.target.db_schema
    if flt.db_schema and flt.db_schema != schema_name:
        return False
    if flt.table and flt.table != table_name:
        return False
    if flt.table_pattern:
        if not table_name or not fnmatch(table_name, flt.table_pattern):
            return False
    return True
