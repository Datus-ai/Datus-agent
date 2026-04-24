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
    catalog: Optional[str] = Field(
        default=None,
        description="Catalog name for three-part identifiers (e.g. StarRocks default_catalog); None when not applicable",
    )
    datasource: Optional[str] = Field(
        default=None,
        description=(
            "Datasource key used to route validator queries to the right connector. "
            "Distinct from ``database``: when the mutating tool writes to a non-default "
            "datasource the validator must hit that same connector rather than falling "
            "through to the parent node's default."
        ),
    )
    database: str = Field(
        ...,
        description="Database identifier — historically the datasource key; may also be a physical DB for three-part DDL",
    )
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
        """Fully qualified name (schema.table or just table). Catalog is intentionally
        excluded — consumers that care about catalog read it from ``self.catalog``.
        """
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
                the merged checks' severity. ``"off"`` discards the merged
                report in full — both ``checks`` and ``warnings`` — because
                the validator skill is declared off and its entire output is
                noise (including meta-warnings like
                ``validator_skill_malformed``). ``"advisory"`` downgrades
                only the checks; warnings still flow through.
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


def describe_target(target: Union[TableTarget, TransferTarget, SessionTarget]) -> str:
    """Human-readable descriptor used to tag checks and render retry prompts."""
    if isinstance(target, TableTarget):
        prefix = f"{target.catalog}." if target.catalog else ""
        return f"table {prefix}{target.database}.{target.fqn}"
    if isinstance(target, TransferTarget):
        prefix = f"{target.target.catalog}." if target.target.catalog else ""
        return f"transfer {target.source.name} -> {prefix}{target.target.database}.{target.target.fqn}"
    if isinstance(target, SessionTarget):
        return f"session[{len(target.targets)}]"
    return repr(target)


def build_retry_prompt(
    final_report: ValidationReport,
    session_targets: List[Union[TableTarget, TransferTarget]],
) -> str:
    """Render a structured retry message that separates already-committed
    correct targets from the ones that need fixing.

    The agent receives this as a user message on the next attempt. Session
    history still carries its own tool-call record, so the agent can
    cross-reference which CREATE/INSERT/transfer already ran.
    """
    # Group checks by their ``_target`` tag (set by builtin_checks during
    # SessionTarget recursion). Skill-based checks may not carry the tag; they
    # go under "session" and are appended whole.
    target_checks: Dict[str, List[CheckResult]] = {}
    untagged: List[CheckResult] = []
    for c in final_report.checks:
        tag = (c.observed or {}).get("_target") if c.observed else None
        if isinstance(tag, str):
            target_checks.setdefault(tag, []).append(c)
        else:
            untagged.append(c)

    ok_targets: List[Union[TableTarget, TransferTarget]] = []
    failed_targets: List[tuple] = []  # (target, checks)
    for target in session_targets:
        tag = describe_target(target)
        checks = target_checks.get(tag, [])
        has_blocking = any((not c.passed) and c.severity == "blocking" for c in checks)
        if has_blocking:
            failed_targets.append((target, checks))
        else:
            ok_targets.append(target)

    lines: List[str] = [
        "The run was blocked by on_end validation. Please fix and retry.",
        "",
    ]

    if ok_targets:
        lines.append("## Already written and correct — DO NOT recreate:")
        for t in ok_targets:
            lines.append(f"  - {describe_target(t)}")
        lines.append("")

    if failed_targets:
        lines.append("## Failed targets — fix these:")
        for t, checks in failed_targets:
            lines.append(f"### {describe_target(t)}")
            for c in checks:
                if c.passed:
                    continue
                line = f"  - **{c.name}** ({c.severity}) failed"
                if c.observed:
                    filtered = {k: v for k, v in c.observed.items() if k != "_target"}
                    if filtered:
                        line += f" — observed: {filtered}"
                if c.expected:
                    line += f"; expected: {c.expected}"
                if c.error:
                    line += f"; error: {c.error}"
                lines.append(line)
            lines.append(
                "  Repair hint: if the table already exists but has the wrong schema, use "
                "ALTER TABLE or DROP + CREATE; if it doesn't exist yet, CREATE it. "
                "For transfer row-count mismatches, re-check the source query "
                "and either re-transfer the missing rows or rewrite the filter."
            )
            lines.append("")

    if untagged:
        lines.append("## Session-level findings:")
        for c in untagged:
            if c.passed:
                continue
            line = f"  - [{c.severity.upper()}] {c.name} (source: {c.source})"
            if c.observed:
                line += f" — observed: {c.observed}"
            if c.error:
                line += f"; error: {c.error}"
            lines.append(line)
        lines.append("")

    if final_report.warnings:
        lines.append("## Warnings:")
        for w in final_report.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("---")
    lines.append("Full report:")
    lines.append(final_report.to_markdown())
    return "\n".join(lines)
