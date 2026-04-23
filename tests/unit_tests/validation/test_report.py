# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.validation.report`` — data model + target matching."""

from __future__ import annotations

from datus.validation.report import (
    CheckResult,
    DBRef,
    SessionTarget,
    TableTarget,
    TargetFilter,
    TransferTarget,
    ValidationReport,
    skill_matches_target,
)


class TestTableTarget:
    def test_fqn_with_schema(self):
        t = TableTarget(database="db1", db_schema="public", table="users")
        assert t.fqn == "public.users"

    def test_fqn_without_schema(self):
        t = TableTarget(database="db1", table="flat")
        assert t.fqn == "flat"

    def test_yaml_alias_loads_schema_key(self):
        """User YAML writes ``schema:`` — the alias must flow through."""
        t = TableTarget.model_validate({"database": "db1", "schema": "analytics", "table": "rev"})
        assert t.db_schema == "analytics"

    def test_populate_by_name_also_works(self):
        t = TableTarget.model_validate({"database": "db1", "db_schema": "s", "table": "t"})
        assert t.db_schema == "s"

    def test_rows_affected_defaults_to_none(self):
        t = TableTarget(database="d", table="t")
        assert t.rows_affected is None


class TestTransferTarget:
    def test_database_proxy(self):
        """``.database`` on a TransferTarget resolves to the target table's db."""
        t = TransferTarget(
            source=DBRef(name="pg"),
            target=TableTarget(database="ch", table="f"),
        )
        assert t.database == "ch"


class TestValidationReport:
    def test_empty_report_has_no_blocking(self):
        r = ValidationReport.empty()
        assert not r.has_blocking_failure()
        assert r.checks == []

    def test_detects_blocking_failure(self):
        r = ValidationReport(
            target=None,
            checks=[
                CheckResult(name="ok", passed=True, source="builtin"),
                CheckResult(name="bad", passed=False, severity="blocking", source="builtin"),
            ],
        )
        assert r.has_blocking_failure()

    def test_advisory_failure_is_not_blocking(self):
        r = ValidationReport(
            target=None,
            checks=[CheckResult(name="hint", passed=False, severity="advisory", source="skill:x")],
        )
        assert not r.has_blocking_failure()

    def test_merge_severity_override_advisory(self):
        base = ValidationReport.empty()
        other = ValidationReport(
            target=None,
            checks=[CheckResult(name="x", passed=False, severity="blocking", source="builtin")],
        )
        base.merge(other, source="skill:foo", severity_override="advisory")
        assert not base.has_blocking_failure()
        assert base.checks[0].source == "skill:foo"
        assert base.checks[0].severity == "advisory"

    def test_merge_severity_override_off_drops_checks(self):
        base = ValidationReport.empty()
        other = ValidationReport(
            target=None,
            checks=[CheckResult(name="x", passed=False, source="builtin")],
        )
        base.merge(other, severity_override="off")
        assert base.checks == []

    def test_to_markdown_reports_failures(self):
        r = ValidationReport(
            target=TableTarget(database="d", db_schema="s", table="t"),
            checks=[
                CheckResult(
                    name="row_count",
                    passed=False,
                    severity="blocking",
                    source="builtin",
                    observed={"rows": 0},
                    expected={"rows_gt": 0},
                )
            ],
        )
        md = r.to_markdown()
        assert "row_count" in md
        assert "BLOCKING" in md
        assert "s.t" in md

    def test_warnings_preserved_through_merge(self):
        base = ValidationReport.empty()
        other = ValidationReport(target=None, checks=[])
        other.add_warning({"type": "validator_skill_malformed", "skill_name": "x"})
        base.merge(other)
        assert len(base.warnings) == 1
        assert base.warnings[0]["skill_name"] == "x"


class TestTargetFilter:
    def test_yaml_schema_alias(self):
        f = TargetFilter.model_validate({"type": "table", "schema": "staging"})
        assert f.db_schema == "staging"
        assert f.type == "table"

    def test_empty_filter_matches_all(self):
        assert skill_matches_target([], TableTarget(database="d", table="t"))
        assert skill_matches_target(
            [], TransferTarget(source=DBRef(name="s"), target=TableTarget(database="d", table="t"))
        )

    def test_type_only_filter_matches_transfer(self):
        filters = [TargetFilter(type="transfer")]
        tt = TransferTarget(source=DBRef(name="s"), target=TableTarget(database="d", table="t"))
        assert skill_matches_target(filters, tt)
        assert not skill_matches_target(filters, TableTarget(database="d", table="t"))

    def test_schema_plus_table_pattern(self):
        filters = [TargetFilter(type="table", db_schema="analytics", table_pattern="revenue_*")]
        assert skill_matches_target(filters, TableTarget(database="d", db_schema="analytics", table="revenue_daily"))
        assert not skill_matches_target(filters, TableTarget(database="d", db_schema="analytics", table="users"))
        assert not skill_matches_target(filters, TableTarget(database="d", db_schema="staging", table="revenue_daily"))

    def test_database_exact_match(self):
        filters = [TargetFilter(database="prod")]
        assert skill_matches_target(filters, TableTarget(database="prod", table="t"))
        assert not skill_matches_target(filters, TableTarget(database="staging", table="t"))

    def test_multiple_filters_any_match(self):
        """Filter list semantics: ANY match activates the skill."""
        filters = [TargetFilter(db_schema="staging"), TargetFilter(db_schema="raw")]
        assert skill_matches_target(filters, TableTarget(database="d", db_schema="raw", table="t"))
        assert skill_matches_target(filters, TableTarget(database="d", db_schema="staging", table="t"))
        assert not skill_matches_target(filters, TableTarget(database="d", db_schema="public", table="t"))

    def test_session_target_matches_when_any_inner_matches(self):
        filters = [TargetFilter(type="transfer")]
        session = SessionTarget(
            targets=[
                TableTarget(database="d", table="t"),  # no match
                TransferTarget(source=DBRef(name="s"), target=TableTarget(database="d", table="f")),  # match
            ]
        )
        assert skill_matches_target(filters, session)

    def test_session_target_no_inner_match(self):
        filters = [TargetFilter(type="transfer")]
        session = SessionTarget(targets=[TableTarget(database="d", table="t")])
        assert not skill_matches_target(filters, session)
