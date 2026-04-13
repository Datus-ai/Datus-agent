# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from pathlib import Path

from datus.tools.skill_tools.skill_config import SkillConfig
from datus.tools.skill_tools.skill_manager import SkillManager
from datus.validation.registry import (
    OutputColumnsMatchContractValidator,
    PinnedDateLiteralValidator,
    SqlParseableValidator,
    SqlPatternValidator,
    ValidatorRegistry,
    run_validators,
)


class TestSqlParseableValidator:
    def test_valid_sql_passes(self):
        validator = SqlParseableValidator()
        result = validator.validate("SELECT 1 AS one", {"db_type": "sqlite"})
        assert result.passed is True
        assert result.issues == []

    def test_invalid_sql_fails(self):
        validator = SqlParseableValidator()
        result = validator.validate("SELECT FROM", {"db_type": "sqlite"})
        assert result.passed is False
        assert result.issues
        assert result.issues[0].code == "sql_parseable"


class TestOutputColumnsMatchContractValidator:
    def test_exact_match_passes(self):
        validator = OutputColumnsMatchContractValidator()
        result = validator.validate(
            "SELECT cds AS school_id, AvgScrRead AS avg_reading FROM satscores",
            {"db_type": "sqlite", "expected_output_columns": ["school_id", "avg_reading"]},
        )
        assert result.passed is True

    def test_missing_column_fails(self):
        validator = OutputColumnsMatchContractValidator()
        result = validator.validate(
            "SELECT cds AS school_id FROM satscores",
            {"db_type": "sqlite", "expected_output_columns": ["school_id", "avg_reading"]},
        )
        assert result.passed is False
        assert any("missing expected columns" in issue.message for issue in result.issues)

    def test_extra_column_fails(self):
        validator = OutputColumnsMatchContractValidator()
        result = validator.validate(
            "SELECT cds AS school_id, AvgScrRead AS avg_reading, AvgScrMath AS avg_math FROM satscores",
            {"db_type": "sqlite", "expected_output_columns": ["school_id", "avg_reading"]},
        )
        assert result.passed is False
        assert any("unexpected columns" in issue.message for issue in result.issues)

    def test_order_mismatch_fails(self):
        validator = OutputColumnsMatchContractValidator()
        result = validator.validate(
            "SELECT AvgScrRead AS avg_reading, cds AS school_id FROM satscores",
            {"db_type": "sqlite", "expected_output_columns": ["school_id", "avg_reading"]},
        )
        assert result.passed is False
        assert any("column order does not match" in issue.message for issue in result.issues)

    def test_missing_context_skips(self):
        validator = OutputColumnsMatchContractValidator()
        result = validator.validate("SELECT 1 AS one", {"db_type": "sqlite"})
        assert result.passed is True
        assert result.metadata == {"skipped": True, "reason": "missing_expected_columns"}


class TestSqlPatternValidator:
    def test_forbidden_pattern_fails(self):
        validator = SqlPatternValidator("no_select_star", must_not_contain=["select *"])
        result = validator.validate("SELECT * FROM satscores", {"db_type": "sqlite"})
        assert result.passed is False
        assert any("forbidden patterns" in issue.message for issue in result.issues)

    def test_required_pattern_passes(self):
        validator = SqlPatternValidator("must_filter", must_contain=["where"])
        result = validator.validate("SELECT cds FROM satscores WHERE AvgScrRead > 500", {"db_type": "sqlite"})
        assert result.passed is True


class TestPinnedDateLiteralValidator:
    def test_dynamic_current_date_fails(self):
        validator = PinnedDateLiteralValidator("pinned_date_literal")
        result = validator.validate(
            "SELECT CURRENT_DATE AS as_of_date",
            {"db_type": "sqlite", "reference_date": "2025-10-27"},
        )
        assert result.passed is False
        assert any("dynamic current-date functions" in issue.message for issue in result.issues)

    def test_pinned_date_literal_passes(self):
        validator = PinnedDateLiteralValidator("pinned_date_literal")
        result = validator.validate(
            "SELECT DATE '2025-10-27' AS as_of_date",
            {"db_type": "sqlite", "reference_date": "2025-10-27"},
        )
        assert result.passed is True

    def test_missing_reference_date_skips(self):
        validator = PinnedDateLiteralValidator("pinned_date_literal")
        result = validator.validate("SELECT CURRENT_DATE AS as_of_date", {"db_type": "sqlite"})
        assert result.passed is True
        assert result.metadata == {"skipped": True, "reason": "missing_reference_date"}


class TestValidatorRegistry:
    def test_loads_skill_declared_validator(self):
        repo_root = Path(__file__).resolve().parents[3]
        skill_manager = SkillManager(config=SkillConfig(directories=[str(repo_root / "skills")]))
        registry = ValidatorRegistry(skill_manager=skill_manager)

        validators = registry.build_validators(
            node_name="gensql",
            skill_patterns=["dbt-layered-generation"],
            context={"expected_output_columns": ["foo"]},
        )

        validator_ids = [validator.validator_id for validator in validators]
        assert "sql_parseable" in validator_ids
        assert "output_columns_match_contract" in validator_ids
        assert "no_select_star" in validator_ids

    def test_loads_pinned_date_validator_only_when_reference_date_is_available(self):
        repo_root = Path(__file__).resolve().parents[3]
        skill_manager = SkillManager(config=SkillConfig(directories=[str(repo_root / "skills")]))
        registry = ValidatorRegistry(skill_manager=skill_manager)

        validators = registry.build_validators(
            node_name="gensql",
            skill_patterns=["dbt-layered-generation"],
            context={"expected_output_columns": ["foo"], "reference_date": "2025-10-27"},
        )

        validator_ids = [validator.validator_id for validator in validators]
        assert "pinned_date_literal" in validator_ids

    def test_run_validators_combines_issues(self):
        validators = [SqlParseableValidator(), OutputColumnsMatchContractValidator()]
        result = run_validators(
            validators,
            "SELECT FROM",
            {"db_type": "sqlite", "expected_output_columns": ["school_id"]},
        )
        assert result.passed is False
        assert len(result.issues) >= 1
