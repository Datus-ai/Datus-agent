# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for gen_visual_dashboard pydantic schemas + header parser.

Pins the wire contract that ``DashboardArtifactTools`` (save-time) and
``Datus-backend.services.dashboard_service`` (view-time) both rely on:

* ``parse_datus_params_header`` — extracts typed parameter declarations
  from the ``-- @datus-params`` header at the top of every saved template.
* ``TemplateParamDecl`` — name regex, type enum, array helpers.
* ``QueryTemplateMetaFile`` — the on-disk ``<slug>.params.json`` shape.
* Node IO models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datus.schemas.gen_visual_dashboard_models import (
    GenVisualDashboardNodeInput,
    GenVisualDashboardNodeResult,
    QueryTemplateMetaFile,
    TemplateParamDecl,
    parse_datus_params_header,
)

# ----------------------------------------------------------------------------- #
# parse_datus_params_header                                                     #
# ----------------------------------------------------------------------------- #


class TestParseDatusParamsHeader:
    def test_single_required_param(self):
        decls = parse_datus_params_header("-- @datus-params start_date:date\nSELECT 1")
        assert len(decls) == 1
        assert decls[0].name == "start_date"
        assert decls[0].type == "date"
        assert decls[0].required is True
        assert decls[0].is_array is False
        assert decls[0].base_type == "date"

    def test_multiple_mixed_params(self):
        decls = parse_datus_params_header(
            "-- @datus-params start_date:date, end_date:date, regions:string[]:optional, min_amount:number\nSELECT 1"
        )
        names = [d.name for d in decls]
        assert names == ["start_date", "end_date", "regions", "min_amount"]
        types = [d.type for d in decls]
        assert types == ["date", "date", "string[]", "number"]
        required = [d.required for d in decls]
        assert required == [True, True, False, True]

    def test_optional_short_form(self):
        decls = parse_datus_params_header("-- @datus-params flag:boolean?\nSELECT 1")
        assert decls[0].required is False

    def test_array_helpers(self):
        decls = parse_datus_params_header("-- @datus-params xs:integer[]\nSELECT 1")
        assert decls[0].is_array is True
        assert decls[0].base_type == "integer"

    def test_leading_blank_lines_ok(self):
        decls = parse_datus_params_header("\n\n-- @datus-params x:string\nSELECT 1")
        assert decls[0].name == "x"

    def test_missing_header_rejected(self):
        with pytest.raises(ValueError, match="missing the required ``-- @datus-params"):
            parse_datus_params_header("SELECT 1")

    def test_other_leading_comment_rejected(self):
        with pytest.raises(ValueError, match="not a ``-- @datus-params"):
            parse_datus_params_header("-- description\n-- @datus-params x:string\nSELECT 1")

    def test_empty_header_accepted_as_no_params(self):
        # ``-- @datus-params`` with no body is the canonical way to declare
        # "this template takes no parameters" — required so the agent can
        # save a genuinely static query (e.g. a catalog rollup) without
        # tripping the "every declared param must be bound in the SQL body"
        # check in save_query_template.
        assert parse_datus_params_header("-- @datus-params\nSELECT 1") == []
        # Stray comma in the body is equivalent — still zero declared params.
        assert parse_datus_params_header("-- @datus-params ,\nSELECT 1") == []
        # Trailing whitespace tolerated.
        assert parse_datus_params_header("-- @datus-params   \nSELECT 1") == []

    def test_glued_header_rejected(self):
        # The ``\s+(...)`` separator between ``@datus-params`` and its body
        # is mandatory when a body is present — ``-- @datus-paramsx:date``
        # is not a legal header. Without this guard the regex would treat
        # the keyword as a prefix of an identifier and silently parse the
        # rest as a declaration. Falls through to the "not a -- @datus-params
        # declaration" branch (a malformed header looks like a non-header
        # comment to the parser).
        with pytest.raises(ValueError, match="not a ``-- @datus-params"):
            parse_datus_params_header("-- @datus-paramsstart_date:date\nSELECT :start_date")
        # Tab also counts as the separator (Python ``\s`` matches tabs).
        decls = parse_datus_params_header("-- @datus-params\tstart_date:date\nSELECT :start_date")
        assert decls[0].name == "start_date"

    def test_malformed_declaration_rejected(self):
        with pytest.raises(ValueError, match="malformed"):
            parse_datus_params_header("-- @datus-params bad:hex\nSELECT 1")

    def test_duplicate_name_rejected(self):
        with pytest.raises(ValueError, match="Duplicate param"):
            parse_datus_params_header("-- @datus-params dup:string, dup:integer\nSELECT 1")


# ----------------------------------------------------------------------------- #
# TemplateParamDecl                                                             #
# ----------------------------------------------------------------------------- #


class TestTemplateParamDecl:
    def test_required_defaults_true(self):
        decl = TemplateParamDecl(name="x", type="integer")
        assert decl.required is True

    @pytest.mark.parametrize(
        "type_, expected_array, expected_base",
        [
            ("string", False, "string"),
            ("integer[]", True, "integer"),
            ("date[]", True, "date"),
            ("boolean", False, "boolean"),
        ],
    )
    def test_array_helpers(self, type_, expected_array, expected_base):
        decl = TemplateParamDecl(name="x", type=type_)
        assert decl.is_array is expected_array
        assert decl.base_type == expected_base

    def test_name_must_match_regex(self):
        with pytest.raises(ValidationError):
            TemplateParamDecl(name="1bad", type="string")

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            TemplateParamDecl(name="x", type="varchar")


# ----------------------------------------------------------------------------- #
# QueryTemplateMetaFile                                                         #
# ----------------------------------------------------------------------------- #


class TestQueryTemplateMetaFile:
    def _payload(self, **overrides):
        base = dict(
            slug="revenue_by_region",
            description="",
            datasource="warehouse",
            params=[{"name": "start_date", "type": "date", "required": True}],
            columns=[{"name": "region", "type": "string"}, {"name": "r", "type": "number"}],
            sample_params={"start_date": "2026-01-01"},
            sample_row_count=2,
            saved_at="2026-05-14T10:00:00Z",
        )
        base.update(overrides)
        return base

    def test_round_trip(self):
        meta = QueryTemplateMetaFile.model_validate(self._payload())
        assert meta.slug == "revenue_by_region"
        assert meta.params[0].name == "start_date"
        assert meta.columns[1].type == "number"

    def test_columns_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            QueryTemplateMetaFile.model_validate(self._payload(columns=[]))

    def test_duplicate_param_names_rejected(self):
        with pytest.raises(ValidationError, match="duplicate param"):
            QueryTemplateMetaFile.model_validate(
                self._payload(
                    params=[
                        {"name": "x", "type": "string", "required": True},
                        {"name": "x", "type": "integer", "required": True},
                    ]
                )
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            QueryTemplateMetaFile.model_validate(self._payload(rogue="oops"))


# ----------------------------------------------------------------------------- #
# Input / Result models                                                         #
# ----------------------------------------------------------------------------- #


class TestNodeIO:
    def test_input_requires_user_message(self):
        with pytest.raises(ValidationError):
            GenVisualDashboardNodeInput()  # type: ignore[call-arg]

    def test_input_round_trip(self):
        inp = GenVisualDashboardNodeInput(user_message="build me a dashboard", catalog="c", database="d", db_schema="s")
        assert inp.user_message == "build me a dashboard"
        assert inp.catalog == "c"
        assert inp.db_schema == "s"

    def test_result_defaults(self):
        result = GenVisualDashboardNodeResult(success=True)
        assert result.success is True
        assert result.response == ""
        assert result.dashboard_slug is None
        assert result.app_jsx_path is None
        assert result.render_file_count == 0
        assert result.template_count == 0
        assert result.tokens_used == 0

    def test_result_with_app_jsx_path(self):
        result = GenVisualDashboardNodeResult(
            success=True,
            dashboard_slug="demo",
            app_jsx_path="dashboards/demo/render/app.jsx",
            render_file_count=4,
            template_count=3,
        )
        assert result.app_jsx_path == "dashboards/demo/render/app.jsx"
        assert result.render_file_count == 4
        assert result.template_count == 3
