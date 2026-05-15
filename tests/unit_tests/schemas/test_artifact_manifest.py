# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for :class:`ArtifactManifest`.

The shared manifest is the contract that the Datus-SaaS list pages, the
CLI HTML compile, and the IDE explorer rely on. These tests pin the
required-field semantics so a future refactor doesn't silently allow
``Untitled report`` cards.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from datus.schemas.artifact_manifest import ArtifactManifest


def _base_payload(**overrides):
    base = {
        "slug": "q1_revenue_review",
        "name": "Q1 revenue review",
        "description": "Quarterly revenue review for the east region.",
        "kind": "report",
        "created_at": "2026-05-14T10:00:00Z",
    }
    base.update(overrides)
    return base


class TestArtifactManifest:
    def test_round_trip(self):
        manifest = ArtifactManifest.model_validate(_base_payload())
        assert manifest.slug == "q1_revenue_review"
        assert manifest.name == "Q1 revenue review"
        assert manifest.kind == "report"

    @pytest.mark.parametrize("kind", ["report", "dashboard"])
    def test_both_kinds_accepted(self, kind):
        manifest = ArtifactManifest.model_validate(_base_payload(kind=kind))
        assert manifest.kind == kind

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(kind="notebook"))

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(name=""))

    def test_empty_description_rejected(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(description=""))

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(rogue="oops"))

    def test_long_name_capped(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(name="x" * 201))

    def test_long_description_capped(self):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(description="x" * 1001))

    def test_chinese_name_accepted(self):
        manifest = ArtifactManifest.model_validate(_base_payload(name="销售季度复盘"))
        assert manifest.name == "销售季度复盘"

    @pytest.mark.parametrize(
        "bad_slug",
        [
            "",
            "Has-Hyphen",
            "has space",
            "Q1_Revenue",  # uppercase
            "中文",
            "a" * 81,
        ],
    )
    def test_invalid_slug_rejected(self, bad_slug):
        with pytest.raises(ValidationError):
            ArtifactManifest.model_validate(_base_payload(slug=bad_slug))
