"""Unit tests for datus.api.models.kb_models — BootstrapDocInput."""

import pytest
from pydantic import ValidationError

from datus.api.models.kb_models import BootstrapDocInput


class TestBootstrapDocInput:
    """Tests for BootstrapDocInput validation."""

    def test_minimal_creation(self):
        """Only platform is required; everything else has defaults or is optional."""
        inp = BootstrapDocInput(platform="snowflake")
        assert inp.platform == "snowflake"
        assert inp.build_mode == "overwrite"
        assert inp.pool_size == 4
        assert inp.source is None
        assert inp.source_type is None

    def test_all_fields(self):
        """All optional fields can be set."""
        inp = BootstrapDocInput(
            platform="duckdb",
            build_mode="check",
            pool_size=8,
            source_type="github",
            source="owner/repo",
            version="1.0.0",
            github_ref="main",
            paths=["docs"],
            chunk_size=2048,
            max_depth=3,
            include_patterns=["*.md"],
            exclude_patterns=["changelog*"],
        )
        assert inp.build_mode == "check"
        assert inp.pool_size == 8
        assert inp.source_type == "github"
        assert inp.paths == ["docs"]

    def test_platform_required(self):
        """Missing platform raises ValidationError."""
        with pytest.raises(ValidationError):
            BootstrapDocInput()

    def test_build_mode_literal(self):
        """build_mode only accepts 'overwrite' or 'check'."""
        with pytest.raises(ValidationError):
            BootstrapDocInput(platform="test", build_mode="invalid")

    def test_pool_size_min(self):
        """pool_size must be >= 1."""
        with pytest.raises(ValidationError):
            BootstrapDocInput(platform="test", pool_size=0)

    def test_pool_size_max(self):
        """pool_size must be <= 16."""
        with pytest.raises(ValidationError):
            BootstrapDocInput(platform="test", pool_size=17)

    def test_pool_size_boundaries(self):
        """pool_size accepts boundary values 1 and 16."""
        inp1 = BootstrapDocInput(platform="test", pool_size=1)
        inp16 = BootstrapDocInput(platform="test", pool_size=16)
        assert inp1.pool_size == 1
        assert inp16.pool_size == 16

    def test_json_roundtrip(self):
        """Model serializes to and deserializes from JSON."""
        inp = BootstrapDocInput(platform="pg", source="owner/repo", version="2.0")
        data = inp.model_dump()
        restored = BootstrapDocInput(**data)
        assert restored.platform == "pg"
        assert restored.source == "owner/repo"
