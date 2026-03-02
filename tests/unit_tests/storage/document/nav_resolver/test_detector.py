# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.document.nav_resolver.detector.

The detect() method requires GitHub API access and cannot be tested at CI
level without mocking external services (which violates the NO MOCK principle).
We therefore focus on constants, dataclass defaults, and _ProbeSpec definitions.
"""

from datus.storage.document.nav_resolver.detector import (
    _PROBE_SPECS,
    FRAMEWORK_DOCUSAURUS,
    FRAMEWORK_HUGO,
    FRAMEWORK_MKDOCS,
    FRAMEWORK_UNKNOWN,
    DocFrameworkDetector,
    FrameworkInfo,
    _ProbeSpec,
)

# ---------------------------------------------------------------------------
# Framework constants
# ---------------------------------------------------------------------------


class TestFrameworkConstants:
    """Tests for module-level framework name constants."""

    def test_framework_docusaurus_value(self):
        """FRAMEWORK_DOCUSAURUS should be 'docusaurus'."""
        assert FRAMEWORK_DOCUSAURUS == "docusaurus"

    def test_framework_hugo_value(self):
        """FRAMEWORK_HUGO should be 'hugo'."""
        assert FRAMEWORK_HUGO == "hugo"

    def test_framework_mkdocs_value(self):
        """FRAMEWORK_MKDOCS should be 'mkdocs'."""
        assert FRAMEWORK_MKDOCS == "mkdocs"

    def test_framework_unknown_value(self):
        """FRAMEWORK_UNKNOWN should be 'unknown'."""
        assert FRAMEWORK_UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# FrameworkInfo defaults
# ---------------------------------------------------------------------------


class TestFrameworkInfo:
    """Tests for FrameworkInfo dataclass."""

    def test_default_framework(self):
        """Default framework should be FRAMEWORK_UNKNOWN."""
        info = FrameworkInfo()
        assert info.framework == FRAMEWORK_UNKNOWN

    def test_default_config_path(self):
        """Default config_path should be empty string."""
        info = FrameworkInfo()
        assert info.config_path == ""

    def test_default_content_root(self):
        """Default content_root should be empty string."""
        info = FrameworkInfo()
        assert info.content_root == ""

    def test_default_config_content(self):
        """Default config_content should be empty string."""
        info = FrameworkInfo()
        assert info.config_content == ""

    def test_custom_values(self):
        """FrameworkInfo should accept custom values for all fields."""
        info = FrameworkInfo(
            framework=FRAMEWORK_MKDOCS,
            config_path="mkdocs.yml",
            content_root="docs/",
            config_content="site_name: Test",
        )
        assert info.framework == FRAMEWORK_MKDOCS
        assert info.config_path == "mkdocs.yml"
        assert info.content_root == "docs/"
        assert info.config_content == "site_name: Test"


# ---------------------------------------------------------------------------
# _ProbeSpec
# ---------------------------------------------------------------------------


class TestProbeSpec:
    """Tests for _ProbeSpec dataclass."""

    def test_probe_spec_defaults(self):
        """content_root_candidates should default to empty list."""
        spec = _ProbeSpec(framework="test", config_candidates=["test.yml"])
        assert spec.content_root_candidates == []

    def test_probe_spec_custom(self):
        """All fields should be settable."""
        spec = _ProbeSpec(
            framework="custom",
            config_candidates=["a.yml", "b.toml"],
            content_root_candidates=["docs/", "content/"],
        )
        assert spec.framework == "custom"
        assert spec.config_candidates == ["a.yml", "b.toml"]
        assert spec.content_root_candidates == ["docs/", "content/"]


# ---------------------------------------------------------------------------
# _PROBE_SPECS ordering and content
# ---------------------------------------------------------------------------


class TestProbeSpecs:
    """Tests for the global _PROBE_SPECS list."""

    def test_probe_specs_length(self):
        """There should be exactly 3 probe specs."""
        assert len(_PROBE_SPECS) == 3

    def test_probe_order_mkdocs_first(self):
        """MkDocs should be probed first (root-level config is quickest to detect)."""
        assert _PROBE_SPECS[0].framework == FRAMEWORK_MKDOCS

    def test_probe_order_hugo_second(self):
        """Hugo should be probed second."""
        assert _PROBE_SPECS[1].framework == FRAMEWORK_HUGO

    def test_probe_order_docusaurus_third(self):
        """Docusaurus should be probed third."""
        assert _PROBE_SPECS[2].framework == FRAMEWORK_DOCUSAURUS

    def test_mkdocs_config_candidates(self):
        """MkDocs should check mkdocs.yml and mkdocs.yaml."""
        candidates = _PROBE_SPECS[0].config_candidates
        assert "mkdocs.yml" in candidates
        assert "mkdocs.yaml" in candidates

    def test_hugo_config_candidates(self):
        """Hugo should check multiple config file names."""
        candidates = _PROBE_SPECS[1].config_candidates
        assert "hugo.yaml" in candidates
        assert "hugo.toml" in candidates

    def test_docusaurus_config_candidates(self):
        """Docusaurus should check sidebars.json and sidebars.js."""
        candidates = _PROBE_SPECS[2].config_candidates
        assert "sidebars.json" in candidates
        assert "sidebars.js" in candidates


# ---------------------------------------------------------------------------
# DocFrameworkDetector instantiation
# ---------------------------------------------------------------------------


class TestDocFrameworkDetector:
    """Tests for DocFrameworkDetector that do not require GitHub API."""

    def test_detector_can_be_instantiated(self):
        """DocFrameworkDetector should be instantiable without arguments."""
        detector = DocFrameworkDetector()
        assert detector is not None
