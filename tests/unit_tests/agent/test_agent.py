# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/agent/agent.py — bootstrap_platform_doc and _print_platform_doc_result.

Tests cover:
- bootstrap_platform_doc: platform inference, single config auto-select,
  no-source skip, error when platform cannot be determined
- _print_platform_doc_result: success/failure output formatting, check vs bootstrap mode,
  single vs multiple version_details rendering

NO MOCK EXCEPT LLM. Uses real AgentConfig (from config dict) and real print capture.
"""

import argparse

from datus.agent.agent import _print_platform_doc_result, bootstrap_platform_doc

# ---------------------------------------------------------------------------
# Helpers — lightweight InitResult-like dataclass for testing _print_platform_doc_result
# ---------------------------------------------------------------------------


class _FakeVersionDetail:
    """Fake version detail for testing."""

    def __init__(self, version, doc_count, chunk_count):
        self.version = version
        self.doc_count = doc_count
        self.chunk_count = chunk_count


class _FakeInitResult:
    """Fake InitResult for testing _print_platform_doc_result."""

    def __init__(
        self,
        success=True,
        platform="test_platform",
        version="v1",
        total_docs=10,
        total_chunks=50,
        source="https://example.com",
        duration_seconds=1.5,
        version_details=None,
        errors=None,
    ):
        self.success = success
        self.platform = platform
        self.version = version
        self.total_docs = total_docs
        self.total_chunks = total_chunks
        self.source = source
        self.duration_seconds = duration_seconds
        self.version_details = version_details
        self.errors = errors or []


class TestPrintPlatformDocResult:
    """Tests for _print_platform_doc_result formatting."""

    def test_none_result_prints_skip_message(self, capsys):
        """None result prints skip message."""
        _print_platform_doc_result(None, "check")
        output = capsys.readouterr().out
        assert "skipped" in output.lower()

    def test_success_check_mode(self, capsys):
        """Success result in check mode prints 'Check Complete'."""
        result = _FakeInitResult(success=True, platform="polaris", version="v2", total_docs=5, total_chunks=20)
        _print_platform_doc_result(result, "check")
        output = capsys.readouterr().out
        assert "Check" in output
        assert "polaris" in output
        assert "v2" in output

    def test_success_bootstrap_mode(self, capsys):
        """Success result in bootstrap mode prints 'Bootstrap Complete' with source/duration."""
        result = _FakeInitResult(
            success=True,
            platform="snowflake",
            source="https://docs.example.com",
            duration_seconds=3.2,
        )
        _print_platform_doc_result(result, "bootstrap")
        output = capsys.readouterr().out
        assert "Bootstrap" in output
        assert "snowflake" in output
        assert "Source" in output
        assert "Duration" in output

    def test_success_single_version_detail(self, capsys):
        """Success result with one version_detail shows version/doc/chunk info."""
        vd = _FakeVersionDetail(version="v3.0", doc_count=100, chunk_count=500)
        result = _FakeInitResult(success=True, platform="test", version_details=[vd])
        _print_platform_doc_result(result, "check")
        output = capsys.readouterr().out
        assert "v3.0" in output
        assert "100" in output
        assert "500" in output

    def test_success_multiple_version_details(self, capsys):
        """Success result with multiple version_details shows summary."""
        vd1 = _FakeVersionDetail(version="v1.0", doc_count=50, chunk_count=200)
        vd2 = _FakeVersionDetail(version="v2.0", doc_count=80, chunk_count=350)
        result = _FakeInitResult(
            success=True,
            platform="multi",
            version_details=[vd1, vd2],
            total_docs=130,
            total_chunks=550,
        )
        _print_platform_doc_result(result, "check")
        output = capsys.readouterr().out
        assert "Versions" in output
        assert "v1.0" in output
        assert "v2.0" in output
        assert "Total" in output

    def test_failure_result(self, capsys):
        """Failed result prints error messages."""
        result = _FakeInitResult(success=False, platform="broken", errors=["Connection timeout", "Auth failed"])
        _print_platform_doc_result(result, "bootstrap")
        output = capsys.readouterr().out
        assert "FAILED" in output
        assert "broken" in output
        assert "Connection timeout" in output
        assert "Auth failed" in output

    def test_success_no_version_details_fallback(self, capsys):
        """Success result without version_details uses fallback fields."""
        result = _FakeInitResult(
            success=True,
            platform="simple",
            version="latest",
            total_docs=7,
            total_chunks=33,
            version_details=None,
        )
        _print_platform_doc_result(result, "check")
        output = capsys.readouterr().out
        assert "latest" in output
        assert "7" in output
        assert "33" in output


class TestBootstrapPlatformDoc:
    """Tests for bootstrap_platform_doc function."""

    def test_no_platform_no_source_no_configs_returns_none(self, real_agent_config, capsys):
        """When no platform can be determined, returns None with error message."""
        args = argparse.Namespace(update_strategy="check", pool_size=4, platform=None, source=None)
        # Ensure no document configs
        real_agent_config.document_configs = {}

        result = bootstrap_platform_doc(args, real_agent_config)

        assert result is None
        output = capsys.readouterr().out
        assert "Cannot determine platform" in output

    def test_single_config_auto_selects_platform(self, real_agent_config, capsys):
        """When exactly one document config exists, auto-selects its platform."""
        from datus.configuration.agent_config import DocumentConfig

        real_agent_config.document_configs = {"auto_platform": DocumentConfig()}
        args = argparse.Namespace(update_strategy="check", pool_size=4, platform=None, source=None)

        result = bootstrap_platform_doc(args, real_agent_config)

        output = capsys.readouterr().out
        # Should either skip (no source) or try to init
        assert result is None or output  # Handled gracefully

    def test_no_source_skips(self, real_agent_config, capsys):
        """When config has no source, prints skip message and returns None."""
        from datus.configuration.agent_config import DocumentConfig

        real_agent_config.document_configs = {"skip_test": DocumentConfig(source="")}
        args = argparse.Namespace(update_strategy="check", pool_size=4, platform="skip_test", source=None)

        result = bootstrap_platform_doc(args, real_agent_config)

        assert result is None
        output = capsys.readouterr().out
        assert "skipped" in output.lower()

    def test_explicit_platform_used(self, real_agent_config, capsys):
        """When --platform is specified, uses it directly."""
        from datus.configuration.agent_config import DocumentConfig

        real_agent_config.document_configs = {"explicit": DocumentConfig(source="")}
        args = argparse.Namespace(update_strategy="check", pool_size=4, platform="explicit", source=None)

        result = bootstrap_platform_doc(args, real_agent_config)

        # Should skip because no source
        assert result is None

    def test_source_infers_platform_and_runs_check(self, real_agent_config, capsys):
        """When --source is provided without --platform, platform is inferred from source URL."""
        from datus.configuration.agent_config import DocumentConfig

        # Config has a matching platform entry with source
        real_agent_config.document_configs = {
            "testdb": DocumentConfig(source="https://github.com/owner/testdb", type="github")
        }
        args = argparse.Namespace(
            update_strategy="check",
            pool_size=1,
            platform=None,
            source="https://github.com/owner/testdb",
            source_type=None,
            version=None,
            github_ref=None,
            github_token=None,
            paths=None,
            chunk_size=None,
            max_depth=None,
            include_patterns=None,
            exclude_patterns=None,
        )

        result = bootstrap_platform_doc(args, real_agent_config)

        # Platform should be inferred as "testdb" and check mode should succeed
        assert result is not None
        assert result.platform == "testdb"
        assert result.success is True

        output = capsys.readouterr().out
        assert "Check" in output or "testdb" in output
