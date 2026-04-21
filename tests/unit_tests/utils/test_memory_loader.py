# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/utils/memory_loader.py"""

import logging
from unittest.mock import patch

from datus.utils.memory_loader import (
    MEMORY_BASE_DIR,
    MEMORY_BYTE_LIMIT,
    MEMORY_FILENAME,
    MEMORY_LINE_LIMIT,
    get_memory_dir,
    has_memory,
    load_memory_context,
)


class TestHasMemory:
    """Tests for has_memory()."""

    def test_chat_has_memory(self):
        assert has_memory("chat") is True

    def test_custom_agent_has_memory(self):
        assert has_memory("my_custom_agent") is True

    def test_gen_sql_no_memory(self):
        assert has_memory("gen_sql") is False

    def test_gen_report_no_memory(self):
        assert has_memory("gen_report") is False

    def test_gen_semantic_model_no_memory(self):
        assert has_memory("gen_semantic_model") is False

    def test_gen_metrics_no_memory(self):
        assert has_memory("gen_metrics") is False

    def test_gen_sql_summary_no_memory(self):
        assert has_memory("gen_sql_summary") is False

    def test_gen_ext_knowledge_no_memory(self):
        assert has_memory("gen_ext_knowledge") is False

    def test_explore_no_memory(self):
        assert has_memory("explore") is False

    def test_compare_no_memory(self):
        assert has_memory("compare") is False


class TestLoadMemoryContext:
    """Tests for load_memory_context()."""

    def test_file_not_found_returns_empty(self, tmp_path):
        assert load_memory_context(str(tmp_path), "chat") == ""

    def test_normal_content(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text("# Memory\n\n- item 1\n- item 2\n", encoding="utf-8")

        result = load_memory_context(str(tmp_path), "chat")
        assert "# Memory" in result
        assert "- item 1" in result
        assert "- item 2" in result

    def test_truncation_at_limit(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME

        # Index-shaped lines so the structure check stays quiet and we
        # isolate the line-cap assertion from the prose-run warning.
        lines = [f"- [entry {i}](topic_{i}.md) — hook {i}" for i in range(MEMORY_LINE_LIMIT + 50)]
        memory_file.write_text("\n".join(lines), encoding="utf-8")

        result = load_memory_context(str(tmp_path), "chat")
        result_lines = result.splitlines()

        # Should have 200 original lines + empty separator + truncation notice = 202
        assert len(result_lines) == MEMORY_LINE_LIMIT + 2
        assert "truncated" in result_lines[-1]
        assert f"entry {MEMORY_LINE_LIMIT - 1}" in result
        # entry 200 must NOT be present (0-indexed, so entry_200 is the 201st)
        assert f"entry {MEMORY_LINE_LIMIT} " not in result

    def test_exact_limit_not_truncated(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME

        lines = [f"- [entry {i}](topic_{i}.md) — hook" for i in range(MEMORY_LINE_LIMIT)]
        memory_file.write_text("\n".join(lines), encoding="utf-8")

        result = load_memory_context(str(tmp_path), "chat")
        assert "truncated" not in result

    def test_empty_file(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text("", encoding="utf-8")

        result = load_memory_context(str(tmp_path), "chat")
        assert result == ""

    def test_custom_agent_memory(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "my_agent"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text("# Custom Agent Memory\n", encoding="utf-8")

        result = load_memory_context(str(tmp_path), "my_agent")
        assert "Custom Agent Memory" in result

    def test_os_error_returns_empty(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text("content", encoding="utf-8")

        with patch("pathlib.Path.open", side_effect=OSError("Permission denied")):
            result = load_memory_context(str(tmp_path), "chat")
        assert result == ""

    def test_unicode_error_returns_empty(self, tmp_path):
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")

        result = load_memory_context(str(tmp_path), "chat")
        # Should return empty string or partial content, not raise
        assert isinstance(result, str)

    def test_byte_truncation_triggers_even_under_line_limit(self, tmp_path):
        """Long lines can blow the byte cap while staying under 200 lines."""
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME

        # 50 lines × 800 chars each ≈ 40 KB — well above MEMORY_BYTE_LIMIT (25 KB)
        # but well below MEMORY_LINE_LIMIT (200 lines).
        long_line = "- [topic](t.md) — " + ("x" * 780)
        memory_file.write_text("\n".join([long_line] * 50), encoding="utf-8")

        result = load_memory_context(str(tmp_path), "chat")
        assert "truncated" in result
        # Byte-truncated content must not exceed the cap by much (warning adds <300 bytes).
        assert len(result.encode("utf-8")) <= MEMORY_BYTE_LIMIT + 500
        assert "bytes" in result  # reason mentions byte unit

    def test_long_section_triggers_split_warning(self, tmp_path, caplog):
        """A section (under one `##` heading) longer than 50 lines should
        warn that the memory has outgrown inline storage."""
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        body = "\n".join(f"- detail line {i}" for i in range(60))
        memory_file.write_text(f"# Memory\n\n## Big topic\n{body}\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="datus.utils.memory_loader"):
            load_memory_context(str(tmp_path), "chat")

        assert any("longer than 50 lines" in r.message for r in caplog.records)

    def test_short_section_does_not_trigger_split_warning(self, tmp_path, caplog):
        """A section under the 50-line threshold should stay inline silently.

        Paragraph text, sub-bullets, and small tables are all fine in
        MEMORY.md as long as a single memory doesn't outgrow ~50 lines.
        """
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text(
            "# Memory\n\n"
            "## Testing conventions\n"
            "- Integration tests must hit a real database, not mocks.\n"
            "  **Why:** prior incident where mock/prod divergence masked a broken migration.\n"
            "  **How to apply:** when writing tests under tests/integration/, skip mock fixtures.\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="datus.utils.memory_loader"):
            load_memory_context(str(tmp_path), "chat")

        assert not any("longer than 50 lines" in r.message for r in caplog.records)

    def test_long_single_line_triggers_warning(self, tmp_path, caplog):
        """Any line longer than 200 chars should emit a soft warning."""
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        long_entry = "- [entry](topic.md) — " + ("x" * 250)
        memory_file.write_text(f"# Memory\n\n{long_entry}\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="datus.utils.memory_loader"):
            load_memory_context(str(tmp_path), "chat")

        assert any("over 200 chars" in r.message for r in caplog.records)

    def test_clean_index_emits_no_warnings(self, tmp_path, caplog):
        """A well-formed MEMORY.md should produce no structural warnings."""
        memory_dir = tmp_path / MEMORY_BASE_DIR / "chat"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / MEMORY_FILENAME
        memory_file.write_text(
            "# Memory\n\n"
            "## Preferences\n"
            "- [terse output](terse.md) — user wants no trailing summaries\n"
            "- [no mocks in tests](testing.md) — integration tests hit real DB\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="datus.utils.memory_loader"):
            load_memory_context(str(tmp_path), "chat")

        structural_warnings = [
            r for r in caplog.records if "longer than 50 lines" in r.message or "over 200 chars" in r.message
        ]
        assert structural_warnings == []


class TestGetMemoryDir:
    """Tests for get_memory_dir()."""

    def test_chat_dir(self):
        assert get_memory_dir(".", "chat") == f"{MEMORY_BASE_DIR}/chat"

    def test_custom_agent_dir(self):
        result = get_memory_dir("/workspace", "my_agent")
        assert result == f"{MEMORY_BASE_DIR}/my_agent"

    def test_dir_is_relative(self):
        result = get_memory_dir("/any/path", "chat")
        assert not result.startswith("/")
