# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for compact_prompts: j2 template rendering + continuation builder."""

from datus.agent.node.compact_prompts import build_continuation_message, render_major_compact_prompt


class TestRenderMajorCompactPrompt:
    def test_renders_sections_1_through_8(self):
        prompt = render_major_compact_prompt(node_role="chat")
        # All eight summary sections must appear by their numbered header.
        for i in (1, 2, 3, 4, 5, 6, 7, 8):
            assert f"## {i}." in prompt, f"missing section {i}"

    def test_omits_recovery_pointer_section(self):
        # Recovery pointers are appended by the host, not the LLM, so the
        # prompt must not invite the model to produce them. Otherwise the
        # model echoes the directory + enumerates archive files (~5k chars
        # of noise observed in production sessions).
        prompt = render_major_compact_prompt(node_role="chat")
        assert "## 9." not in prompt
        assert "Recovery pointers" not in prompt
        assert "archive_dir" not in prompt

    def test_explicit_constraint_against_recovery_pointers(self):
        prompt = render_major_compact_prompt(node_role="chat")
        assert "appended by the host" in prompt

    def test_substitutes_node_role(self):
        prompt = render_major_compact_prompt(node_role="gen_report")
        assert "`gen_report`" in prompt

    def test_no_tools_constraint_present(self):
        prompt = render_major_compact_prompt(node_role="chat")
        assert "TEXT ONLY" in prompt
        assert "tool calls will be rejected" in prompt.lower()

    def test_custom_instructions_appear_when_provided(self):
        prompt = render_major_compact_prompt(
            node_role="chat",
            custom_instructions="Focus on SQL changes.",
        )
        assert "Additional instructions" in prompt
        assert "Focus on SQL changes." in prompt

    def test_custom_instructions_block_absent_by_default(self):
        prompt = render_major_compact_prompt(node_role="chat")
        assert "Additional instructions" not in prompt


class TestBuildContinuationMessage:
    def test_embeds_summary_verbatim(self):
        msg = build_continuation_message("SUMMARY_BODY", "/h.jsonl")
        assert "SUMMARY_BODY" in msg

    def test_appends_jsonl_pointer_with_read_file_hint(self):
        msg = build_continuation_message("body", "/path/h.jsonl")
        # The pointer must reference both the JSONL path and the ``read_file``
        # tool name; otherwise the next turn has no way to recover detail
        # dropped by the summary.
        assert "/path/h.jsonl" in msg
        assert "read_file" in msg

    def test_strips_surrounding_whitespace_in_summary(self):
        msg = build_continuation_message("   trimmed   ", "h")
        # ``summary.strip()`` is applied; the raw whitespace shouldn't reach
        # the model as part of the visible body.
        assert "   trimmed   " not in msg
        assert "trimmed" in msg

    def test_no_archive_dir_in_message(self):
        # The archive directory was previously surfaced as a second recovery
        # pointer; the LLM mirrored it back and enumerated every archived
        # file. The host-appended pointer is now JSONL-only — anything that
        # needs an archived blob must be cited by name in the summary.
        msg = build_continuation_message("body", "/h.jsonl")
        assert "Archived tool I/O" not in msg
        assert "archive_dir" not in msg

    def test_empty_jsonl_path_drops_pointer(self):
        # When the JSONL dump failed there is no path to point at, so the
        # helper must emit the bare summary instead of a broken
        # ``read_file('')`` line.
        msg = build_continuation_message("body only", "")
        assert msg == "body only"
