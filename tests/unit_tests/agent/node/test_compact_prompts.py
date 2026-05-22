# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for compact_prompts: j2 template rendering + continuation builder."""

from datus.agent.node.compact_prompts import build_continuation_message, render_major_compact_prompt


class TestRenderMajorCompactPrompt:
    def test_renders_all_10_sections(self):
        prompt = render_major_compact_prompt(
            node_role="chat",
            history_jsonl_path="/tmp/h.jsonl",
            archive_dir="/tmp/data",
        )
        # Each of the 10 sections must appear by its numbered header.
        for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            assert f"## {i}." in prompt, f"missing section {i}"

    def test_substitutes_node_role(self):
        prompt = render_major_compact_prompt(
            node_role="gen_report",
            history_jsonl_path="/x.jsonl",
            archive_dir="/x/data",
        )
        assert "`gen_report`" in prompt

    def test_substitutes_recovery_paths(self):
        prompt = render_major_compact_prompt(
            node_role="chat",
            history_jsonl_path="/tmp/h.jsonl",
            archive_dir="/tmp/data",
        )
        assert "/tmp/h.jsonl" in prompt
        assert "/tmp/data" in prompt

    def test_no_tools_constraint_present(self):
        prompt = render_major_compact_prompt(node_role="chat", history_jsonl_path="x", archive_dir="x")
        assert "TEXT ONLY" in prompt
        assert "tool calls will be rejected" in prompt.lower()

    def test_custom_instructions_appear_when_provided(self):
        prompt = render_major_compact_prompt(
            node_role="chat",
            history_jsonl_path="x",
            archive_dir="x",
            custom_instructions="Focus on SQL changes.",
        )
        assert "Additional instructions" in prompt
        assert "Focus on SQL changes." in prompt

    def test_custom_instructions_block_absent_by_default(self):
        prompt = render_major_compact_prompt(node_role="chat", history_jsonl_path="x", archive_dir="x")
        assert "Additional instructions" not in prompt


class TestBuildContinuationMessage:
    def test_embeds_summary_verbatim(self):
        msg = build_continuation_message("SUMMARY_BODY", "/h.jsonl", "/data/")
        assert "SUMMARY_BODY" in msg

    def test_announces_continuation(self):
        msg = build_continuation_message("s", "/h.jsonl", "/data/")
        # The opening line must signal "this is a resumed session" so the LLM
        # doesn't greet the user or recap the conversation.
        assert "continued from a previous conversation" in msg
        assert "Do not greet" in msg

    def test_includes_recovery_paths(self):
        msg = build_continuation_message("s", "/path/h.jsonl", "/path/data/")
        assert "/path/h.jsonl" in msg
        assert "/path/data/" in msg
        # The hint should reference ``read_file`` so the LLM knows how to load
        # original content; otherwise the recovery pointer is dead weight.
        assert "read_file" in msg

    def test_strips_surrounding_whitespace_in_summary(self):
        msg = build_continuation_message("   trimmed   ", "h", "d")
        # ``summary.strip()`` is applied; the raw whitespace shouldn't reach
        # the model as part of the visible body.
        assert "   trimmed   " not in msg
        assert "trimmed" in msg
