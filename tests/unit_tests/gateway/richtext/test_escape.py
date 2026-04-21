# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.gateway.richtext.escape."""

from datus.gateway.richtext.escape import slack_escape


class TestSlackEscape:
    def test_basic_escape(self):
        assert slack_escape("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_preserve_user_mention(self):
        assert slack_escape("Hello <@U123ABC>!") == "Hello <@U123ABC>!"

    def test_preserve_channel_mention(self):
        assert slack_escape("See <#C456DEF>") == "See <#C456DEF>"

    def test_preserve_special_mention(self):
        assert slack_escape("Hey <!here> and <!channel>") == "Hey <!here> and <!channel>"

    def test_preserve_url_token(self):
        result = slack_escape("Visit <https://example.com|Example>")
        assert result == "Visit <https://example.com|Example>"

    def test_preserve_http_url(self):
        result = slack_escape("Link: <http://example.com>")
        assert result == "Link: <http://example.com>"

    def test_preserve_mailto(self):
        result = slack_escape("Email <mailto:a@b.com|a@b.com>")
        assert result == "Email <mailto:a@b.com|a@b.com>"

    def test_mixed_content(self):
        text = "Hello <@U123>, check <https://example.com|site> for x < y & z > w"
        result = slack_escape(text)
        assert "<@U123>" in result
        assert "<https://example.com|site>" in result
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_no_tokens(self):
        assert slack_escape("plain text") == "plain text"

    def test_empty_string(self):
        assert slack_escape("") == ""

    def test_non_slack_angle_brackets_escaped(self):
        result = slack_escape("<not a slack token>")
        assert result == "&lt;not a slack token&gt;"

    def test_angle_brackets_with_newline_not_matched(self):
        result = slack_escape("<line1\nline2>")
        assert "&lt;" in result
        assert "&gt;" in result
