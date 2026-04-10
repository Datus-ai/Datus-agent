# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Markdown IR layer: Markdown -> IR -> platform-specific rich text."""

from datus.claw.richtext.chunker import chunk_text
from datus.claw.richtext.escape import slack_escape
from datus.claw.richtext.ir import MarkdownIR
from datus.claw.richtext.parser import markdown_to_ir
from datus.claw.richtext.render import render_ir

__all__ = ["MarkdownIR", "chunk_text", "markdown_to_ir", "render_ir", "slack_escape"]
