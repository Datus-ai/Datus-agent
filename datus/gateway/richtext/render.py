# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""IR -> platform-specific rich text renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from datus.gateway.richtext.ir import MarkdownIR, StyleType


@dataclass
class StyleMarker:
    """Opening and closing marker strings for a style."""

    open: str
    close: str


@dataclass
class RenderOptions:
    """Configuration for rendering IR to a specific platform format."""

    style_markers: dict[StyleType, StyleMarker] = field(default_factory=dict)
    escape_fn: Optional[Callable[[str], str]] = None
    link_builder: Optional[Callable[[str, str], str]] = None


def render_ir(ir: MarkdownIR, options: RenderOptions) -> str:
    """Render a MarkdownIR to a platform-specific string.

    Algorithm:
    1. Collect all span boundaries (open/close markers) with their positions.
    2. Sort by position; at the same position, closes come before opens.
    3. Walk through the plain text, inserting markers at boundary positions.
    """
    if not ir.text:
        return ""

    # Collect boundary events: (position, priority, marker_string)
    # priority: 0 = close, 1 = open (close before open at same position)
    events: list[tuple[int, int, str]] = []

    for span in ir.styles:
        marker = options.style_markers.get(span.style)
        if not marker:
            continue
        events.append((span.start, 1, marker.open))
        events.append((span.end, 0, marker.close))

    if options.link_builder:
        for link in ir.links:
            # Link markers are handled specially: we replace the text segment
            # with the link_builder output. We use sentinel markers.
            pass

    # Sort: by position, then close before open
    events.sort(key=lambda e: (e[0], e[1]))

    # Handle links via link_builder
    # We need to identify link regions and replace them.
    # Strategy: build the text with style markers first, then handle links.
    # But links may overlap with styles... simpler approach: handle links
    # by wrapping the already-rendered text segment.

    # First pass: build text with style markers (no links)
    result_parts: list[str] = []
    prev_pos = 0

    for pos, _priority, marker in events:
        if pos > prev_pos:
            segment = ir.text[prev_pos:pos]
            if options.escape_fn:
                segment = options.escape_fn(segment)
            result_parts.append(segment)
        elif pos == prev_pos:
            pass  # No text between markers
        result_parts.append(marker)
        prev_pos = pos

    # Append remaining text
    if prev_pos < len(ir.text):
        segment = ir.text[prev_pos:]
        if options.escape_fn:
            segment = options.escape_fn(segment)
        result_parts.append(segment)

    rendered = "".join(result_parts)

    # If there are links and a link_builder, apply link replacements.
    # We process links by finding the rendered text for each link span
    # and replacing it with the link_builder output.
    if options.link_builder and ir.links:
        rendered = _apply_links(ir, options, events, rendered)

    return rendered


def _apply_links(ir: MarkdownIR, options: RenderOptions, events: list, rendered: str) -> str:
    """Replace link text regions with link_builder output.

    We need to map plain-text offsets to rendered-text offsets accounting
    for inserted markers.
    """
    # Build a mapping from plain-text position -> rendered position offset
    # We track the cumulative marker length inserted before each position.
    marker_offsets: list[tuple[int, int]] = []  # (plain_pos, cumulative_extra)
    cumulative = 0
    for pos, _priority, marker in events:
        cumulative += len(marker)
        marker_offsets.append((pos, cumulative))

    def plain_to_rendered(plain_pos: int) -> int:
        """Map a plain-text offset to a rendered-text offset."""
        extra = 0
        for mpos, cum in marker_offsets:
            if mpos <= plain_pos:
                extra = cum
            else:
                break
        return plain_pos + extra

    # Apply link replacements in reverse order to preserve positions
    sorted_links = sorted(ir.links, key=lambda lk: lk.start, reverse=True)
    for link in sorted_links:
        r_start = plain_to_rendered(link.start)
        r_end = plain_to_rendered(link.end)
        link_text = rendered[r_start:r_end]
        replacement = options.link_builder(link_text, link.href)
        rendered = rendered[:r_start] + replacement + rendered[r_end:]

    return rendered
