# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Markdown -> MarkdownIR parser using markdown-it-py."""

from __future__ import annotations

from markdown_it import MarkdownIt

from datus.gateway.richtext.ir import LinkSpan, MarkdownIR, StyleSpan, StyleType

# Token type -> StyleType mapping for open/close pairs
_STYLE_MAP: dict[str, StyleType] = {
    "strong": StyleType.BOLD,
    "em": StyleType.ITALIC,
    "s": StyleType.STRIKETHROUGH,
}

# Heading tag -> level
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def markdown_to_ir(
    markdown: str,
    *,
    heading_style: str = "heading",
    table_mode: str = "off",
) -> MarkdownIR:
    """Parse a Markdown string into a MarkdownIR.

    Parameters
    ----------
    heading_style:
        ``"heading"`` — emit ``StyleType.HEADING`` (default, backward compat).
        ``"bold"`` — emit ``StyleType.BOLD`` instead.
        ``"none"`` — no style span, just plain text.
    table_mode:
        ``"off"`` — pipe-aligned text with ``StyleType.TABLE`` span (default).
        ``"bullets"`` — each data row as a bullet list grouped by first column.
        ``"code"`` — pipe-aligned text with ``StyleType.CODE_BLOCK`` span.
    """
    if not markdown:
        return MarkdownIR(text="")

    md = MarkdownIt("commonmark").enable("strikethrough").enable("table")
    tokens = md.parse(markdown)

    ctx = _ParseContext(heading_style=heading_style, table_mode=table_mode)
    _walk_tokens(tokens, ctx)

    return MarkdownIR(text=ctx.text, styles=ctx.styles, links=ctx.links)


class _ParseContext:
    """Mutable state accumulator for the token walk."""

    __slots__ = (
        "text",
        "styles",
        "links",
        "_style_stack",
        "_link_stack",
        "_list_depth",
        "_blockquote_start",
        "_table_row",
        "_table_rows",
        "_table_is_header",
        "_in_table_cell",
        "_cell_buffer",
        "_heading_style",
        "_table_mode",
        "_ordered_list_indices",
        "_in_ordered_list",
        "_suppress_para_newline",
    )

    def __init__(self, heading_style: str = "heading", table_mode: str = "off") -> None:
        self.text: str = ""
        self.styles: list[StyleSpan] = []
        self.links: list[LinkSpan] = []
        self._style_stack: list[tuple[StyleType, int, dict | None]] = []
        self._link_stack: list[tuple[int, str]] = []
        self._list_depth: int = 0
        self._blockquote_start: int | None = None
        self._table_row: list[str] = []
        self._table_rows: list[list[str]] = []
        self._table_is_header: bool = False
        self._in_table_cell: bool = False
        self._cell_buffer: str = ""
        self._heading_style: str = heading_style
        self._table_mode: str = table_mode
        self._ordered_list_indices: list[int] = []
        self._in_ordered_list: bool = False
        self._suppress_para_newline: bool = False

    @property
    def pos(self) -> int:
        return len(self.text)

    def append(self, s: str) -> None:
        self.text += s

    def push_style(self, style: StyleType, meta: dict | None = None) -> None:
        self._style_stack.append((style, self.pos, meta))

    def pop_style(self, style: StyleType) -> None:
        # Pop the most recent matching style
        for i in range(len(self._style_stack) - 1, -1, -1):
            if self._style_stack[i][0] == style:
                st, start, meta = self._style_stack.pop(i)
                if self.pos > start:
                    self.styles.append(StyleSpan(start=start, end=self.pos, style=st, meta=meta))
                return

    def push_link(self, href: str) -> None:
        self._link_stack.append((self.pos, href))

    def pop_link(self) -> None:
        if self._link_stack:
            start, href = self._link_stack.pop()
            if self.pos > start:
                self.links.append(LinkSpan(start=start, end=self.pos, href=href))


def _walk_tokens(tokens: list, ctx: _ParseContext) -> None:
    """Walk a flat list of markdown-it tokens and populate *ctx*."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i = _process_token(tok, tokens, i, ctx)


def _process_token(tok, tokens: list, idx: int, ctx: _ParseContext) -> int:
    """Process a single token. Returns the next index to process."""
    tag = getattr(tok, "tag", "")
    tok_type: str = tok.type
    children = getattr(tok, "children", None)

    # --- Inline container ---
    if tok_type == "inline" and children:
        if ctx._in_table_cell:
            ctx._cell_buffer += _extract_inline_text(children)
        else:
            _walk_inline(children, ctx)
        return idx + 1

    # --- Fence / code block ---
    if tok_type == "fence":
        _handle_fence(tok, ctx)
        return idx + 1

    # --- Code block (indented) ---
    if tok_type == "code_block":
        _handle_code_block(tok, ctx)
        return idx + 1

    # --- Headings ---
    if tok_type == "heading_open" and tag in _HEADING_TAGS:
        level = int(tag[1])
        if ctx._heading_style == "bold":
            ctx.push_style(StyleType.BOLD)
        elif ctx._heading_style == "heading":
            ctx.push_style(StyleType.HEADING, meta={"level": level})
        # "none" — no style push
        return idx + 1

    if tok_type == "heading_close":
        if ctx._heading_style == "bold":
            ctx.pop_style(StyleType.BOLD)
        elif ctx._heading_style == "heading":
            ctx.pop_style(StyleType.HEADING)
        ctx.append("\n")
        return idx + 1

    # --- Blockquote ---
    if tok_type == "blockquote_open":
        ctx._blockquote_start = ctx.pos
        return idx + 1

    if tok_type == "blockquote_close":
        if ctx._blockquote_start is not None:
            start = ctx._blockquote_start
            ctx._blockquote_start = None
            if ctx.pos > start:
                ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=StyleType.BLOCKQUOTE))
        return idx + 1

    # --- Lists ---
    if tok_type == "bullet_list_open":
        ctx._list_depth += 1
        return idx + 1

    if tok_type == "bullet_list_close":
        ctx._list_depth -= 1
        return idx + 1

    if tok_type == "ordered_list_open":
        ctx._list_depth += 1
        start_val = 1
        if hasattr(tok, "attrGet") and tok.attrGet("start") is not None:
            start_val = int(tok.attrGet("start"))
        elif isinstance(getattr(tok, "attrs", None), dict) and "start" in tok.attrs:
            start_val = int(tok.attrs["start"])
        ctx._ordered_list_indices.append(start_val)
        ctx._in_ordered_list = True
        return idx + 1

    if tok_type == "ordered_list_close":
        ctx._list_depth -= 1
        if ctx._ordered_list_indices:
            ctx._ordered_list_indices.pop()
        ctx._in_ordered_list = bool(ctx._ordered_list_indices)
        return idx + 1

    if tok_type == "list_item_open":
        indent = "  " * max(0, ctx._list_depth - 1)
        if ctx._in_ordered_list and ctx._ordered_list_indices:
            idx_val = ctx._ordered_list_indices[-1]
            ctx.append(f"{indent}{idx_val}. ")
            ctx._ordered_list_indices[-1] = idx_val + 1
        else:
            ctx.append(f"{indent}\u2022 ")
        ctx._suppress_para_newline = True
        return idx + 1

    if tok_type == "list_item_close":
        return idx + 1

    # --- Paragraph ---
    if tok_type == "paragraph_open":
        if ctx._suppress_para_newline:
            ctx._suppress_para_newline = False
        elif ctx.text and not ctx.text.endswith("\n"):
            ctx.append("\n")
        return idx + 1

    if tok_type == "paragraph_close":
        if ctx.text and not ctx.text.endswith("\n"):
            ctx.append("\n")
        return idx + 1

    # --- Table ---
    if tok_type == "table_open":
        ctx._table_rows = []
        return idx + 1

    if tok_type == "table_close":
        _flush_table(ctx)
        return idx + 1

    if tok_type == "thead_open":
        ctx._table_is_header = True
        return idx + 1

    if tok_type == "thead_close":
        ctx._table_is_header = False
        return idx + 1

    if tok_type in ("tbody_open", "tbody_close"):
        return idx + 1

    if tok_type == "tr_open":
        ctx._table_row = []
        return idx + 1

    if tok_type == "tr_close":
        ctx._table_rows.append(list(ctx._table_row))
        ctx._table_row = []
        return idx + 1

    if tok_type in ("th_open", "td_open"):
        ctx._in_table_cell = True
        ctx._cell_buffer = ""
        return idx + 1

    if tok_type in ("th_close", "td_close"):
        ctx._table_row.append(ctx._cell_buffer.strip())
        ctx._in_table_cell = False
        ctx._cell_buffer = ""
        return idx + 1

    # --- Horizontal rule ---
    if tok_type == "hr":
        if ctx.text and not ctx.text.endswith("\n"):
            ctx.append("\n")
        ctx.append("---\n")
        return idx + 1

    return idx + 1


def _walk_inline(tokens: list, ctx: _ParseContext) -> None:
    """Walk inline-level tokens."""
    for tok in tokens:
        tok_type: str = tok.type

        # Plain text content
        if tok_type == "text":
            ctx.append(tok.content)
            continue

        # Soft/hard breaks
        if tok_type in ("softbreak", "hardbreak"):
            ctx.append("\n")
            continue

        # Inline code
        if tok_type == "code_inline":
            start = ctx.pos
            ctx.append(tok.content)
            if ctx.pos > start:
                ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=StyleType.CODE))
            continue

        # Style open/close (bold, italic, strikethrough)
        style = _STYLE_MAP.get(tok.tag)
        if style:
            if tok_type.endswith("_open"):
                ctx.push_style(style)
            elif tok_type.endswith("_close"):
                ctx.pop_style(style)
            continue

        # Links
        if tok_type == "link_open":
            href = ""
            for attr_name, attr_val in (tok.attrs or {}).items():
                if attr_name == "href":
                    href = attr_val
                    break
            ctx.push_link(href)
            continue

        if tok_type == "link_close":
            ctx.pop_link()
            continue

        # Images — treat alt text as plain text
        if tok_type == "image":
            alt = tok.content or getattr(tok, "alt", "") or ""
            if alt:
                ctx.append(alt)
            continue

        # HTML inline — skip
        if tok_type == "html_inline":
            continue


def _extract_inline_text(tokens: list) -> str:
    """Extract plain text content from inline tokens (for table cells)."""
    parts: list[str] = []
    for tok in tokens:
        if tok.type == "text":
            parts.append(tok.content)
        elif tok.type == "code_inline":
            parts.append(tok.content)
        elif tok.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return "".join(parts)


def _flush_table(ctx: _ParseContext) -> None:
    """Convert accumulated table rows based on ``ctx._table_mode``."""
    if not ctx._table_rows:
        return
    if ctx.text and not ctx.text.endswith("\n"):
        ctx.append("\n")

    if ctx._table_mode == "bullets":
        _flush_table_bullets(ctx)
    elif ctx._table_mode == "code":
        _flush_table_pipe(ctx, style=StyleType.CODE_BLOCK)
    else:
        _flush_table_pipe(ctx, style=StyleType.TABLE)


def _flush_table_pipe(ctx: _ParseContext, style: StyleType = StyleType.TABLE) -> None:
    """Emit pipe-aligned text with the given style span."""
    col_count = max(len(row) for row in ctx._table_rows) if ctx._table_rows else 0
    col_widths = [0] * col_count
    for row in ctx._table_rows:
        for j, cell in enumerate(row):
            if j < col_count:
                col_widths[j] = max(col_widths[j], len(cell))

    start = ctx.pos
    for i, row in enumerate(ctx._table_rows):
        padded = [cell.ljust(col_widths[j]) if j < len(col_widths) else cell for j, cell in enumerate(row)]
        line = "| " + " | ".join(padded) + " |"
        ctx.append(line + "\n")
        if i == 0:
            sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
            ctx.append(sep + "\n")

    if ctx.pos > start:
        ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=style))

    ctx._table_rows = []


def _flush_table_bullets(ctx: _ParseContext) -> None:
    """Emit each data row as a bullet list grouped by the first column value."""
    if not ctx._table_rows:
        return

    headers = ctx._table_rows[0] if ctx._table_rows else []
    data_rows = ctx._table_rows[1:]

    for row_idx, row in enumerate(data_rows):
        if row_idx > 0:
            ctx.append("\n")
        # First column value as bold label
        first_val = row[0] if row else ""
        start = ctx.pos
        ctx.append(first_val)
        if ctx.pos > start:
            ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=StyleType.BOLD))
        ctx.append("\n")
        # Remaining columns as bullet items
        for col_idx in range(1, max(len(headers), len(row))):
            header = headers[col_idx] if col_idx < len(headers) else ""
            value = row[col_idx] if col_idx < len(row) else ""
            ctx.append(f"\u2022 {header}: {value}\n")

    ctx._table_rows = []


def _handle_fence(tok, ctx: _ParseContext) -> None:
    """Handle fenced code blocks (```)."""
    if ctx.text and not ctx.text.endswith("\n"):
        ctx.append("\n")
    content = tok.content
    if content.endswith("\n"):
        content = content[:-1]
    start = ctx.pos
    ctx.append(content)
    lang = (tok.info or "").strip()
    meta = {"language": lang} if lang else None
    if ctx.pos > start:
        ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=StyleType.CODE_BLOCK, meta=meta))
    ctx.append("\n")


def _handle_code_block(tok, ctx: _ParseContext) -> None:
    """Handle indented code blocks."""
    if ctx.text and not ctx.text.endswith("\n"):
        ctx.append("\n")
    content = tok.content
    if content.endswith("\n"):
        content = content[:-1]
    start = ctx.pos
    ctx.append(content)
    if ctx.pos > start:
        ctx.styles.append(StyleSpan(start=start, end=ctx.pos, style=StyleType.CODE_BLOCK))
    ctx.append("\n")
