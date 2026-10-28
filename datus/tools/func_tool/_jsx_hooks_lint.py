# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Static detection of the "early return before hooks" Rules-of-Hooks violation.

``validate_render`` used to check only paths (entry point, import targets,
``useQuerySql`` slugs), so a render tree that parses fine but throws the moment
React mounts it still passed. The single most common such failure — by a wide
margin on smaller models — is a guard clause placed above the hook calls::

    export default function App() {
        const { data, loading } = useQuerySql('queries/sales');
        if (loading) return <Spinner />;          // ← early return
        const totals = useMemo(() => sum(data), [data]);   // ← never reached
                                                          //   on the 1st render
        ...
    }

React throws "Rendered fewer hooks than expected" on the second render and the
whole artifact goes blank. Catching it here turns a user-visible blank iframe
into a tool-result the subagent can fix before it ever delivers.

Scope is deliberately narrow: *early return before a hook call, in the same
function*. Conditional hooks (``if (x) useMemo(...)``) and the missing/incorrect
import families are NOT checked here.

Known misses, all deliberate — every one of them trades recall for the
guarantee that correct code is never rejected:

* ``if (x) foo(); else return null;`` — a ``return`` is only anchored when the
  previous significant character is one of ``{};):``, which is what stops JSX
  text from being read as code. ``else`` does not qualify.
* A guard with neither a semicolon nor braces (``if (!x) return null`` on its
  own line) leaves the return statement open to the end of the function, so
  later hooks are not reported. Closing it at the newline instead would
  misjudge multi-line unparenthesised JSX returns, which is the worse trade.
* Hooks inside the returned expression itself are legal and skipped; see
  ``_Frame.return_open``.

Design constraints
------------------

A false positive is far worse than a miss: it blocks a correct artifact and
sends the subagent editing code that was already fine. So the scanner is
conservative everywhere it is unsure, and bails out entirely (returning no
issues) on anything it cannot confidently tokenize.

The core of the analysis is *function-frame* tracking rather than brace depth.
A ``return`` belongs to the nearest enclosing **function body**, not the nearest
enclosing brace, which is what makes both of these correct::

    const f = () => { if (xx) return; };   // f's return, not the component's
    useThing();                            // → NOT a violation

    useEffect(() => { if (!x) return; }, [x]);   // the arrow's return
    const y = useMemo(...);                      // → NOT a violation

    if (loading) { return null; }   // the *component's* return (a block `{`
    const y = useMemo(...);         //  does not open a new function frame)
                                    // → violation
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional

# Anything larger is almost certainly not hand-authored render code; skip it
# rather than spend time (and risk) scanning a bundled blob.
_MAX_SOURCE_CHARS = 512 * 1024

_IDENT_START = frozenset(string.ascii_letters + "_$")
_IDENT_PART = frozenset(string.ascii_letters + string.digits + "_$")

# `useFoo` / `useFOO` — the React convention. `use` alone, `used`, `useful`
# etc. deliberately do not match.
_HOOK_NAME_RE = re.compile(r"^use[A-Z][A-Za-z0-9_$]*$")

# `{` preceded by `)` opens a function body — unless the `(` belonged to one of
# these, in which case it is a plain block sharing the enclosing function frame.
_BLOCK_PAREN_HEADS = frozenset({"if", "for", "while", "switch", "catch", "with", "await"})

# A `return` is only treated as a statement when the previous significant
# character is one of these. This is what keeps JSX *text* from being mistaken
# for code: in `<div>return</div>` the previous character is `>`, which is not
# in the set. `)` covers the dominant `if (x) return ...` guard clause, `:`
# covers `case 'a': return ...`.
_RETURN_PREV_CHARS = frozenset("{};):")

# Characters after which a `/` starts a regex literal rather than a division.
# `{`, `}` and `>` are intentionally excluded: they precede the `/` of a
# self-closing JSX tag (`<Foo bar={x} />`), and treating that as a regex would
# swallow the rest of the line — including braces, which would corrupt the
# frame stack.
_REGEX_PREV_CHARS = frozenset("(,=:[!&|?;+-*%~^")
_REGEX_PREV_TOKENS = frozenset(
    {
        "return",
        "typeof",
        "case",
        "in",
        "of",
        "new",
        "delete",
        "void",
        "instanceof",
        "yield",
        "await",
        "do",
        "else",
    }
)


@dataclass(frozen=True)
class HookOrderIssue:
    """One hook call that is unreachable on some render because of a return."""

    hook_name: str
    hook_line: int
    return_line: int


@dataclass
class _Frame:
    """One function body. ``return_line``/``return_pos`` are the first found."""

    return_line: int = 0
    return_pos: int = -1
    # True between the anchoring ``return`` and the end of that statement.
    # Hooks *inside* the returned expression run unconditionally on the paths
    # that reach the return, so they are not violations — see ``_scan``.
    return_open: bool = False
    # ``len(braces)`` when the return was anchored, used to find its end.
    return_depth: int = 0


@dataclass
class _Brace:
    """One open ``{``. ``opens_frame`` braces also pushed a :class:`_Frame`."""

    opens_frame: bool
    restore_template: bool = False


class _Bail(Exception):
    """Raised when the source cannot be tokenized with confidence."""


def find_hook_order_issues(source: str) -> List[HookOrderIssue]:
    """Return every hook call that sits after an early return in its function.

    Returns an empty list — never a partial result — when the source cannot be
    scanned confidently (unbalanced braces, unterminated string/comment, or an
    implausibly large file).
    """
    if not source or len(source) > _MAX_SOURCE_CHARS:
        return []

    try:
        return _scan(source)
    except _Bail:
        return []


def format_hook_order_issues(rel: str, source: str) -> List[str]:
    """Render :func:`find_hook_order_issues` output as ``validate_render`` issues."""
    return [
        f"render/{rel}: {issue.hook_name}() on line {issue.hook_line} is called below the "
        f"`return` on line {issue.return_line}, so it is skipped whenever that branch is taken "
        'and React throws "Rendered fewer hooks than expected". Move every hook call above the '
        "first return, and gate the *rendered output* instead of the hook calls."
        for issue in find_hook_order_issues(source)
    ]


def _scan(source: str) -> List[HookOrderIssue]:  # noqa: C901 — one tokenizer loop
    issues: List[HookOrderIssue] = []
    frames: List[_Frame] = [_Frame()]
    braces: List[_Brace] = []
    # Head token of each currently-open `(`, so a `)` can tell `if (...)` from
    # `foo(...)` when the following `{` is classified.
    paren_heads: List[str] = []
    last_paren_head = ""

    n = len(source)
    i = 0
    line = 1
    in_template = False
    prev_char = ""  # last significant character
    prev_token = ""  # last significant identifier / `=>`

    while i < n:
        ch = source[i]

        if ch == "\n":
            line += 1
            i += 1
            continue

        if ch in " \t\r\f\v":
            i += 1
            continue

        # ---- template literal bodies (outside `${ }`)
        # Must precede the comment branches: a template is raw text, so the
        # `//` in an interpolated URL (`` `https://api/${id}` ``) is not a
        # comment. Reading it as one swallowed the closing backtick and bailed
        # the whole file, silently disabling the check for it.
        if in_template:
            if ch == "`":
                in_template = False
                prev_char = "`"
                prev_token = ""
                i += 1
                continue
            if ch == "\\":
                i += 2
                continue
            if ch == "$" and i + 1 < n and source[i + 1] == "{":
                braces.append(_Brace(opens_frame=False, restore_template=True))
                in_template = False
                prev_char = "{"
                prev_token = ""
                i += 2
                continue
            i += 1
            continue

        # ---- comments
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i = source.find("\n", i)
            if i == -1:
                break
            continue

        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                raise _Bail("unterminated block comment")
            line += source.count("\n", i, end)
            i = end + 2
            continue

        # ---- strings
        if ch in "'\"":
            end = _skip_quoted(source, i, ch)
            if end is None:
                # Not a string after all — an apostrophe in JSX text
                # (``<p>Don't forget</p>``) is the common case. A real string
                # literal cannot contain a raw newline, so skipping just this
                # character and carrying on is the correct reading.
                prev_char = ch
                prev_token = ""
                i += 1
                continue
            i = end
            prev_char = ch
            prev_token = ""
            continue

        if ch == "`":
            in_template = True
            i += 1
            continue

        # ---- regex literal vs division
        if ch == "/":
            if _starts_regex(prev_char, prev_token):
                end = _skip_regex(source, i)
                if end is not None:
                    i = end
                    prev_char = "/"
                    prev_token = ""
                    continue
            # Division (or a JSX `/>`): fall through as a plain operator.
            prev_char = "/"
            prev_token = ""
            i += 1
            continue

        # ---- identifiers / keywords
        if ch in _IDENT_START:
            start = i
            i += 1
            while i < n and source[i] in _IDENT_PART:
                i += 1
            token = source[start:i]

            if token == "return" and prev_char in _RETURN_PREV_CHARS:
                # `{ return: 1 }` — a property key, not a statement.
                if _next_significant_char(source, i) != ":":
                    frame = frames[-1]
                    if frame.return_pos < 0:
                        frame.return_pos = start
                        frame.return_line = line
                        frame.return_open = True
                        frame.return_depth = len(braces)
            elif _HOOK_NAME_RE.match(token) and _next_significant_char(source, i) == "(":
                frame = frames[-1]
                # ``return_open`` skips hooks written *into* the returned
                # expression — ``return <div ref={useRef(null)} />`` calls the
                # hook on every render that reaches the return, so it is legal.
                if frame.return_pos >= 0 and not frame.return_open:
                    issues.append(
                        HookOrderIssue(
                            hook_name=token,
                            hook_line=line,
                            return_line=frame.return_line,
                        )
                    )

            prev_char = token[-1]
            prev_token = token
            continue

        # ---- structural punctuation
        if ch == "(":
            paren_heads.append(prev_token)
            prev_char = "("
            prev_token = ""
            i += 1
            continue

        if ch == ")":
            last_paren_head = paren_heads.pop() if paren_heads else ""
            prev_char = ")"
            prev_token = ""
            i += 1
            continue

        if ch == "{":
            opens_frame = prev_token == "=>" or (prev_char == ")" and last_paren_head not in _BLOCK_PAREN_HEADS)
            braces.append(_Brace(opens_frame=opens_frame))
            if opens_frame:
                frames.append(_Frame())
            prev_char = "{"
            prev_token = ""
            i += 1
            continue

        if ch == "}":
            if not braces:
                raise _Bail("unbalanced closing brace")
            brace = braces.pop()
            if brace.opens_frame:
                if len(frames) == 1:
                    raise _Bail("frame stack underflow")
                frames.pop()
            if brace.restore_template:
                in_template = True
            # The block holding the return just closed, so the statement is
            # over even without a semicolon (``if (x) { return null }``).
            frame = frames[-1]
            if frame.return_open and len(braces) < frame.return_depth:
                frame.return_open = False
            prev_char = "}"
            prev_token = ""
            i += 1
            continue

        if ch == ";":
            frame = frames[-1]
            if frame.return_open and len(braces) == frame.return_depth:
                frame.return_open = False
            prev_char = ";"
            prev_token = ""
            i += 1
            continue

        if ch == "=" and i + 1 < n and source[i + 1] == ">":
            prev_char = ">"
            prev_token = "=>"
            i += 2
            continue

        prev_char = ch
        prev_token = ""
        i += 1

    if braces or in_template:
        raise _Bail("unbalanced braces or unterminated template literal")

    return issues


def _skip_quoted(source: str, i: int, quote: str) -> Optional[int]:
    """Return the index just past a ``'``/``"`` string, or ``None`` if it isn't one."""
    n = len(source)
    j = i + 1
    while j < n:
        c = source[j]
        if c == "\\":
            j += 2
            continue
        if c == quote:
            return j + 1
        if c == "\n":
            return None
        j += 1
    return None


def _starts_regex(prev_char: str, prev_token: str) -> bool:
    if prev_token:
        return prev_token in _REGEX_PREV_TOKENS
    return prev_char == "" or prev_char in _REGEX_PREV_CHARS


def _skip_regex(source: str, i: int) -> Optional[int]:
    """Return the index just past a regex literal, or ``None`` if it isn't one.

    Regex literals cannot span lines, so hitting a newline first means the `/`
    was really a division (or a JSX `/>`), and the caller falls back to that.
    """
    n = len(source)
    j = i + 1
    in_class = False
    while j < n:
        c = source[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            return None
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            j += 1
            # Trailing flags.
            while j < n and source[j] in _IDENT_PART:
                j += 1
            return j
        j += 1
    return None


def _next_significant_char(source: str, i: int) -> str:
    """Peek at the next character, skipping whitespace and comments."""
    n = len(source)
    while i < n:
        c = source[i]
        if c in " \t\r\n\f\v":
            i += 1
            continue
        if c == "/" and i + 1 < n:
            if source[i + 1] == "/":
                nl = source.find("\n", i)
                if nl == -1:
                    return ""
                i = nl + 1
                continue
            if source[i + 1] == "*":
                end = source.find("*/", i + 2)
                if end == -1:
                    return ""
                i = end + 2
                continue
        return c
    return ""
