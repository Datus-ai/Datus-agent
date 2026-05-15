# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Multi-backend clipboard helper for the TUI selection feature.

Selection-release in :class:`DatusApp` writes the highlighted text to the
system clipboard automatically. Two backends are tried in order:

1. ``pyperclip`` — already a project dependency. Works on macOS via
   ``pbcopy``, on Linux when ``xclip`` / ``xsel`` are installed, and on
   Windows via PowerShell. Fails silently on remote SSH sessions or
   container environments without a display server.
2. **OSC 52 escape sequence** — universal fallback that asks the
   *terminal emulator* (iTerm2, Kitty, Alacritty, WezTerm, modern xterm)
   to put the text on the clipboard. Works over SSH because the escape
   travels through the same pty that carries normal output.

Inside tmux the OSC 52 byte stream is wrapped with the ``DCS pass-through``
sequence (``\\x1bPtmux;<escaped>\\x1b\\\\``) so tmux relays the payload to
the outer terminal instead of swallowing it. The wrapping is applied when
``$TMUX`` is set; users running ``set -g set-clipboard on`` in their tmux
config get the same behaviour either way.
"""

from __future__ import annotations

import base64
import os
import sys
from typing import IO

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Some terminals limit the OSC 52 payload to ~100 KiB. Selections larger
# than this almost certainly indicate the user dragged across the entire
# scrollback by accident, so silently dropping the OSC 52 path is a kinder
# UX than blowing up tmux's pass-through buffer.
_OSC52_MAX_BYTES: int = 100 * 1024


def copy_to_clipboard(text: str, *, stream: IO[str] | None = None) -> bool:
    """Write ``text`` to the system clipboard. Returns ``True`` on success.

    ``stream`` (defaulting to ``sys.stdout``) is the destination for the
    OSC 52 escape when the pyperclip backend is unavailable; tests inject a
    :class:`io.StringIO` to capture the bytes.
    """
    if not text:
        return False
    if _pyperclip_copy(text):
        return True
    return _osc52_copy(text, stream=stream)


def _pyperclip_copy(text: str) -> bool:
    """Try the pyperclip backend; return ``True`` only on a confirmed write."""
    try:
        import pyperclip
    except Exception as exc:  # pragma: no cover - depends on the env
        logger.debug("pyperclip import failed: %s", exc)
        return False
    try:
        pyperclip.copy(text)
    except Exception as exc:  # pragma: no cover - depends on the env
        # pyperclip raises ``PyperclipException`` on Linux with no
        # clipboard tool installed, or on macOS in restricted sandboxes.
        # Falling through to OSC 52 still gives the user a working copy
        # in those situations.
        logger.debug("pyperclip.copy raised: %s", exc)
        return False
    return True


def _osc52_copy(text: str, *, stream: IO[str] | None = None) -> bool:
    """Emit the OSC 52 ``set clipboard`` escape on ``stream``.

    Returns ``False`` only when the payload exceeds
    :data:`_OSC52_MAX_BYTES` (most terminals refuse such payloads anyway).
    A failed write to the stream is treated as a non-fatal best-effort —
    the user's selection still exists in software, they just don't get the
    system clipboard updated.
    """
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    if len(encoded) > _OSC52_MAX_BYTES:
        logger.debug("OSC 52 payload exceeds %d bytes; skipping clipboard write", _OSC52_MAX_BYTES)
        return False

    payload = f"\x1b]52;c;{encoded}\x07"
    if os.environ.get("TMUX"):
        # tmux DCS pass-through: wrap with ``\x1bPtmux;<escaped>\x1b\\``
        # and double every ESC inside the payload, so tmux unwraps it
        # cleanly when forwarding to the outer terminal.
        escaped = payload.replace("\x1b", "\x1b\x1b")
        payload = f"\x1bPtmux;{escaped}\x1b\\"

    out = stream if stream is not None else sys.stdout
    try:
        out.write(payload)
        flush = getattr(out, "flush", None)
        if callable(flush):
            flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("OSC 52 write failed: %s", exc)
        return False
    return True


__all__ = ["copy_to_clipboard"]
