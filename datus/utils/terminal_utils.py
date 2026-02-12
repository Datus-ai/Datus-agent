# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Terminal utilities shared across layers (CLI, storage, etc.)."""

import sys
from contextlib import contextmanager


@contextmanager
def suppress_keyboard_input():
    """Suppress terminal control characters during streaming output.

    Disables special control characters (Ctrl+O/DISCARD, Ctrl+S/STOP,
    Ctrl+Q/START, Ctrl+V/LNEXT, Ctrl+R/REPRINT) that can freeze or
    disrupt terminal output.  ICANON, ECHO, and ISIG are left unchanged
    so that Rich Live, asyncio, and Ctrl+C all work normally.

    On exit, the original terminal settings are restored and any
    keystrokes buffered during streaming are flushed.

    On non-Unix platforms (Windows) or non-terminal environments
    (Streamlit, Jupyter, web servers) this is a no-op.
    """
    try:
        import termios
    except ImportError:
        # Non-Unix platform (e.g. Windows)
        yield
        return

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        # AttributeError: stdin replaced with object lacking fileno() (e.g. Streamlit, Jupyter)
        # OSError/termios.error: stdin is not a real terminal (e.g. piped, /dev/null, web server)
        yield
        return

    # Indices of control characters to disable.
    # Setting them to 0 (b'\x00') means "no character assigned".
    cc_to_disable = []
    for name in ("VDISCARD", "VSTOP", "VSTART", "VLNEXT", "VREPRINT"):
        idx = getattr(termios, name, None)
        if idx is not None:
            cc_to_disable.append(idx)

    try:
        new_settings = termios.tcgetattr(fd)
        for idx in cc_to_disable:
            new_settings[6][idx] = b"\x00"
        # Also disable IXON (software flow control) to prevent Ctrl+S/Q
        # from pausing/resuming output at the driver level.
        new_settings[0] &= ~termios.IXON
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_settings)
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except termios.error:
            pass
