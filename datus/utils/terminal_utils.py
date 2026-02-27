# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Terminal utilities shared across layers (CLI, storage, etc.)."""

import os
import select
import sys
import threading
from contextlib import contextmanager

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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


@contextmanager
def interrupt_on_escape(interrupt_controller):
    """Listen for ESC key and trigger interrupt_controller when detected.

    Starts a daemon thread that puts the terminal in non-canonical, no-echo
    mode and polls stdin for ESC (\\x1b). On detection, calls
    interrupt_controller.interrupt(). Ctrl+C (\\x03) sends SIGINT to
    preserve the original KeyboardInterrupt behavior.

    On non-Unix platforms or non-terminal environments this is a no-op.

    Args:
        interrupt_controller: InterruptController instance to signal on ESC
    """
    try:
        import termios
    except ImportError:
        yield
        return

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        yield
        return

    stop_event = threading.Event()

    def _listener():
        try:
            # Set terminal to raw-like mode: non-canonical, no echo
            new_settings = termios.tcgetattr(fd)
            # Turn off ICANON and ECHO in lflag
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
            # Set VMIN=0, VTIME=0 for non-blocking reads
            new_settings[6][termios.VMIN] = 0
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)

            while not stop_event.is_set():
                # Use select with timeout to avoid busy-waiting
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    try:
                        ch = os.read(fd, 1)
                    except OSError:
                        break
                    if ch == b"\x1b":  # ESC
                        logger.info("ESC key detected, triggering interrupt")
                        interrupt_controller.interrupt()
                        break
                    elif ch == b"\x03":  # Ctrl+C
                        # Send SIGINT to preserve original behavior
                        import signal

                        os.kill(os.getpid(), signal.SIGINT)
                        break
        except Exception:
            # Silently ignore errors in the listener thread
            pass

    listener_thread = threading.Thread(target=_listener, daemon=True)
    listener_thread.start()

    try:
        yield
    finally:
        stop_event.set()
        listener_thread.join(timeout=1.0)
        # Restore original terminal settings
        try:
            termios.tcsetattr(fd, termios.TCSANOW, old_settings)
        except termios.error:
            pass
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except termios.error:
            pass
