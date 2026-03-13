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

    # Check for non-terminal environments (CI, piped stdin, web servers) outside
    # the except block to avoid Python chaining termios.error as __context__
    # on any exception raised inside the `with` body.
    _has_terminal = True
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        _has_terminal = False

    if not _has_terminal:
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


class EscapeGuard:
    """Guard object for pausing/resuming ESC key listening.

    Returned by interrupt_on_escape context manager. Use paused() to
    temporarily suspend ESC listening during interactive prompts that
    use prompt_toolkit or other libraries managing their own terminal mode.

    Arrow keys send escape sequences starting with \\x1b (ESC), which would
    otherwise be intercepted by the listener and trigger a false interrupt.
    """

    def __init__(self, pause_event=None, paused_ack=None):
        self._pause_event = pause_event
        self._paused_ack = paused_ack

    @contextmanager
    def paused(self):
        """Temporarily pause ESC listening and restore terminal settings."""
        if self._pause_event is None:
            # No-op guard (non-Unix or non-terminal environment)
            yield
            return

        self._paused_ack.clear()
        self._pause_event.set()
        # Wait for listener thread to acknowledge the pause
        self._paused_ack.wait(timeout=1.0)
        try:
            yield
        finally:
            self._pause_event.clear()


@contextmanager
def interrupt_on_escape(interrupt_controller, key_callbacks=None):
    """Listen for ESC key and trigger interrupt_controller when detected.

    Starts a daemon thread that puts the terminal in non-canonical, no-echo
    mode and polls stdin for ESC (\\x1b). On detection, calls
    interrupt_controller.interrupt(). Ctrl+C (\\x03) sends SIGINT to
    preserve the original KeyboardInterrupt behavior.

    Additional key callbacks can be registered via key_callbacks dict,
    mapping raw byte values to callables. These callbacks are invoked
    without breaking the listener loop (e.g. Ctrl+O = b"\\x0f").

    Yields an EscapeGuard that can temporarily pause listening during
    interactive prompts (e.g. prompt_toolkit) to avoid intercepting
    arrow-key escape sequences.

    On non-Unix platforms or non-terminal environments this is a no-op.

    Args:
        interrupt_controller: InterruptController instance to signal on ESC
        key_callbacks: Optional dict mapping bytes to callables, invoked
            when the corresponding key is detected (without breaking the loop).
    """
    try:
        import termios
    except ImportError:
        yield EscapeGuard()
        return

    # Check for non-terminal environments outside the except block to avoid
    # Python chaining termios.error as __context__ on caller exceptions.
    _has_terminal = True
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (AttributeError, OSError, termios.error):
        _has_terminal = False

    if not _has_terminal:
        yield EscapeGuard()
        return

    stop_event = threading.Event()
    pause_event = threading.Event()
    paused_ack = threading.Event()

    def _listener():
        try:
            # Set terminal to raw-like mode: non-canonical, no echo
            new_settings = termios.tcgetattr(fd)
            # Turn off ICANON, ECHO and IEXTEN in lflag.
            # IEXTEN must be cleared so that Ctrl+O (VDISCARD) and other
            # extended characters are passed through to os.read() instead
            # of being intercepted by the terminal driver.
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO | termios.IEXTEN)
            # Set VMIN=0, VTIME=0 for non-blocking reads
            new_settings[6][termios.VMIN] = 0
            new_settings[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)

            while not stop_event.is_set():
                # Check for pause request (interactive prompt is active)
                if pause_event.is_set():
                    # Restore terminal settings for prompt_toolkit
                    termios.tcsetattr(fd, termios.TCSANOW, old_settings)
                    paused_ack.set()
                    # Wait for resume or stop
                    while pause_event.is_set() and not stop_event.is_set():
                        stop_event.wait(0.05)
                    if stop_event.is_set():
                        break
                    # Flush leftover input and re-enter raw mode
                    try:
                        termios.tcflush(fd, termios.TCIFLUSH)
                    except termios.error:
                        pass
                    new_settings = termios.tcgetattr(fd)
                    new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO | termios.IEXTEN)
                    new_settings[6][termios.VMIN] = 0
                    new_settings[6][termios.VTIME] = 0
                    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
                    continue

                # Use select with timeout to avoid busy-waiting
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    try:
                        ch = os.read(fd, 1)
                    except OSError:
                        break
                    if ch == b"\x1b":  # ESC byte received
                        # Arrow keys and other special keys send escape sequences
                        # starting with \x1b (e.g. \x1b[A for Up). Wait briefly to
                        # check if more bytes follow. If they do, it's an escape
                        # sequence (not a standalone ESC press) — consume and ignore.
                        follow_ready, _, _ = select.select([fd], [], [], 0.05)
                        if follow_ready:
                            # More bytes available — this is an escape sequence, not ESC.
                            # Drain the remaining bytes of the sequence.
                            try:
                                os.read(fd, 16)
                            except OSError:
                                pass
                        elif not interrupt_controller.is_interrupted:
                            # No follow-up bytes and not already interrupted
                            # — genuine ESC key press
                            logger.info("ESC key detected, triggering interrupt")
                            interrupt_controller.interrupt()
                            # Do NOT break here: keep the listener alive so that
                            # key_callbacks (e.g. Ctrl+O for verbose toggle) remain
                            # functional while the interrupt propagates through the
                            # async execution stack.
                    elif ch == b"\x03":  # Ctrl+C
                        # Send SIGINT to preserve original behavior
                        import signal

                        os.kill(os.getpid(), signal.SIGINT)
                        break
                    elif key_callbacks and ch in key_callbacks:
                        try:
                            key_callbacks[ch]()
                        except Exception:
                            logger.exception("Key callback failed", extra={"key": ch.hex()})
        except Exception:
            # Silently ignore errors in the listener thread
            pass

    listener_thread = threading.Thread(target=_listener, daemon=True)
    listener_thread.start()

    try:
        yield EscapeGuard(pause_event, paused_ack)
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
