import shutil
import unicodedata
from typing import Any, Dict, List, Optional

from prompt_toolkit.styles import Style
from rich.console import Console

from datus.cli.cli_styles import (
    CLR_CURRENT,
    CLR_CURSOR,
    PROMPT_ONLY_STYLE,
    SYM_ARROW,
    print_error,
    print_warning,
)
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_FREE_TEXT_SENTINEL = "__free_text__"

BACK_SENTINEL = "__back__"


def prompt_with_back(label: str, default: str = "", password: bool = False) -> str:
    """Prompt with ESC to go back. Uses prompt_toolkit for key handling.

    Returns :data:`BACK_SENTINEL` if ESC pressed, otherwise the entered value.
    """

    def _inner():
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("escape")
        def _esc(event):
            event.app.exit(result=BACK_SENTINEL)

        session = PromptSession(key_bindings=kb)
        suffix = f" ({default})" if default else ""
        result = session.prompt(f"{label}{suffix}: ", is_password=password)
        if result == BACK_SENTINEL:
            return BACK_SENTINEL
        return result.strip() if result.strip() else default

    try:
        return _run_prompt_in_terminal(_inner)
    except (KeyboardInterrupt, EOFError):
        return BACK_SENTINEL


def _run_prompt_in_terminal(fn: Any) -> Any:
    """Run a blocking prompt function, suspending the outer TUI if active.

    Unlike :func:`_run_sub_application` which wraps a prompt_toolkit
    ``Application``, this helper wraps an arbitrary callable (typically
    ``prompt_toolkit.prompt()``) that internally creates its own
    Application.  It uses the same ``in_terminal()`` mechanism to
    release stdin from the outer TUI while the callable runs.
    """
    import asyncio

    from prompt_toolkit.application import get_app_or_none

    pt_app = get_app_or_none()
    pt_loop = getattr(pt_app, "loop", None) if pt_app is not None else None
    if pt_loop is None or not pt_loop.is_running():
        return fn()

    from prompt_toolkit.application.run_in_terminal import in_terminal

    async def _scheduled():
        async with in_terminal():
            return await asyncio.get_running_loop().run_in_executor(None, fn)

    future = asyncio.run_coroutine_threadsafe(_scheduled(), pt_loop)
    return future.result()


def _run_sub_application(app: Any) -> Any:
    """Run a prompt_toolkit Application, suspending the outer TUI if active.

    When the Datus REPL is running in TUI mode, a persistent
    :class:`~prompt_toolkit.application.Application` owns stdin on the
    main thread.  Spawning a nested Application from the worker thread
    without releasing stdin causes freezes and display corruption.

    This helper detects the outer Application via
    :func:`get_app_or_none`, schedules the nested run inside
    ``in_terminal()`` on the main loop (which temporarily detaches the
    outer app's input reader), and blocks the worker until it finishes.

    In non-TUI mode (no outer Application running) the nested app is
    executed directly with ``app.run()``.
    """
    import asyncio

    from prompt_toolkit.application import get_app_or_none

    pt_app = get_app_or_none()
    pt_loop = getattr(pt_app, "loop", None) if pt_app is not None else None
    if pt_loop is None or not pt_loop.is_running():
        return app.run()

    from prompt_toolkit.application.run_in_terminal import in_terminal

    async def _scheduled():
        async with in_terminal():
            return await app.run_async()

    future = asyncio.run_coroutine_threadsafe(_scheduled(), pt_loop)
    return future.result()


def select_choice(
    console: Console,
    choices: Dict[str, str],
    default: str = "",
    allow_free_text: bool = False,
    current: str = "",
) -> str:
    """Interactive choice selector with arrow-key navigation.

    Uses prompt_toolkit Application for proper terminal handling.
    Up/Down arrows to navigate, Enter to confirm, or press shortcut key directly.
    When the list exceeds the terminal height, the view scrolls with the cursor
    and PageUp/PageDown jumps a screenful at a time.
    When ``allow_free_text`` is True, a "Type custom answer..." entry is appended.
    Choosing it, or pressing ``/``, opens the standard multiline input prompt so
    paste works reliably.

    Args:
        console: Rich Console (used for fallback output on error)
        choices: Ordered dict of {key: display_text}
                 e.g. {"y": "Allow (once)", "a": "Always allow (session)", "n": "Deny"}
        default: Default choice key (pre-selected on start)
        allow_free_text: When True, append a free-text option and allow ``/`` shortcut.
        current: Key of the currently active value. Highlighted with ``CLR_CURRENT``
                 and suffixed with ``(current)``.

    Returns:
        Selected choice key string, or the user's free-text input.
    """
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        display_choices = dict(choices)
        if allow_free_text:
            display_choices[_FREE_TEXT_SENTINEL] = "Type custom answer..."

        keys = list(display_choices.keys())
        total = len(keys)
        selected = [keys.index(default) if default in keys else 0]

        # Scroll window: reserve 3 lines for scroll header + hint + safety margin.
        term_height = shutil.get_terminal_size((120, 40)).lines
        max_visible = max(3, term_height - 3)
        offset = [0]
        if selected[0] >= max_visible:
            offset[0] = min(max(0, selected[0] - max_visible // 2), max(0, total - max_visible))

        def _ensure_visible() -> None:
            if selected[0] < offset[0]:
                offset[0] = selected[0]
            elif selected[0] >= offset[0] + max_visible:
                offset[0] = selected[0] - max_visible + 1

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            selected[0] = (selected[0] - 1) % total
            if selected[0] == total - 1:
                offset[0] = max(0, total - max_visible)
            else:
                _ensure_visible()

        @kb.add("down")
        def _move_down(event):
            selected[0] = (selected[0] + 1) % total
            if selected[0] == 0:
                offset[0] = 0
            else:
                _ensure_visible()

        @kb.add("pageup")
        def _page_up(event):
            selected[0] = max(0, selected[0] - max_visible)
            _ensure_visible()

        @kb.add("pagedown")
        def _page_down(event):
            selected[0] = min(total - 1, selected[0] + max_visible)
            _ensure_visible()

        @kb.add("enter")
        def _confirm(event):
            sel = keys[selected[0]]
            event.app.exit(result=sel)

        @kb.add("c-c")
        def _cancel(event):
            event.app.exit(result=default)

        # Direct shortcut keys (press y/a/n to pick immediately)
        # Only register single-character keys; multi-char keys (e.g. "10") are
        # invalid in prompt_toolkit and must be selected via arrow keys + Enter.
        for _i, _key in enumerate(keys):
            if _key == _FREE_TEXT_SENTINEL or len(_key) > 1:
                continue

            @kb.add(_key)
            def _select_direct(event, k=_key):
                event.app.exit(result=k)

        if allow_free_text:

            @kb.add("/")
            def _free_text_shortcut(event):
                event.app.exit(result=_FREE_TEXT_SENTINEL)

        items = list(display_choices.items())

        def _get_formatted_text():
            lines = []
            visible_end = min(offset[0] + max_visible, total)
            if total > max_visible:
                lines.append(("ansiyellow", f"  ({offset[0] + 1}-{visible_end} of {total})\n"))
            for i in range(offset[0], visible_end):
                key, display = items[i]
                is_sel = i == selected[0]
                is_current = key == current and key != _FREE_TEXT_SENTINEL
                if key == _FREE_TEXT_SENTINEL:
                    label = f"  [/] {display}"
                elif key == display:
                    suffix = " (current)" if is_current else ""
                    label = f"  {display}{suffix}"
                else:
                    suffix = " (current)" if is_current else ""
                    label = f"  [{key}] {display}{suffix}"
                if is_sel:
                    lines.append((CLR_CURSOR, f"  {SYM_ARROW}{label}\n"))
                elif is_current:
                    lines.append((CLR_CURRENT, f"    {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
            return lines

        app = Application(
            layout=Layout(
                Window(FormattedTextControl(_get_formatted_text, show_cursor=False), always_hide_cursor=True)
            ),
            key_bindings=kb,
            full_screen=False,
        )

        result = _run_sub_application(app)
        if allow_free_text and result == _FREE_TEXT_SENTINEL:
            console.print()
            console.print("[dim](Paste supported. Enter to submit)[/]")
            return prompt_input(console, message="Your input", multiline=True)
        return result

    except (KeyboardInterrupt, EOFError):
        print_warning(console, "\nInput cancelled")
        return default
    except Exception as e:
        logger.error(f"Interactive select error: {e}")
        print_error(console, f"Selection error: {str(e)}")
        return default


def select_multi_choice(
    console: Console,
    choices: Dict[str, str],
    default_selected: Optional[List[str]] = None,
    allow_free_text: bool = False,
) -> List[str]:
    """Interactive multi-select with arrow-key navigation and Space to toggle.

    Uses prompt_toolkit Application for proper terminal handling.
    Up/Down arrows to navigate, Space to toggle, ``a`` to toggle all,
    Enter to confirm selection.
    When ``allow_free_text`` is True, a "Type custom answer..." entry is appended;
    choosing it (or pressing ``/``) opens the multiline input prompt.

    Args:
        console: Rich Console (used for fallback output on error)
        choices: Ordered dict of {key: display_text}
        default_selected: List of keys that should be pre-selected
        allow_free_text: When True, append a free-text option and allow ``/`` shortcut.

    Returns:
        List of selected choice keys, or a single-element list with user's free-text input.
        Empty list if nothing is selected.
    """
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        display_choices = dict(choices)
        if allow_free_text:
            display_choices[_FREE_TEXT_SENTINEL] = "Type custom answer..."

        keys = list(display_choices.keys())
        total = len(keys)
        cursor = [0]
        checked = {k for k in (default_selected or []) if k in keys and k != _FREE_TEXT_SENTINEL}

        # Scroll window: reserve 4 lines for scroll header + footer hint + safety.
        term_height = shutil.get_terminal_size((120, 40)).lines
        max_visible = max(3, term_height - 4)
        offset = [0]

        def _ensure_visible() -> None:
            if cursor[0] < offset[0]:
                offset[0] = cursor[0]
            elif cursor[0] >= offset[0] + max_visible:
                offset[0] = cursor[0] - max_visible + 1

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            cursor[0] = (cursor[0] - 1) % total
            if cursor[0] == total - 1:
                offset[0] = max(0, total - max_visible)
            else:
                _ensure_visible()

        @kb.add("down")
        def _move_down(event):
            cursor[0] = (cursor[0] + 1) % total
            if cursor[0] == 0:
                offset[0] = 0
            else:
                _ensure_visible()

        @kb.add("pageup")
        def _page_up(event):
            cursor[0] = max(0, cursor[0] - max_visible)
            _ensure_visible()

        @kb.add("pagedown")
        def _page_down(event):
            cursor[0] = min(total - 1, cursor[0] + max_visible)
            _ensure_visible()

        @kb.add("space")
        def _toggle(event):
            key = keys[cursor[0]]
            if key == _FREE_TEXT_SENTINEL:
                return
            if key in checked:
                checked.discard(key)
            else:
                checked.add(key)

        @kb.add("a")
        def _toggle_all(event):
            real_keys = [k for k in keys if k != _FREE_TEXT_SENTINEL]
            if all(k in checked for k in real_keys):
                checked.clear()
            else:
                checked.update(real_keys)

        @kb.add("enter")
        def _confirm(event):
            event.app.exit(result=[k for k in keys if k in checked])

        @kb.add("c-c")
        def _cancel(event):
            event.app.exit(result=[])

        if allow_free_text:

            @kb.add("/")
            def _free_text_shortcut(event):
                event.app.exit(result=[_FREE_TEXT_SENTINEL])

        items = list(display_choices.items())

        def _get_formatted_text():
            lines = []
            visible_end = min(offset[0] + max_visible, total)
            if total > max_visible:
                lines.append(("ansiyellow", f"  ({offset[0] + 1}-{visible_end} of {total})\n"))
            for i in range(offset[0], visible_end):
                key, display = items[i]
                is_cur = i == cursor[0]
                if key == _FREE_TEXT_SENTINEL:
                    label = f"  [/] {display}"
                else:
                    mark = "\u2713" if key in checked else " "
                    label = f"  [{mark}] {display}"
                if is_cur:
                    lines.append((CLR_CURSOR, f"    {label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
            lines.append(("ansibrightblack", "  [Space] toggle  [a] all  [Enter] confirm\n"))
            return lines

        app = Application(
            layout=Layout(
                Window(FormattedTextControl(_get_formatted_text, show_cursor=False), always_hide_cursor=True)
            ),
            key_bindings=kb,
            full_screen=False,
        )

        result = _run_sub_application(app)

        if allow_free_text and result == [_FREE_TEXT_SENTINEL]:
            console.print()
            console.print("[dim](Paste supported. Enter to submit)[/]")
            text = prompt_input(console, message="Your input", multiline=True)
            return [text] if text else []

        return result

    except (KeyboardInterrupt, EOFError):
        print_warning(console, "\nInput cancelled")
        return []
    except Exception as e:
        logger.error(f"Interactive multi-select error: {e}")
        print_error(console, f"Multi-select error: {str(e)}")
        return []


def select_list(
    console: Console,
    items: List[List[str]],
    headers: Optional[List[str]] = None,
    column_widths: Optional[List[int]] = None,
    max_visible: int = 15,
) -> Optional[int]:
    """Interactive list selector with arrow-key navigation.

    Each item occupies two lines:
      - Line 1 (primary): items[i][0] — the main content (highlighted when selected)
      - Line 2 (secondary): remaining columns joined with ``  `` (dim / bright when selected)

    Up/Down arrows to navigate (with wrap-around), PageUp/PageDown for jumping,
    Enter to confirm, q/Ctrl+C to cancel.

    Args:
        console: Rich Console (used for fallback output on error)
        items: List of rows, each row is a list of column strings.
               First element is the primary line; remaining elements form the secondary line.
        headers: Ignored (kept for API compatibility).
        column_widths: Ignored (kept for API compatibility).
        max_visible: Maximum items visible at once before scrolling.

    Returns:
        Selected row index (0-based), or None if cancelled or empty.
    """
    if not items:
        return None

    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        term_size = shutil.get_terminal_size((120, 40))
        term_width = term_size.columns
        term_height = term_size.lines
        content_width = term_width - 6  # leave room for "  → " prefix

        total = len(items)
        selected = [0]
        offset = [0]  # scroll offset

        # Each item takes 3 lines (primary + secondary + blank); reserve 2 for scroll info + hint
        max_items_by_height = max(1, (term_height - 2) // 3)
        max_visible = min(max_visible, max_items_by_height)

        def _display_width(text: str) -> int:
            """Calculate terminal display width accounting for wide chars."""
            w = 0
            for ch in text:
                w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            return w

        def _clip(text: str, width: int) -> str:
            """Hard-clip text to terminal display width (CJK-aware)."""
            w = 0
            for i, ch in enumerate(text):
                cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                if w + cw > width:
                    return text[:i]
                w += cw
            return text

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            selected[0] = (selected[0] - 1) % total
            if selected[0] < offset[0]:
                offset[0] = selected[0]
            elif selected[0] == total - 1:
                offset[0] = max(0, total - max_visible)

        @kb.add("down")
        def _move_down(event):
            selected[0] = (selected[0] + 1) % total
            if selected[0] >= offset[0] + max_visible:
                offset[0] = selected[0] - max_visible + 1
            elif selected[0] == 0:
                offset[0] = 0

        @kb.add("pageup")
        def _page_up(event):
            selected[0] = max(0, selected[0] - max_visible)
            offset[0] = max(0, offset[0] - max_visible)

        @kb.add("pagedown")
        def _page_down(event):
            selected[0] = min(total - 1, selected[0] + max_visible)
            offset[0] = min(max(0, total - max_visible), offset[0] + max_visible)

        @kb.add("enter")
        def _confirm(event):
            event.app.exit(result=selected[0])

        @kb.add("escape")
        def _escape(event):
            event.app.exit(result=None)

        @kb.add("c-c")
        def _cancel(event):
            event.app.exit(result=None)

        def _get_formatted_text():
            lines = []

            # Scroll info
            visible_end = min(offset[0] + max_visible, total)
            if total > max_visible:
                lines.append(("ansiyellow", f"  ({offset[0] + 1}-{visible_end} of {total})\n"))

            for i in range(offset[0], visible_end):
                row = items[i]
                primary = _clip(row[0], content_width) if row else ""
                secondary = "  ".join(row[1:]) if len(row) > 1 else ""
                secondary = _clip(secondary, content_width)

                is_sel = i == selected[0]
                if is_sel:
                    lines.append((CLR_CURSOR, f"  {SYM_ARROW} {primary}\n"))
                    lines.append((CLR_CURSOR, f"      {secondary}\n"))
                else:
                    lines.append(("", f"    {primary}\n"))
                    lines.append(("ansibrightblack", f"      {secondary}\n"))
                lines.append(("", "\n"))

            lines.append(("ansibrightblack", "  [\u2191\u2193] navigate  [Enter] select  [Esc] cancel\n"))
            return lines

        app = Application(
            layout=Layout(
                Window(FormattedTextControl(_get_formatted_text, show_cursor=False), always_hide_cursor=True)
            ),
            key_bindings=kb,
            full_screen=False,
        )

        return _run_sub_application(app)

    except (KeyboardInterrupt, EOFError):
        print_warning(console, "\nSelection cancelled.")
        return None
    except Exception as e:
        logger.error(f"Interactive list error: {e}")
        print_error(console, f"List selection error: {str(e)}")
        return None


def prompt_input(
    console: Console,
    message: str,
    default: str = "",
    choices: list = None,
    multiline: bool = False,
    style=None,
    allow_interrupt: bool = False,
    is_password: bool = False,
):
    """
    Unified input method using prompt_toolkit to avoid conflicts with rich.Prompt.ask().

    Args:
        message: The prompt message to display
        default: Default value if user presses Enter without input
        choices: List of valid choices (validates input)
        multiline: Whether to allow multiline input
        is_password: Mask input with ``*`` (e.g. API keys). Mutually exclusive
            with ``multiline`` (prompt_toolkit ignores the mask in multiline).

    Returns:
        User input string or default value
    """
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.validation import ValidationError, Validator

        # Format the prompt message
        if default:
            prompt_text = f"{message} ({default}): "
        else:
            prompt_text = f"{message}: "

        # Create validator for choices if provided
        validator = None
        if choices:

            class ChoiceValidator(Validator):
                def validate(self, document):
                    text = document.text.strip()
                    if text and text not in choices:
                        raise ValidationError(message=f"Please choose from: {', '.join(choices)}")

            validator = ChoiceValidator()

            # Add choices to prompt text
            prompt_text = f"{message} ({'/'.join(choices)}): "
            # if default:
            #     prompt_text = f"{message} ({'/'.join(choices)}) ({default}): "

        # Use the existing session for consistency but create a temporary one for this input
        from prompt_toolkit.history import InMemoryHistory

        if not style:
            style = Style.from_dict(PROMPT_ONLY_STYLE)

        # When multiline, override Enter to submit (newlines only via paste).
        key_bindings = None
        if multiline:
            from prompt_toolkit.key_binding import KeyBindings

            key_bindings = KeyBindings()

            @key_bindings.add("enter")
            def _submit(event):
                event.current_buffer.validate_and_handle()

        def _do_prompt():
            return prompt(
                HTML(f"<ansigreen><b>{prompt_text}</b></ansigreen>"),
                default=default,
                validator=validator,
                multiline=multiline,
                key_bindings=key_bindings,
                history=InMemoryHistory(),
                style=style,
                is_password=is_password and not multiline,
            )

        result = _run_prompt_in_terminal(_do_prompt)

        return result if multiline else result.strip()

    except (KeyboardInterrupt, EOFError):
        if allow_interrupt:
            raise
        # Handle Ctrl+C or Ctrl+D gracefully
        print_warning(console, "\nInput cancelled")
        return default
    except Exception as e:
        logger.error(f"Input prompt error: {e}")
        print_error(console, f"Input error: {str(e)}")
        return default


def confirm_prompt(console: Console, message: str, default: bool = False) -> bool:
    """Yes/No prompt built on :func:`select_choice` — TUI worker-thread safe.

    Replaces ``rich.prompt.Confirm.ask`` for callers that run inside the
    :class:`~datus.cli.tui.app.DatusApp` worker thread, where the rich
    prompt competes with prompt_toolkit's raw-mode stdin and swallows
    keys. ``y``/``n`` shortcuts also work via ``select_choice``'s
    single-character direct-select binding.
    """
    console.print(f"[bold]{message}[/bold]")
    choices = {"y": "Yes", "n": "No"}
    default_key = "y" if default else "n"
    result = select_choice(console, choices, default=default_key)
    return result == "y"
