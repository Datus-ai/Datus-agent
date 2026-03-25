import shutil
import unicodedata
from typing import Dict, List, Optional

from prompt_toolkit.styles import Style
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


_FREE_TEXT_SENTINEL = "__free_text__"


def select_choice(
    console: Console,
    choices: Dict[str, str],
    default: str = "",
    allow_free_text: bool = False,
) -> str:
    """Interactive choice selector with arrow-key navigation.

    Uses prompt_toolkit Application for proper terminal handling.
    Up/Down arrows to navigate, Enter to confirm, or press shortcut key directly.
    When ``allow_free_text`` is True, a "Type custom answer..." entry is appended.
    Choosing it, or pressing ``/``, opens the standard multiline input prompt so
    paste works reliably.

    Args:
        console: Rich Console (used for fallback output on error)
        choices: Ordered dict of {key: display_text}
                 e.g. {"y": "Allow (once)", "a": "Always allow (session)", "n": "Deny"}
        default: Default choice key (pre-selected on start)
        allow_free_text: When True, append a free-text option and allow ``/`` shortcut.

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
        selected = [keys.index(default) if default in keys else 0]

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            selected[0] = (selected[0] - 1) % len(keys)

        @kb.add("down")
        def _move_down(event):
            selected[0] = (selected[0] + 1) % len(keys)

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

        def _get_formatted_text():
            lines = []
            for i, (key, display) in enumerate(display_choices.items()):
                is_sel = i == selected[0]
                if key == _FREE_TEXT_SENTINEL:
                    label = f"  [/] {display}"
                else:
                    label = f"  [{key}] {display}"
                if is_sel:
                    lines.append(("ansicyan bold", f"  \u2192{label}\n"))
                else:
                    lines.append(("", f"    {label}\n"))
            return lines

        app = Application(
            layout=Layout(Window(FormattedTextControl(_get_formatted_text))),
            key_bindings=kb,
            full_screen=False,
        )

        result = app.run()
        if allow_free_text and result == _FREE_TEXT_SENTINEL:
            console.print()
            console.print("[dim](Paste supported. Enter to submit)[/]")
            return prompt_input(console, message="Your input", multiline=True)
        return result

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Input cancelled[/]")
        return default
    except Exception as e:
        logger.error(f"Interactive select error: {e}")
        console.print(f"[bold red]Selection error:[/] {str(e)}")
        return default


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
                    lines.append(("ansicyan bold", f"  \u2192 {primary}\n"))
                    lines.append(("ansicyan", f"      {secondary}\n"))
                else:
                    lines.append(("", f"    {primary}\n"))
                    lines.append(("ansibrightblack", f"      {secondary}\n"))
                lines.append(("", "\n"))

            lines.append(("ansibrightblack", "  [\u2191\u2193] navigate  [Enter] select  [Esc] cancel\n"))
            return lines

        app = Application(
            layout=Layout(Window(FormattedTextControl(_get_formatted_text))),
            key_bindings=kb,
            full_screen=False,
        )

        return app.run()

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Selection cancelled.[/]")
        return None
    except Exception as e:
        logger.error(f"Interactive list error: {e}")
        console.print(f"[bold red]List selection error:[/] {str(e)}")
        return None


def prompt_input(
    console: Console,
    message: str,
    default: str = "",
    choices: list = None,
    multiline: bool = False,
    style=None,
    allow_interrupt: bool = False,
):
    """
    Unified input method using prompt_toolkit to avoid conflicts with rich.Prompt.ask().

    Args:
        message: The prompt message to display
        default: Default value if user presses Enter without input
        choices: List of valid choices (validates input)
        multiline: Whether to allow multiline input

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
            style = Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                }
            )

        # When multiline, override Enter to submit (newlines only via paste).
        key_bindings = None
        if multiline:
            from prompt_toolkit.key_binding import KeyBindings

            key_bindings = KeyBindings()

            @key_bindings.add("enter")
            def _submit(event):
                event.current_buffer.validate_and_handle()

        result = prompt(
            HTML(f"<ansigreen><b>{prompt_text}</b></ansigreen>"),
            default=default,
            validator=validator,
            multiline=multiline,
            key_bindings=key_bindings,
            history=InMemoryHistory(),  # Separate history for sub-prompts
            style=style,  # Use same style as main session
        )

        return result if multiline else result.strip()

    except (KeyboardInterrupt, EOFError):
        if allow_interrupt:
            raise
        # Handle Ctrl+C or Ctrl+D gracefully
        console.print("\n[yellow]Input cancelled[/]")
        return default
    except Exception as e:
        logger.error(f"Input prompt error: {e}")
        console.print(f"[bold red]Input error:[/] {str(e)}")
        return default
