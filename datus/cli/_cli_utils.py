from typing import Dict

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
    When ``allow_free_text`` is True, a "Type custom answer..." entry is appended
    and the user can also press ``/`` at any time to enter free-text mode.

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

        # State: inline editing for free-text option
        editing = [False]
        text_buf = [""]

        kb = KeyBindings()

        @kb.add("up")
        def _move_up(event):
            if editing[0]:
                return  # ignore arrow keys while editing
            selected[0] = (selected[0] - 1) % len(keys)

        @kb.add("down")
        def _move_down(event):
            if editing[0]:
                return
            selected[0] = (selected[0] + 1) % len(keys)

        @kb.add("enter")
        def _confirm(event):
            if editing[0]:
                event.app.exit(result=text_buf[0])
            else:
                sel = keys[selected[0]]
                if sel == _FREE_TEXT_SENTINEL:
                    editing[0] = True  # enter inline editing mode
                else:
                    event.app.exit(result=sel)

        @kb.add("c-c")
        def _cancel(event):
            if editing[0]:
                editing[0] = False
                text_buf[0] = ""
            else:
                event.app.exit(result=default)

        @kb.add("backspace")
        def _backspace(event):
            if editing[0]:
                text_buf[0] = text_buf[0][:-1]

        # Direct shortcut keys (press y/a/n to pick immediately)
        for _i, _key in enumerate(keys):
            if _key == _FREE_TEXT_SENTINEL:
                continue

            @kb.add(_key)
            def _select_direct(event, k=_key):
                if editing[0]:
                    text_buf[0] += k
                else:
                    event.app.exit(result=k)

        if allow_free_text:

            @kb.add("/")
            def _free_text_shortcut(event):
                if editing[0]:
                    text_buf[0] += "/"
                else:
                    # Jump to free-text and start editing
                    selected[0] = keys.index(_FREE_TEXT_SENTINEL)
                    editing[0] = True

            @kb.add("<any>")
            def _any_key(event):
                ch = event.data
                if not ch.isprintable() or len(ch) != 1:
                    return
                if editing[0]:
                    text_buf[0] += ch
                elif keys[selected[0]] == _FREE_TEXT_SENTINEL:
                    # Start inline editing with first character
                    editing[0] = True
                    text_buf[0] = ch

        def _get_formatted_text():
            lines = []
            for i, (key, display) in enumerate(display_choices.items()):
                is_sel = i == selected[0]
                if key == _FREE_TEXT_SENTINEL:
                    if editing[0]:
                        cursor = "\u2588"  # block cursor
                        label = f"  [/] {text_buf[0]}{cursor}"
                    else:
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

        return app.run()

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Input cancelled[/]")
        return default
    except Exception as e:
        logger.error(f"Interactive select error: {e}")
        console.print(f"[bold red]Selection error:[/] {str(e)}")
        return default


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

        result = prompt(
            HTML(f"<ansigreen><b>{prompt_text}</b></ansigreen>"),
            default=default,
            validator=validator,
            multiline=multiline,
            history=InMemoryHistory(),  # Separate history for sub-prompts
            style=style,  # Use same style as main session
        )

        return result.strip()

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
