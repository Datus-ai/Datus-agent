from prompt_toolkit.styles import Style
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


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
