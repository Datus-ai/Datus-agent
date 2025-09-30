# -*- coding: utf-8 -*-
"""YAML editor with syntax highlighting for interactive editing."""

import tempfile
from typing import Tuple

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.data import YamlLexer
from rich.console import Console

from datus.utils.loggings import get_logger

logger = get_logger(__name__)
console = Console()


def edit_yaml_multiline(yaml_content: str, title: str = "Edit YAML") -> Tuple[str, bool]:
    """
    Interactive YAML editor with syntax highlighting.

    Args:
        yaml_content: Initial YAML content to edit
        title: Title to display

    Returns:
        Tuple of (edited_content, confirmed)
        - edited_content: Edited YAML content
        - confirmed: True if user confirmed, False if cancelled
    """
    console.print(f"\n[bold cyan]{title}[/]")
    console.print("[dim]Press Ctrl+D or type 'done' on a new line to finish[/]")
    console.print("[dim]Press Ctrl+C to cancel[/]")
    console.print("-" * 50)

    try:
        # Create prompt session with YAML syntax highlighting
        session = PromptSession(lexer=PygmentsLexer(YamlLexer))

        # Collect lines
        lines = []
        while True:
            try:
                line = session.prompt("")
                if line.strip().lower() == "done":
                    break
                lines.append(line)
            except EOFError:
                # Ctrl+D pressed
                break

        edited_content = "\n".join(lines) if lines else yaml_content

        # Validate YAML syntax
        try:
            yaml.safe_load_all(edited_content)
        except yaml.YAMLError as e:
            console.print(f"[red]YAML syntax error: {e}[/]")
            console.print("[yellow]Do you want to retry editing? (y/n)[/]")
            retry = input().strip().lower()
            if retry in ["y", "yes"]:
                return edit_yaml_multiline(edited_content, title)
            else:
                return yaml_content, False

        return edited_content, True

    except KeyboardInterrupt:
        console.print("\n[yellow]Editing cancelled[/]")
        return yaml_content, False


def edit_yaml_in_editor(yaml_content: str) -> Tuple[str, bool]:
    """
    Edit YAML content in external editor (vim/nano/etc).

    Args:
        yaml_content: Initial YAML content

    Returns:
        Tuple of (edited_content, confirmed)
    """
    # Determine editor
    import os
    import subprocess

    editor = os.environ.get("EDITOR", "vim")

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_file:
        tmp_file.write(yaml_content)
        tmp_file_path = tmp_file.name

    try:
        # Open editor
        result = subprocess.run([editor, tmp_file_path])

        if result.returncode == 0:
            # Read edited content
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                edited_content = f.read()

            # Validate YAML
            try:
                yaml.safe_load_all(edited_content)
                return edited_content, True
            except yaml.YAMLError as e:
                console.print(f"[red]YAML syntax error: {e}[/]")
                console.print("[yellow]Press Enter to retry, or Ctrl+C to cancel[/]")
                input()
                return edit_yaml_in_editor(edited_content)
        else:
            console.print("[yellow]Editor exited with error[/]")
            return yaml_content, False

    except Exception as e:
        logger.error(f"Failed to edit YAML: {e}")
        console.print(f"[red]Error: {e}[/]")
        return yaml_content, False

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass
