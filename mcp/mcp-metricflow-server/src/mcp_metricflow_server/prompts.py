"""Prompt management for MCP MetricFlow Server."""

from pathlib import Path


def get_prompt(name: str) -> str:
    """
    Get prompt content from markdown files.

    Args:
        name: Name of the prompt file (e.g., 'list_metrics')

    Returns:
        Content of the prompt file
    """
    # Get the docs directory relative to this file
    docs_dir = Path(__file__).parent.parent.parent / "docs"
    prompt_file = docs_dir / f"{name}.md"

    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    else:
        # Return a basic description if file doesn't exist
        return f"Tool: {name}"
