# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# Utility for better display in the console using rich
from typing import Any, Dict, Optional

from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree


def dict_to_tree(data: Dict[str, Any], tree: Optional[Tree] = None, console=None) -> Tree:
    """
    Convert a dictionary to a tree structure for display.

    Args:
        data: The dictionary to convert.
        tree: An optional existing tree to append to.
        console: Optional rich Console, used to get screen width for truncation.

    Returns:
        A Tree object representing the dictionary structure.
    """
    if tree is None:
        tree = Tree("--")

    # If console is not provided, use a default width
    screen_width = console.size.width if console else 100
    max_length = screen_width * 3
    # get theme from console
    theme = "light"  # "monokai"

    for key, value in data.items():
        if isinstance(value, dict) and value:
            branch = tree.add(f"[bold blue]{key}[/]")
            dict_to_tree(value, branch, console)
        elif isinstance(value, list) and value:
            branch = tree.add(f"[bold blue]{key}[/]")
            for index, item in enumerate(value):
                item_key = f"{key}[{index}]"
                if isinstance(item, dict):
                    dict_to_tree(item, branch.add(f"[bold blue]{item_key}[/]"), console)
                else:
                    item_str = str(item)
                    if len(item_str) > max_length:
                        item_str = item_str[: max_length - 3] + "..."
                    branch.add(f"[bold blue]{item_key}[/]: {item_str}")
        else:
            value_str = str(value)
            if key == "sql_query":
                sql_code = value if isinstance(value, str) else str(value)
                tree.add(
                    Group(
                        Text(f"{key}:", style="bold blue"),
                        Syntax(sql_code, "sql", theme=theme, line_numbers=False, word_wrap=True),
                    )
                )
            elif len(value_str) > max_length:
                value_str = value_str[: max_length - 3] + "..."
                tree.add(f"[bold blue]{key}[/]: {value_str}")
            else:
                tree.add(f"[bold blue]{key}[/]: {value_str}")
    return tree
