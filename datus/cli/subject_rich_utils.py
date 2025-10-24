# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from typing import Any

from rich.console import RenderableType
from rich.syntax import Syntax
from rich.text import Text

from datus.utils.json_utils import to_pretty_str

SQL_TAG_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]


def build_historical_sql_tags(tags: Any, tag_splitter: str = " ") -> RenderableType:
    if not tags:
        return Text()
    if isinstance(tags, list) or isinstance(tags, dict):
        return Syntax(to_pretty_str(tags), lexer="json")

    tags = [t.strip() for t in str(tags).split(",")]
    tags_text = Text()
    for i, tag in enumerate(tags):
        color = SQL_TAG_COLORS[i % len(SQL_TAG_COLORS)]
        tags_text.append(f" {tag} ", style=f"white on {color}")
        tags_text.append(tag_splitter)
    return tags_text
