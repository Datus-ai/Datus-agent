"""Matching for catalog picker scopes and submitted chat references."""

import re
from collections.abc import Mapping


def catalog_scope_matches(coordinates: Mapping[str, str], pattern: Mapping[str, str]) -> bool:
    if not pattern.get("table_name"):
        return False

    return all(
        not expected or re.fullmatch(re.escape(expected).replace(r"\*", ".*"), coordinates.get(field, "")) is not None
        for field, expected in pattern.items()
    )
