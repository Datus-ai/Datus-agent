# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers for normalizing metric subject paths."""

from __future__ import annotations

from typing import Iterable, Optional

GENERIC_METRIC_SUBJECT_ROOTS = {"metrics", "metric", "unknown"}


def default_metric_subject_path(datasource: str = "", table_name: str = "") -> list[str]:
    datasource = str(datasource or "").strip()
    table = str(table_name or "Unknown").strip() or "Unknown"
    if datasource:
        return [datasource, table]
    return ["Metrics", table]


def normalize_metric_subject_path(
    subject_path: Optional[Iterable[str]],
    *,
    datasource: str = "",
    table_name: str = "",
) -> list[str]:
    """Normalize generated metric subject paths without rewriting business roots.

    The datasource is a storage scope, so generic roots such as ``Metrics`` should
    not become a parallel top-level subject tree under a datasource-scoped run.
    Generated paths also commonly include the datasource twice
    (``ac_manage/ac_manage/activity``); collapse only that exact duplicate root.
    """

    parts = [str(part).strip() for part in subject_path or [] if str(part).strip()]
    if not parts:
        return default_metric_subject_path(datasource, table_name)

    datasource = str(datasource or "").strip()
    if not datasource:
        return parts

    first = _normalize_part(parts[0])
    if first in GENERIC_METRIC_SUBJECT_ROOTS:
        tail = parts[1:]
    elif _same_part(parts[0], datasource):
        tail = parts[1:]
    else:
        return parts

    while tail and _same_part(tail[0], datasource):
        tail = tail[1:]

    if not tail:
        tail = [str(table_name or "Unknown").strip() or "Unknown"]
    return [datasource, *tail]


def normalize_metric_subject_tree_tag(tag: str, *, datasource: str = "", table_name: str = "") -> str:
    prefix = "subject_tree:"
    if not isinstance(tag, str) or not tag.startswith(prefix):
        return tag
    path = tag.split(prefix, 1)[1].strip()
    parts = [part.strip() for part in path.split("/") if part.strip()]
    normalized = normalize_metric_subject_path(parts, datasource=datasource, table_name=table_name)
    return f"{prefix} {'/'.join(normalized)}"


def _same_part(left: str, right: str) -> bool:
    return _normalize_part(left) == _normalize_part(right)


def _normalize_part(value: str) -> str:
    return str(value or "").strip().lower()
