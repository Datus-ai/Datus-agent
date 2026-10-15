# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Host-managed metric identities carried in DATUS extension metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from cuid2 import Cuid

from datus.storage.metric.store import normalize_metric_name

_UID_PATTERN = re.compile(r"[a-z][a-z0-9]{23}")
_UID_GENERATOR = Cuid(length=24)


def _extension(metric: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    extensions = metric.get("custom_extensions", [])
    if not isinstance(extensions, list) or any(not isinstance(item, dict) for item in extensions):
        raise ValueError(f"{metric.get('name')}: custom_extensions must be a list of objects")
    entries = [item for item in extensions if str(item.get("vendor_name", "")).strip().upper() == "DATUS"]
    if len(entries) > 1:
        raise ValueError(f"{metric.get('name')}: metric identity requires a single DATUS extension")
    if not entries:
        return None, {}
    raw = entries[0].get("data")
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise ValueError(f"{metric.get('name')}: invalid DATUS extension JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{metric.get('name')}: DATUS extension data must encode an object")
    return entries[0], data


def validate_metric_uid(uid: Any, metric_name: str) -> str:
    """Validate a 24-character CUID2, allowing missing identities in legacy files."""
    if uid is None or uid == "":
        return ""
    if not isinstance(uid, str) or not _UID_PATTERN.fullmatch(uid):
        raise ValueError(f"{metric_name}: uid must be a 24-character CUID2 ([a-z][a-z0-9]{{23}})")
    return uid


def metric_uid(metric: dict[str, Any]) -> str:
    """Read and validate a persisted 24-character CUID2 identity."""
    _, payload = _extension(metric)
    return validate_metric_uid(payload.get("uid"), str(metric.get("name") or ""))


def _set_metric_uid(metric: dict[str, Any], uid: str) -> bool:
    """Set or remove one metric's UID without mutating shared YAML extensions."""
    extension, payload = _extension(metric)
    payload = dict(payload)
    if uid:
        if payload.get("uid") == uid:
            return False
        payload["uid"] = uid
    else:
        if "uid" not in payload:
            return False
        del payload["uid"]
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    extensions = list(metric.get("custom_extensions", []))
    if extension is None:
        extensions.append({"vendor_name": "DATUS", "data": data})
    else:
        # YAML anchors can share extension objects across different metrics.
        extensions = [{**item, "data": data} if item is extension else item for item in extensions]
    metric["custom_extensions"] = extensions
    return True


def ensure_metric_uid(metric: dict[str, Any], previous: dict[str, Any] | None = None) -> bool:
    """Preserve the authoritative UID, generating one when it is missing.

    With a prior definition, only its UID is authoritative. Without one, the
    metric is an already-persisted definition being prepared for validation.
    Return whether its extension changed.
    """
    uid = metric_uid(previous if previous is not None else metric)
    return _set_metric_uid(metric, uid or _UID_GENERATOR.generate())


def _named_metrics(document: Any) -> list[dict[str, Any]]:
    """Read named metrics without collapsing entries; native validation checks structure."""
    models = document.get("semantic_model") if isinstance(document, dict) else None
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        return []
    metrics = models[0].get("metrics", [])
    if not isinstance(metrics, list):
        return []
    return [metric for metric in metrics if isinstance(metric, dict) and normalize_metric_name(metric.get("name"))]


def metric_definitions(document: Any) -> dict[str, dict[str, Any]]:
    """Index exact names so edits can repair a pre-existing case-insensitive conflict."""
    return {str(metric["name"]).strip(): metric for metric in _named_metrics(document)}


def _validate_metric_names(metrics: list[dict[str, Any]]) -> None:
    """Reject names that would collide under the KB's datasource-local business key."""
    owners: dict[str, str] = {}
    for metric in metrics:
        name = str(metric.get("name") or "").strip()
        key = normalize_metric_name(name)
        if not key:
            continue
        if key in owners:
            raise ValueError(
                f"Metric name conflict: {owners[key]!r} and {name!r}. "
                "Metric names must be unique ignoring case within a datasource."
            )
        owners[key] = name


def validate_unique_metric_uids(metrics: Iterable[tuple[str, str]]) -> None:
    """Reject shared identities within a model while allowing missing legacy UIDs."""
    owners: dict[str, str] = {}
    for name, uid in metrics:
        if not uid:
            continue
        if uid in owners:
            raise ValueError(f"Duplicate metric uid {uid}: {owners[uid]} and {name}")
        owners[uid] = name


def validate_metric_uids(metrics: list[dict[str, Any]]) -> None:
    """Reject conflicting names and identities within a model, allowing legacy missing ids."""
    _validate_metric_names(metrics)
    validate_unique_metric_uids((str(metric.get("name") or ""), metric_uid(metric)) for metric in metrics)


def prepare_metric_uids(document: Any, previous: Any = None) -> list[str]:
    """Preserve prior identities during edits, or fill missing UIDs at finalization.

    An edit supplies its original document: unmatched candidate UIDs are cleared
    for host generation at finalization. Finalization supplies no prior document
    and preserves every persisted UID while assigning missing ones.
    """
    metrics = _named_metrics(document)
    _validate_metric_names(metrics)
    old_metrics = metric_definitions(previous)
    old_metrics_by_key: dict[str, list[dict[str, Any]]] = {}
    old_metrics_by_uid: dict[str, list[dict[str, Any]]] = {}
    for metric in _named_metrics(previous):
        name = str(metric["name"]).strip()
        old_metrics_by_key.setdefault(normalize_metric_name(name), []).append(metric)
        try:
            uid = metric_uid(metric)
        except ValueError:
            # A removed malformed definition cannot establish a rename. A
            # retained one is still rejected when its identity is preserved.
            continue
        if uid:
            old_metrics_by_uid.setdefault(uid, []).append(metric)

    matched: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        name = str(metric["name"]).strip()
        old_metric = old_metrics.get(name)
        if old_metric is None:
            # A case-only rename can retain its UID when the old name is unambiguous.
            matches = old_metrics_by_key.get(normalize_metric_name(name), [])
            if len(matches) == 1:
                old_metric = matches[0]
        if old_metric is not None:
            matched[name] = old_metric

    # An explicit UID may identify a rename only when both sides are unambiguous
    # and the old definition is no longer matched to a retained metric.
    unmatched_by_uid: dict[str, list[str]] = {}
    for metric in metrics:
        name = str(metric["name"]).strip()
        if name not in matched:
            _, payload = _extension(metric)
            uid = payload.get("uid")
            if isinstance(uid, str) and uid:
                unmatched_by_uid.setdefault(uid, []).append(name)
    retained = {id(metric) for metric in matched.values()}
    for uid, names in unmatched_by_uid.items():
        prior = old_metrics_by_uid.get(uid, [])
        if len(names) == 1 and len(prior) == 1 and id(prior[0]) not in retained:
            matched[names[0]] = prior[0]

    changed = []
    for metric in metrics:
        if previous is None:
            updated = ensure_metric_uid(metric)
        else:
            prior = matched.get(str(metric["name"]).strip())
            updated = _set_metric_uid(metric, metric_uid(prior) if prior is not None else "")
        if updated:
            changed.append(metric["name"])
    validate_metric_uids(metrics)
    return changed
