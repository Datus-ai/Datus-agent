# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
"""Resolve persisted artifact identities without changing runtime/KB namespaces."""

from typing import Any, Mapping


def resolve_artifact_datasource(config: Any, reference: Mapping[str, Any]) -> str:
    if config is None:
        identity = reference.get("datasource_id") if "datasource_id" in reference else reference.get("datasource")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("DATASOURCE_UNAVAILABLE: invalid artifact identity")
        return identity
    configs = config.services.datasources
    if "datasource_id" in reference:
        identity = reference["datasource_id"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("DATASOURCE_UNAVAILABLE: invalid artifact datasource_id")
        matches = [name for name, entry in configs.items() if (getattr(entry, "datasource_id", "") or name) == identity]
    else:
        legacy = reference.get("datasource")
        matches = [name for name in configs if legacy == name]
    if len(matches) != 1:
        raise ValueError("DATASOURCE_UNAVAILABLE: artifact datasource is missing or ambiguous")
    return matches[0]


def artifact_identity(config: Any, name: str) -> dict[str, str]:
    if config is None:
        # A directly supplied connector has no registry (standalone embedding).
        return {"datasource_id": name}
    entry = config.services.datasources.get(name)
    if entry is None:
        raise ValueError("DATASOURCE_UNAVAILABLE: artifact datasource is not configured")
    return {"datasource_id": getattr(entry, "datasource_id", "") or name}


def saved_artifact_context(config: Any, name: str, context: dict | None) -> dict | None:
    if context is None:
        return None
    return {**{key: value for key, value in context.items() if key != "datasource"}, **artifact_identity(config, name)}


def runtime_artifact_query(config: Any, query: Any) -> Any:
    """Return a request-local copy; never rewrite saved bytes or model snapshots."""
    name = resolve_artifact_datasource(config, query.model_dump())
    context = query.execution_context
    if context is not None:
        context = {key: value for key, value in context.items() if key != "datasource_id"}
        context["datasource"] = name
    return query.model_copy(
        update={**artifact_identity(config, name), "datasource": name, "execution_context": context}
    )


def artifact_reference_key(query: Any, config: Any = None) -> str:
    name = resolve_artifact_datasource(config, query.model_dump())
    return artifact_identity(config, name)["datasource_id"]
