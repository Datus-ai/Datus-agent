"""Resolve persisted datasource identities against an already authorized config.

Names are legacy references and display labels. An explicit ID is authoritative:
an unknown ID must never fall through to a same-named source or the default.
Standalone configurations without host IDs keep their YAML keys as identities.
"""

from typing import Any, Mapping, Optional

from datus.utils.exceptions import DatusException, ErrorCode


def datasource_identity(config: Any, name: str) -> str:
    entry = config.services.datasources.get(name)
    identity = getattr(entry, "datasource_id", "")
    return identity if isinstance(identity, str) and identity else name


def resolve_datasource(config: Any, *, datasource_id: Optional[str] = None, datasource: Optional[str] = None) -> str:
    sources = getattr(getattr(config, "services", None), "datasources", None)
    if not isinstance(sources, dict):
        if datasource_id is None:
            current = getattr(config, "current_datasource", "")
            return datasource or (current if isinstance(current, str) else "")
        raise DatusException(
            ErrorCode.COMMON_VALIDATION_FAILED, message="A datasource ID requires an authorized roster"
        )
    if datasource_id is not None:
        matches = [name for name in sources if datasource_identity(config, name) == datasource_id]
        if datasource_id and len(matches) == 1:
            return matches[0]
    elif datasource:
        matches = [
            name
            for name, entry in sources.items()
            if datasource == name or datasource in getattr(entry, "datasource_aliases", [])
        ]
        if len(matches) == 1:
            return matches[0]
    else:
        return config.current_datasource or ""
    raise DatusException(
        ErrorCode.COMMON_VALIDATION_FAILED,
        message=f"Datasource {datasource_id if datasource_id is not None else datasource!r} "
        "is not bound to this project; select a valid datasource",
    )


def resolve_execution_context(config: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    name = resolve_datasource(
        config,
        datasource_id=values.get("datasource_id", values.get("datasourceId")),
        datasource=values.get("datasource_name", values.get("datasourceName", values.get("datasource"))),
    )
    result["datasource"] = name
    sources = getattr(getattr(config, "services", None), "datasources", None)
    if (
        name
        and isinstance(sources, dict)
        and isinstance(getattr(sources[name], "datasource_id", None), str)
        and sources[name].datasource_id
    ):
        result["datasource_id"] = datasource_identity(config, name)
    return result


def configured_datasource_id(config: Any, name: str) -> Optional[str]:
    """Only hosts supplying a concrete ID opt into ID-bearing persistence."""
    sources = getattr(getattr(config, "services", None), "datasources", None)
    if isinstance(sources, dict):
        # current_db_config also accepts physical database names and may return
        # the primary config for an unknown key. Never use that fallback to
        # assign an identity to a stale artifact label.
        if name not in sources:
            try:
                name = resolve_datasource(config, datasource=name)
            except DatusException:
                return None
        entry = sources.get(name)
        value = getattr(entry, "datasource_id", None)
        return value if isinstance(value, str) and value else None
    getter = getattr(config, "current_db_config", None)
    if not callable(getter):
        return None
    value = getattr(getter(name), "datasource_id", None)
    return value if isinstance(value, str) and value else None


def legacy_semantic_paths(config: Any, name: str) -> list[str]:
    getter = getattr(getattr(config, "path_manager", None), "legacy_semantic_model_paths", None)
    paths = getter(name) if callable(getter) else []
    return [value for value in paths if isinstance(value, str)] if isinstance(paths, (list, tuple)) else []


def resolve_subagent_config(config: Any, node: Any) -> Any:
    """Translate frozen ID scopes only on a runtime copy, preserving the snapshot."""
    if not isinstance(node, dict) or not isinstance(node.get("scoped_context"), dict):
        return node
    scope = node["scoped_context"]
    if scope.get("datasource_id") is None and scope.get("tables_by_datasource_id") is None:
        return node
    resolved = dict(scope)
    identity = resolved.pop("datasource_id", None)
    if identity is not None:
        resolved["datasource"] = resolve_datasource(config, datasource_id=identity)
    groups = resolved.pop("tables_by_datasource_id", None)
    if groups is not None:
        resolved["tables_by_datasource"] = {
            resolve_datasource(config, datasource_id=key): value for key, value in groups.items()
        }
    return {**node, "scoped_context": resolved}
