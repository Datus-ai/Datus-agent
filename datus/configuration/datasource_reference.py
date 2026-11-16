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
    sources = config.services.datasources
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
        datasource=values.get("datasource"),
    )
    result["datasource"] = name
    if name and getattr(config.services.datasources[name], "datasource_id", ""):
        result["datasource_id"] = datasource_identity(config, name)
    return result


def configured_datasource_id(config: Any, name: str) -> Optional[str]:
    """Only hosts supplying a concrete ID opt into ID-bearing persistence."""
    getter = getattr(config, "current_db_config", None)
    if not callable(getter):
        return None
    value = getattr(getter(name), "datasource_id", None)
    return value if isinstance(value, str) and value else None


def legacy_semantic_paths(config: Any, name: str) -> list[str]:
    getter = getattr(getattr(config, "path_manager", None), "legacy_semantic_model_paths", None)
    paths = getter(name) if callable(getter) else []
    return [value for value in paths if isinstance(value, str)] if isinstance(paths, (list, tuple)) else []
