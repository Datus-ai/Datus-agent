# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Bind the existing ScopedContext to an isolated configurable-agent execution."""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterator

from wcmatch import glob

from datus.configuration.scoped_context_overrides import effective_subagent
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.schemas.agent_spec import AgentSpec
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool
from datus.tools.func_tool.fs_path_policy import PathAllowlist, PathZone, classify_path
from datus.tools.func_tool.semantic_model_filesystem_tools import SemanticModelFilesystemTools
from datus.tools.registry.contracts import tool_contract
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.reference_paths import split_reference_path
from datus.utils.sql_utils import parse_table_name_parts


def legacy_scope_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Project common lists into the configuration consumed by existing scoped tools."""
    result = dict(raw)
    for key in ("tools", "skills"):
        if isinstance(result.get(key), list):
            result[key] = ",".join(result[key])
    if isinstance(result.get("mcp"), list):
        # MCP instances are assembled by the common node, not legacy tool factories.
        result["mcp"] = ""
    return result


class ConfigurableScope:
    """Use explicit data/file declarations while retaining existing fine-grained filters."""

    def __init__(self, spec: AgentSpec, caller_config: Any):
        self.name = spec.name
        self.host_read_roots: tuple[Path, ...] = ()
        self.declared = (spec.scoped_context or ScopedContext()).model_copy(deep=True)
        self.root = Path(caller_config.project_root).resolve()
        self.config = copy(caller_config)
        self.config.services = copy(caller_config.services)
        self._available = caller_config.services.datasources
        requested = self._sources(self.declared)
        unknown = set(requested) - set(self._available)
        if unknown:
            self._invalid(f"Datasources are unavailable in this project: {sorted(unknown)}")
        raw = caller_config.sub_agent_config(spec.name) or {}
        raw = legacy_scope_config(raw)
        inherited = ScopedContext.model_validate(raw.get("scoped_context") or {})
        if inherited.tables_by_datasource is not None or inherited.datasource:
            requested = [name for name in requested if name in self._sources(inherited)]
        self._base_sources = tuple(requested)
        self._base = inherited.model_copy(deep=True)
        for field in ("tables", "metrics", "sqls"):
            selected = getattr(self.declared, field)
            if selected is not None:
                setattr(self._base, field, self._intersect_filter(getattr(inherited, field), selected, field))
        if self.declared.tables_by_datasource is not None or inherited.tables_by_datasource is not None:
            self._base.tables_by_datasource = {
                name: self._intersect_filter(
                    (inherited.tables_by_datasource or {}).get(name, inherited.tables),
                    (self.declared.tables_by_datasource or {}).get(name, self.declared.tables),
                    "tables",
                    name,
                )
                or "*"
                for name in requested
            }
        self._base.datasource = self.declared.datasource or caller_config.current_datasource if requested else ""
        if requested and self._base.datasource not in requested:
            self._invalid("Declare scoped_context.datasource within the allowed datasource set.")
        self._base.project_files = self.declared.project_files
        self._file_filters = [self.declared.project_files or []]
        if inherited.project_files is not None:
            self._file_filters.append(inherited.project_files)
        self._caller_datasource = caller_config.current_datasource
        self._caller_runtime_context = getattr(caller_config, "_runtime_db_context", None)
        raw.update(tools=",".join(spec.tools), skills=",".join(spec.skills), mcp="")
        self.config.agentic_nodes = dict(caller_config.agentic_nodes)
        self.config.agentic_nodes[spec.name] = raw
        # New agents always retain the caller profile, including unattended runs.
        self.config.workflow_permission_profile = (
            getattr(caller_config, "workflow_permission_profile", None)
            or getattr(caller_config, "active_profile_name", None)
            or "normal"
        )
        self.config.restrict_tool_visibility = True
        self._validate_patterns(self.declared.project_files or [])
        self.prepare(None)

    @staticmethod
    def _invalid(message: str) -> None:
        raise DatusException(ErrorCode.COMMON_CONFIG_ERROR, message=message)

    @property
    def declared_datasources(self) -> tuple[str, ...]:
        return self._base_sources

    @staticmethod
    def _sources(scope: ScopedContext) -> list[str]:
        if scope.tables_by_datasource is not None:
            if any(not ConfigurableScope._tokens(pattern) for pattern in scope.tables_by_datasource.values()):
                ConfigurableScope._invalid("Each scoped_context.tables_by_datasource entry requires a table selection.")
            return list(scope.tables_by_datasource)
        return [scope.datasource] if scope.datasource else []

    @staticmethod
    def _tokens(value: str | None) -> list[str]:
        return [token.strip() for token in (value or "").replace("\n", ",").split(",") if token.strip()]

    def _narrow_filter(self, parent: str | None, selected: str | None, field: str, source: str = "") -> str | None:
        if selected is None:
            return parent
        allowed, requested = self._tokens(parent), self._tokens(selected)
        if not allowed:
            return selected
        if not requested:
            self._invalid(f"Runtime scoped_context.{field} cannot clear a declared filter.")
        for token in requested:
            if not any(self._filter_within(pattern, token, field, source) for pattern in allowed):
                self._invalid(f"Runtime scoped_context.{field} must narrow the declared patterns.")
        return selected

    def _filter_within(self, parent: str, child: str, field: str, source: str = "") -> bool:
        if child == parent or parent == "*":
            return True
        if field == "tables":
            dialect = getattr(self._available.get(source or self.declared.datasource), "type", "")
            parts = parse_table_name_parts(child, dialect)
            return all(
                not bound
                or bound in {"*", "%"}
                or bound == parts.get(key)
                or (
                    parts.get(key)
                    and not any(c in parts[key] for c in "*%?[")
                    and fnmatchcase(parts[key], bound.replace("%", "*"))
                )
                for key, bound in parse_table_name_parts(parent, dialect).items()
            )
        parts = split_reference_path(child.replace("/", "."))
        bounds = split_reference_path(parent.replace("/", "."))
        return (
            not any(any(c in part for c in "*%?[") for part in parts)
            and len(parts) >= len(bounds)
            and all(fnmatchcase(value, pattern) for value, pattern in zip(parts, bounds))
        )

    def _intersect_filter(self, parent: str | None, selected: str | None, field: str, source: str = "") -> str | None:
        """Retain only representable patterns allowed by both caller and declaration."""
        allowed, requested = self._tokens(parent), self._tokens(selected)
        if not allowed:
            return selected
        if not requested:
            return parent
        common = []
        for bound in allowed:
            for token in requested:
                if self._filter_within(bound, token, field, source):
                    common.append(token)
                elif self._filter_within(token, bound, field, source):
                    common.append(bound)
        if not common:
            self._invalid(f"Declared and caller scoped_context.{field} do not overlap.")
        return ",".join(dict.fromkeys(common))

    def _validate_patterns(self, patterns: list[str]) -> None:
        for pattern in patterns:
            path = Path(pattern)
            if path.is_absolute() or ".." in path.parts or "\\" in pattern:
                self._invalid("Project file selections must be relative paths or globs within the project.")

    def prepare(self, scope: ScopedContext | None, user_input: Any = None) -> None:
        """Apply partial per-turn narrowing; omitted fields keep the static bounds."""
        self.effective = self._base.model_copy(deep=True)
        if scope is not None:
            for field in ("tables", "metrics", "sqls"):
                if field in scope.model_fields_set:
                    setattr(
                        self.effective,
                        field,
                        self._narrow_filter(getattr(self._base, field), getattr(scope, field), field),
                    )
            if "tables" in scope.model_fields_set and self.effective.tables_by_datasource is not None:
                self.effective.tables_by_datasource = {
                    name: self._narrow_filter(pattern, scope.tables, "tables", name)
                    for name, pattern in self.effective.tables_by_datasource.items()
                    if scope.tables_by_datasource is None or name in scope.tables_by_datasource
                }
            if scope.tables_by_datasource is not None:
                if not set(scope.tables_by_datasource).issubset(self._base_sources):
                    self._invalid("Runtime scoped_context exceeds the declared datasource set.")
                self.effective.tables_by_datasource = {
                    name: self._narrow_filter(
                        (self.effective.tables_by_datasource or {}).get(name, self.effective.tables),
                        pattern,
                        "tables",
                        name,
                    )
                    for name, pattern in scope.tables_by_datasource.items()
                }
            if scope.datasource:
                if scope.datasource not in self._base_sources:
                    self._invalid("Runtime scoped_context.datasource exceeds the declared datasource set.")
                self.effective.datasource = scope.datasource
            if scope.project_files is not None:
                self._validate_patterns(scope.project_files)
                self.effective.project_files = scope.project_files
        selected = self._sources(self.effective)
        if not selected:
            self.effective.datasource = ""
        elif self.effective.datasource not in selected:
            self._invalid("The default datasource must be included in scoped_context.tables_by_datasource.")
        self.datasources = tuple(name for name in selected if name in self._base_sources)
        self.config.services.datasources = {name: self._available[name] for name in self.datasources}
        self.config._current_datasource = self.effective.datasource or ""
        self.config.db_type = getattr(self._available.get(self.config.current_datasource), "type", "")
        self.config._runtime_db_context = (
            self._caller_runtime_context if self.config.current_datasource == self._caller_datasource else None
        )
        if self.config.current_datasource and self.config.current_datasource == self._caller_datasource and user_input:
            database = getattr(user_input, "database", None)
            coordinates = {
                "catalog": getattr(user_input, "catalog", None),
                "database": database
                if database != self._caller_datasource
                or getattr(user_input, "database_is_physical", False)
                or database in self.config.list_databases(self._caller_datasource)
                else None,
                "schema": getattr(user_input, "db_schema", None),
            }
            if any(coordinates.values()):
                self.config.set_runtime_db_context(
                    self.config.runtime_db_context(),
                    datasource=self.config.current_datasource,
                    **{key: value for key, value in coordinates.items() if value},
                )
        self.config.agentic_nodes[self.name]["scoped_context"] = self.effective.model_dump(exclude_none=True)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Reuse the existing ContextVar scope mechanism for actual tool/RAG calls."""
        config = SubAgentConfig.model_validate(self.config.agentic_nodes[self.name])
        with effective_subagent(self.name, config):
            yield

    def allows_file(self, path: Path) -> bool:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            return False
        relative = resolved.relative_to(self.root).as_posix()
        filters = [*self._file_filters, self.effective.project_files or []]
        return all(
            any(glob.globmatch(relative, pattern, flags=glob.GLOBSTAR | glob.DOTGLOB) for pattern in patterns)
            for patterns in filters
        )

    def allows_host_read(self, path: Path) -> bool:
        """Read-only host-authored specifications, separate from business resources."""
        resolved = path.resolve()
        return any(resolved.is_relative_to(root.resolve()) for root in self.host_read_roots)

    def require_tool_scope(self, tool_id: str, arguments: dict[str, Any]) -> None:
        """Check per-turn restrictions before existing tool-level table/policy checks."""
        category, method = tool_id.split(".", 1)
        if arguments and category == "filesystem_tools":
            path = (self.root / arguments.get("path", ".")).resolve()
            if method in {"glob", "grep"}:
                if not path.is_relative_to(self.root) or not self.effective.project_files:
                    self._invalid("File search is outside scoped_context.project_files.")
            elif not self.allows_file(path) and not (
                method in {"read_file", "read_image"} and self.allows_host_read(path)
            ):
                self._invalid("File operation is outside scoped_context.project_files.")
        if category in {"db_tools", "context_search_tools", "semantic_tools", "reference_template_tools"}:
            allowed = set(self.datasources)
            datasource = arguments.get("datasource") or self.config.current_datasource
            targets = [value for key, value in arguments.items() if key.endswith("datasource") and value]
            if not allowed or datasource not in allowed or any(target not in allowed for target in targets):
                self._invalid("This tool call requires a permitted scoped_context.datasource or tables_by_datasource.")
        if arguments and category == "db_tools" and method in {"load_file_as_table", "import_database_file"}:
            path = arguments.get("file_path") or arguments.get("path") or ""
            if not path or not self.allows_file(self.root / path):
                self._invalid("File import is outside scoped_context.project_files.")
        if category in {"db_tools", "reference_template_tools"}:
            from datus.utils.sql_utils import looks_like_sql_file_ref

            sql = arguments.get("sql", "")
            if (
                isinstance(sql, str)
                and looks_like_sql_file_ref(sql.lower() if method == "export_query" else sql)
                and (not self.allows_file(self.root / sql.strip()))
            ):
                self._invalid("SQL file is outside scoped_context.project_files.")
        if arguments and category == "db_tools" and method == "export_query":
            output = arguments.get("output_file")
            if not output or not self.allows_file(self.root / output):
                self._invalid("Exports require an explicit output_file within scoped_context.project_files.")
            if not self.allows_file(self.root / f"{output}.meta.json"):
                self._invalid("The export metadata file is outside scoped_context.project_files.")


class ScopedAgentFilesystem(FilesystemFuncTool):
    """Keep existing path policy and additionally restrict reads, writes and searches."""

    def __init__(self, *, execution_scope: ConfigurableScope, **kwargs: Any):
        self.execution_scope = execution_scope
        super().__init__(**kwargs)

    def _classify(self, path: str):
        resolved = classify_path(
            path,
            root_path=self._root_resolved,
            current_node=self._current_node,
            datus_home=self._datus_home,
            session_data_dir=self._session_data_dir,
            allowlist=self.path_allowlist,
        )
        if self.execution_scope.allows_host_read(resolved.resolved):
            return replace(resolved, zone=PathZone.WHITELIST, read_only=True)
        if resolved.resolved.is_dir() and resolved.resolved.is_relative_to(self.execution_scope.root):
            return resolved
        if not self.execution_scope.allows_file(resolved.resolved):
            return replace(resolved, zone=PathZone.HIDDEN)
        return super()._classify(path)

    def _walk_files(self, seed, include_pattern: str = ""):
        for path in super()._walk_files(seed, include_pattern):
            if self.execution_scope.allows_file(path):
                yield path


def _is_draft_artifact(path: Path, root: Path) -> bool:
    if not path.is_relative_to(root) or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        return False
    relative = path.relative_to(root)
    return relative.parent == Path("subject/ontology") or (
        relative.is_relative_to("subject/semantic_models") and "metrics" not in relative.parts[3:-1]
    )


class _SemanticReferenceFilesystem(ScopedAgentFilesystem):
    """Read authorized live references without exposing stale business models."""

    def _classify(self, path: str):
        resolved = classify_path(path, root_path=self._root_resolved)
        if _is_draft_artifact(resolved.resolved, self._root_resolved):
            return replace(resolved, zone=PathZone.HIDDEN)
        resolved = super()._classify(path)
        plans = self._root_resolved / ".datus/semantic-model-plans"
        if resolved.resolved.is_relative_to(plans) and self.execution_scope.allows_file(resolved.resolved):
            resolved = replace(resolved, zone=PathZone.WHITELIST)
        return replace(resolved, read_only=True)

    def _walk_files(self, seed, include_pattern: str = ""):
        for path in super()._walk_files(seed, include_pattern):
            if not _is_draft_artifact(path, self._root_resolved):
                yield path
        plans = self._root_resolved / ".datus/semantic-model-plans"
        if plans.is_relative_to(seed.resolved):
            target = plans
        elif seed.resolved.is_relative_to(plans):
            target = seed.resolved
        else:
            return
        # The ordinary project walker deliberately hides .datus internals.
        # Only this host-owned plan directory is a searchable reference root.
        tool = ScopedAgentFilesystem(execution_scope=self.execution_scope, root_path=str(plans), strict=True)
        yield from tool._walk_files(tool._classify(str(target)), include_pattern)


class ScopedSemanticFilesystem(ScopedAgentFilesystem, SemanticModelFilesystemTools):
    """Edit the isolated model while reading authorized live project references."""

    def __init__(self, *, reference_scope: ConfigurableScope | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._references = None
        if reference_scope is not None:

            def reference_allowlist():
                allowlist = self.path_allowlist or PathAllowlist()
                return allowlist.model_copy(
                    update={"read": (*allowlist.read, reference_scope.root / ".datus/semantic-model-plans")}
                )

            self._references = _SemanticReferenceFilesystem(
                execution_scope=reference_scope,
                root_path=str(reference_scope.root),
                current_node=self._current_node,
                datus_home=str(self._datus_home) if self._datus_home is not None else None,
                session_data_dir=self._session_data_dir,
                strict=True,
                path_allowlist_provider=reference_allowlist,
            )

    def _reference_path(self, path: str) -> str:
        resolved = Path(path)
        if resolved.is_absolute() and resolved.is_relative_to(self._root_resolved):
            return str(resolved.relative_to(self._root_resolved))
        return path

    @tool_contract(access="read", direct_call=True)
    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> FuncToolResult:
        """Read draft models or authorized project references, including reviewed plans."""
        if self._references is not None and not _is_draft_artifact(
            (self._root_resolved / path).resolve(), self._root_resolved
        ):
            return self._references.read_file(self._reference_path(path), offset, limit)
        return super().read_file(path, offset, limit)

    @tool_contract(access="read", direct_call=True)
    def read_image(self, path: str) -> list[Any] | FuncToolResult:
        """Read an authorized reference image using the existing image size limits."""
        if self._references is not None:
            return self._references.read_image(self._reference_path(path))
        return super().read_image(path)

    @staticmethod
    def _merge_search(draft: FuncToolResult, reference: FuncToolResult, key: str, limit: int) -> FuncToolResult:
        if not draft.success and not reference.success:
            return draft
        values = []
        truncated = False
        for result in (draft, reference):
            if result.success:
                values.extend(result.result[key])
                truncated |= result.result.get("truncated", False)
        return FuncToolResult(result={key: values[:limit], "truncated": truncated or len(values) > limit})

    @tool_contract(access="read", direct_call=True)
    def glob(self, pattern: str, path: str = ".") -> FuncToolResult:
        """Search draft models and authorized references with the existing result limit."""
        result = super().glob(pattern, path)
        if self._references is None:
            return result
        reference = self._references.glob(pattern, self._reference_path(path))
        return self._merge_search(result, reference, "files", 200)

    @tool_contract(access="read", direct_call=True)
    def grep(self, pattern: str, path: str = ".", include: str = "", case_sensitive: bool = True) -> FuncToolResult:
        """Search draft contents and authorized live references without duplicate models."""
        result = super().grep(pattern, path, include, case_sensitive)
        if self._references is None:
            return result
        reference = self._references.grep(pattern, self._reference_path(path), include, case_sensitive)
        return self._merge_search(result, reference, "matches", 100)
