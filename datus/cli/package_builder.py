# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Pure logic behind ``datus package`` — export a project as a self-contained zip.

Design contract (see ``DatusPackage-review.md``):

* **Self-contained**: the generated ``conf/agent.yml`` pins ``home: .`` and a
  fixed ``project_name`` so the unzipped directory is the entire runtime —
  the receiver's ``~/.datus`` is never touched.
* **Zero secrets**: ``conf/agent.yml`` / ``conf/.mcp.json`` are *generated*,
  never copied. Every secret-bearing field is overwritten with a ``${VAR}``
  placeholder (schema-driven — the raw YAML may hold plaintext and values
  alone cannot tell). A final content scan over the staged manifest fails
  the build on any real-looking secret; there is no bypass flag, but the
  scan is a safety net rather than an exhaustive guarantee — binary-sniffed
  files are skipped and reads are capped at ``_SCAN_READ_CAP_BYTES``.
* **Sources, not indexes**: metric/semantic YAML sources ship with a
  generated ``scripts/rebuild_kb.sh``; binary LanceDB indexes never do.

This module deliberately imports no Rich / prompt_toolkit — the interactive
surface lives in ``package_cli.py`` (same split as ``plugin_pack.py`` /
``plugin_cli.py``). ``build_package`` never raises; failures come back as
``PackageResult(ok=False, error=...)``.

Note for future config sections: when a new ``agent.yml`` section carries a
credential, add its path to ``_sanitize_agent_tree`` below. Unknown sections
pass through the generated YAML untouched, so a forgotten entry is only
caught by the final content scan (which fails the build rather than leaks).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

import datus
from datus.api.services.dashboard_service import _DASHBOARD_ARTIFACT_DIRS
from datus.api.services.report_service import _REPORT_ARTIFACT_DIRS
from datus.cli.upgrade_service import DatusPackage, enumerate_datus_packages
from datus.configuration.agent_config import _normalize_project_name, _validate_project_name
from datus.configuration.agent_config_loader import parse_config_path
from datus.configuration.project_config import PROJECT_CONFIG_REL, load_project_override
from datus.schemas.artifact_manifest import ARTIFACT_SLUG_RE
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import DatusPathManager

logger = get_logger(__name__)

PACKAGE_FORMAT = "datus-project-package"
# Deliberately still 1 while the format is unreleased. ``env_vars`` changed
# shape here -- one object per variable (``var`` / ``config_paths`` /
# ``preexisting``) instead of a flat list of names -- which would be breaking
# for anything doing ``sorted(manifest["env_vars"])``. Nothing reads it: no
# reader exists in this repo, and no package has been published for the format
# to be compatible with. Bump on the first change made after a package exists in
# the wild.
PACKAGE_FORMAT_VERSION = 1
PACKAGE_MANIFEST_NAME = "package_manifest.json"

# Names whose top-level directories are runtime state under ``home: .`` on the
# receiver — never packaged. ``output*`` is prefix-matched.
_TOP_LEVEL_EXCLUDED_DIRS = frozenset(
    {"sessions", "data", "logs", "run", "cache", "save", "trajectory", ".venv", ".git", ".datus"}
)
# Directories owned by the component selectors — removed from the generic walk
# so the default full-tree include cannot bypass an explicit selection.
_SELECTOR_OWNED_TOP_DIRS = frozenset({"reports", "dashboards", "template"})
# Files that live at the ``home`` root and are runtime state under ``home: .``:
# ``history`` is the REPL command history — user activity, possibly sensitive
# queries. Top-level only, so a project's own ``docs/history`` still ships.
_TOP_LEVEL_EXCLUDED_FILES = frozenset({"history"})
# Excluded at any depth. ``__MACOSX`` is Archive-Utility litter from a prior
# unzip; ``.Spotlight-V100``/``.Trashes``/``.fseventsd``/… appear when the
# project sits at the root of an external volume.
_ANY_DEPTH_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        ".venv",
        ".git",
        "__MACOSX",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
        ".TemporaryItems",
        ".DocumentRevisions-V100",
    }
)
_ANY_DEPTH_EXCLUDED_FILES = frozenset({".env", ".DS_Store"})
# ``*.sw?``/``*~`` are editor swap/backup files — transient by nature, they
# routinely vanish between collection and zip write.
_EXCLUDED_FILE_SUFFIXES = (".duckdb.wal", ".swp", ".swo", ".swx", "~")


def _is_junk_path(path: Path) -> bool:
    """OS/editor litter that must never ship, wherever it is found.

    Applied by the generic walk AND by every component selector — the
    selectors rglob their own subtrees and bypass the walk's pruning, so a
    macOS AppleDouble sidecar (``._orders.yml`` written next to real files
    on SMB/FAT volumes to carry xattrs) would otherwise be staged as a
    semantic-model/render file and even end up in ``rebuild_kb.sh``.
    """
    name = path.name
    return (
        name in _ANY_DEPTH_EXCLUDED_FILES
        or name.endswith(_EXCLUDED_FILE_SUFFIXES)
        or name.startswith("._")
        or any(part in _ANY_DEPTH_EXCLUDED_DIRS for part in path.parts)
    )


# Generated files replace these — never copy the originals.
_GENERATED_CONF_RELPATHS = frozenset({"conf/agent.yml", "conf/.mcp.json"})

_LARGE_FILE_WARN_BYTES = 100 * 1024 * 1024
_SCAN_READ_CAP_BYTES = 5 * 1024 * 1024
_BINARY_SNIFF_BYTES = 8192

_DIST_CSS_NAME = "index.css"
_DIST_JS_NAME = "index.umd.js"

# ``${VAR}`` / ``${VAR:-default}`` — the whole-value placeholder shape that
# ``resolve_env`` (agent_config.py) expands at load time.
_PLACEHOLDER_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}$")
# Non-anchored variant for harvesting placeholders embedded inside larger
# strings (URIs, header values, …).
_PLACEHOLDER_ANY_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}")

# Key names treated as secret-bearing wherever they appear in config trees.
# Base list mirrors ``observability.config.RedactConfig`` and is extended for
# packaging (see DatusPackage-review.md §5).
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "token",
        "secret",
        "private_key",
        "private_key_file_pwd",
        "app_secret",
        "app_token",
        "bot_token",
        "access_key_id",
        "access_key_secret",
        "secret_key",
        "client_secret",
        "auth_token",
    }
)

_SECRET_CONTENT_PATTERNS: Tuple[Tuple[str, "re.Pattern[bytes]"], ...] = (
    ("pem_private_key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # Fernet tokens are base64url starting with the 0x80 version byte.
    ("fernet_token", re.compile(rb"\bgAAAAA[A-Za-z0-9_-]{20,}")),
    (
        "token_prefix",
        re.compile(
            rb"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"
            rb"|xox[bp]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|glpat-[A-Za-z0-9_-]{20,})"
        ),
    ),
)

# SQLAlchemy-style ``scheme://user:password@host/...`` — rewrite only the
# password component.
_URI_CREDENTIAL_RE = re.compile(r"^(?P<prefix>[A-Za-z0-9+]+://[^:/@]+:)(?P<pwd>[^@]+)(?P<suffix>@.*)$")


# --------------------------------------------------------------------------- #
# Result / option dataclasses                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PackageOptions:
    """Everything the wizard (or ``--yes`` defaults) collected."""

    root: Path
    output: Optional[Path] = None
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    # ``None`` means "all"; an explicit (possibly empty) tuple means exactly these.
    subagents: Optional[Tuple[str, ...]] = None
    skills: Optional[Tuple[str, ...]] = None
    metrics: Optional[Tuple[str, ...]] = None
    # Subject-tree paths (one or two levels) gating metric docs and
    # reference-SQL summaries. ``None`` = every subject area.
    subjects: Optional[Tuple[str, ...]] = None
    plugins: Optional[Tuple[str, ...]] = None
    reports: Optional[Tuple[str, ...]] = None
    dashboards: Optional[Tuple[str, ...]] = None
    report_dist: Optional[Path] = None
    # Non-interactive marker (--yes). No prompt currently reads it — editable
    # installs only warn — but it records how the options were collected.
    assume_yes: bool = False


@dataclass
class EnvVarBinding:
    """One env var the receiver must export, harvested or generated."""

    var: str
    config_path: str
    preexisting: bool


@dataclass(frozen=True)
class EnvVarRequirement:
    """One env var the receiver must bind, with every config field that uses it.

    The per-variable view of :class:`EnvVarBinding`, which is recorded once per
    ``(var, config_path)`` pair. Both the README table and the package manifest
    consume this shape; see :func:`group_env_vars`.
    """

    var: str
    config_paths: List[str]
    preexisting: bool

    def as_manifest_record(self) -> Dict[str, Any]:
        """Serialize for ``package_manifest.json`` (``env_vars`` entries)."""
        return {"var": self.var, "config_paths": list(self.config_paths), "preexisting": self.preexisting}


@dataclass
class SecretFinding:
    arcname: str
    locator: str
    kind: str


@dataclass
class StagedEntry:
    """One zip member: a disk file XOR generated in-memory content."""

    arcname: str
    source: Optional[Path] = None
    content: Optional[bytes] = None
    executable: bool = False

    def read_bytes(self, cap: Optional[int] = None) -> bytes:
        if self.content is not None:
            return self.content if cap is None else self.content[:cap]
        assert self.source is not None
        with open(self.source, "rb") as fh:
            return fh.read() if cap is None else fh.read(cap)

    def size(self) -> int:
        if self.content is not None:
            return len(self.content)
        assert self.source is not None
        return self.source.stat().st_size


@dataclass
class PackageResult:
    ok: bool
    zip_path: str = ""
    file_count: int = 0
    total_bytes: int = 0
    env_vars: List[EnvVarBinding] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    secret_findings: List[SecretFinding] = field(default_factory=list)
    # What the selection actually resolved to — mirrors package_manifest.json
    # so the CLI can report it without reopening the zip.
    selections: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PackageError(DatusException):
    """Internal failure signal; converted to ``PackageResult`` at the boundary.

    A coded :class:`DatusException` (``PACKAGE_BUILD_ERROR``) so anything
    escaping the boundary keeps the repository's ``error_code=…`` contract,
    while remaining catchable as ``PackageError`` for the builder's own
    recoverable control flow (same shape as ``plugins.store.StoreError``).
    """

    def __init__(self, message: str):
        super().__init__(ErrorCode.PACKAGE_BUILD_ERROR, message=message)


# --------------------------------------------------------------------------- #
# Raw config access + enumeration helpers (shared with the wizard)            #
# --------------------------------------------------------------------------- #


def load_raw_agent_config() -> Optional[Dict[str, Any]]:
    """Read the *unexpanded* ``agent:`` dict straight from the YAML file.

    A fresh read (not the process-global ``configuration_manager()``) so no
    in-memory mutation from a previously constructed ``AgentConfig`` can
    leak into the generated file. Returns ``None`` when no config exists.
    """
    try:
        path = parse_config_path("")
    except DatusException:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("package: cannot read agent config %s: %s", path, exc)
        return None
    agent = data.get("agent")
    return agent if isinstance(agent, dict) else None


def resolve_effective_project_name(root: Path, raw: Dict[str, Any]) -> str:
    """Project name to pin into the package: override > agent.yml > CWD-derived."""
    override = load_project_override(cwd=str(root))
    if override is not None and override.project_name:
        return override.project_name
    raw_name = raw.get("project_name")
    if isinstance(raw_name, str) and raw_name.strip():
        try:
            # Private-by-convention validator; packaging must apply the same
            # shape rule the loader does or the receiver diverges.
            return _validate_project_name(raw_name.strip())
        except DatusException:
            logger.warning("package: agent.yml project_name %r invalid; deriving from CWD", raw_name)
    return _normalize_project_name(str(root))


def resolve_source_home(raw: Dict[str, Any], root: Path) -> Path:
    """The *source* project's ``home`` (where templates live)."""
    home = raw.get("home")
    if isinstance(home, str) and home.strip():
        expanded = Path(home).expanduser()
        if not expanded.is_absolute():
            expanded = (root / expanded).resolve()
        return expanded
    return DatusPathManager.resolve_home(None)


def list_subagents(raw: Dict[str, Any]) -> Dict[str, str]:
    """``{name: description}`` for every ``agentic_nodes`` entry."""
    nodes = raw.get("agentic_nodes")
    out: Dict[str, str] = {}
    if isinstance(nodes, dict):
        for name, entry in nodes.items():
            desc = entry.get("agent_description") if isinstance(entry, dict) else ""
            out[str(name)] = str(desc or "")
    return out


def list_skills(root: Path) -> Dict[str, Path]:
    """``{skill_name: directory}`` — project ``./.datus/skills`` ∪ global
    ``~/.datus/skills``; project wins on a name clash. Global skills are
    materialized into the zip because ``~/.datus/skills`` is a hardcoded
    path that does NOT follow ``home: .`` on the receiver."""
    out: Dict[str, Path] = {}
    for base in (Path.home() / ".datus" / "skills", root / ".datus" / "skills"):
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                out[entry.name] = entry
    return out


_SQL_SUMMARIES_REL = "subject/sql_summaries"
# ``tags: ["subject_tree: a/b/c", ...]`` — how a metric doc records its
# subject assignment (see datus/storage/metric/subject_path.py).
_SUBJECT_TAG_PREFIX = "subject_tree:"


SUBJECT_MENU_MAX_DEPTH = 2


def _subject_segments(path_expr: Any) -> List[str]:
    """``"a/b/c"`` → ``["a", "b", "c"]``."""
    return [part.strip() for part in str(path_expr or "").split("/") if part.strip()]


def _subject_prefixes(path_expr: Any, max_depth: int = SUBJECT_MENU_MAX_DEPTH) -> List[str]:
    """Selectable ancestors of a subject path, shallowest first.

    ``运营/活动/SR`` with ``max_depth=2`` yields ``["运营", "运营/活动"]`` — the
    paths a user can pick in order to include that entry. The menu stops at
    two levels: deeper trees produce a screen too long to scan, and a
    second-level node is already a coherent subject area.
    """
    segments = _subject_segments(path_expr)[:max_depth]
    return ["/".join(segments[: depth + 1]) for depth in range(len(segments))]


def _subject_matches(path_expr: Any, selected: Set[str]) -> bool:
    """True when any ancestor of ``path_expr`` was selected.

    Picking a parent takes its whole subtree, so a depth-3 entry is kept by
    a depth-1 or depth-2 selection.
    """
    segments = _subject_segments(path_expr)
    return any("/".join(segments[: depth + 1]) in selected for depth in range(len(segments)))


def _count_by_subject_prefix(values: Sequence[Any]) -> Dict[str, int]:
    """``{selectable_path: entries beneath it}`` — a parent counts its subtree."""
    counts: Dict[str, int] = {}
    for value in values:
        for prefix in _subject_prefixes(value):
            counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def _summary_subject_values(root: Path) -> List[str]:
    """``subject_tree`` of every committed reference-SQL summary."""
    base = root / _SQL_SUMMARIES_REL
    if not base.is_dir():
        return []
    return [str(_read_yaml_mapping(path).get("subject_tree") or "") for path in sorted(base.rglob("*.y*ml"))]


def _metric_subject_values(root: Path) -> List[str]:
    """``subject_tree`` tag of every metric document in the project."""
    values: List[str] = []
    for path in _metric_yaml_files(root):
        values.extend(_metric_doc_subjects(path))
    return values


def _metric_yaml_files(root: Path) -> List[Path]:
    base = root / "subject" / "semantic_models"
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.y*ml") if p.is_file() and "metrics" in p.relative_to(base).parts[:-1])


def _metric_subject_path(doc: Any) -> str:
    """Full subject path of one ``metric:`` document, or ``""`` when untagged.

    ``gen_metrics`` writes the tag under ``metric.locked_metadata.tags``;
    ``metric.tags`` is accepted too since hand-authored files use the
    shorter form.
    """
    metric = doc.get("metric") if isinstance(doc, dict) else None
    if not isinstance(metric, dict):
        return ""
    locked = metric.get("locked_metadata")
    tag_lists = [metric.get("tags"), locked.get("tags") if isinstance(locked, dict) else None]
    for tags in tag_lists:
        for tag in tags or []:
            if isinstance(tag, str) and tag.strip().startswith(_SUBJECT_TAG_PREFIX):
                path = "/".join(_subject_segments(tag.split(_SUBJECT_TAG_PREFIX, 1)[1]))
                if path:
                    return path
    return ""


def _split_yaml_documents(text: str) -> List[str]:
    """Split a multi-document YAML file on ``---``, keeping each chunk verbatim.

    Text-level so the packaged file keeps its original formatting, comments
    and key order — a reserialize through PyYAML would rewrite all three.
    """
    chunks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            chunks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    chunks.append("\n".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def _metric_doc_subjects(path: Path) -> List[str]:
    """Subject paths tagged on the metric documents inside one YAML file."""
    roots: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            docs = list(yaml.safe_load_all(fh))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("package: unreadable metric yaml %s: %s", path, exc)
        return roots
    for doc in docs:
        subject_path = _metric_subject_path(doc)
        if subject_path:
            roots.append(subject_path)
    return roots


def filter_metric_yaml(path: Path, selected_subjects: Sequence[str]) -> Optional[bytes]:
    """Keep only the metric documents under the selected subject roots.

    A metric file holds one document per metric, and a table's metrics
    routinely span several subject areas (baisheng: 22 metrics across 4).
    Filtering whole files would therefore be all-or-nothing, which is what
    "selecting a subject" is supposed to avoid. Returns ``None`` when every
    document is kept, so the caller can ship the file untouched.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("package: unreadable metric yaml %s: %s", path, exc)
        return None

    wanted = set(selected_subjects)
    chunks = _split_yaml_documents(text)
    kept: List[str] = []
    for chunk in chunks:
        try:
            doc = yaml.safe_load(chunk)
        except yaml.YAMLError:
            kept.append(chunk)  # unparseable: keep rather than silently drop
            continue
        subject_path = _metric_subject_path(doc)
        # Untagged metrics belong to no subject and would match no selection;
        # keep them rather than dropping them from every filtered package.
        if not subject_path or _subject_matches(subject_path, wanted):
            kept.append(chunk)
    if len(kept) == len(chunks):
        return None
    return ("\n---\n".join(kept) + "\n").encode("utf-8")


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("package: unreadable yaml %s: %s", path, exc)
        return {}
    return doc if isinstance(doc, dict) else {}


def _vector_db_subject_roots(root: Path, raw: Dict[str, Any], project_name: str) -> Dict[str, str]:
    """``{root_name: description}`` read from the project's subject tree store.

    The subject tree is the KB's own registry of subject areas (one tree per
    datasource under ``{home}/data/{project}/``), so it — not the artifacts —
    is the authoritative menu. Reading it needs the source project's path
    manager, which is installed and restored around the read. Any failure
    (KB never built, backend unavailable) degrades to an empty mapping and
    the caller falls back to artifact-derived roots.
    """
    from datus.utils.path_manager import get_path_manager, set_current_path_manager

    datasources = list(((raw.get("services") or {}).get("datasources")) or {})
    if not datasources:
        return {}
    try:
        previous = get_path_manager()
    except Exception:  # pragma: no cover - no context installed yet
        previous = None

    roots: Dict[str, str] = {}
    try:
        set_current_path_manager(
            DatusPathManager(datus_home=resolve_source_home(raw, root), project_name=project_name, project_root=root)
        )
        from datus.storage.subject_tree.store import SubjectTreeStore

        for datasource in datasources:
            try:
                store = SubjectTreeStore(project=project_name, datasource_id=str(datasource))
                for node in store.get_children(None) or []:
                    name = str(node.get("name") or "").strip()
                    if not name:
                        continue
                    roots.setdefault(name, str(node.get("description") or ""))
                    # Second level: a root alone is usually too coarse to be a
                    # useful selection (baisheng keeps all 22 metrics under one).
                    for child in store.get_children(node.get("node_id")) or []:
                        child_name = str(child.get("name") or "").strip()
                        if child_name:
                            roots.setdefault(f"{name}/{child_name}", str(child.get("description") or ""))
            except Exception as exc:
                logger.debug("package: subject tree unavailable for %s: %s", datasource, exc)
    except Exception as exc:
        logger.warning("package: cannot read the subject tree from the vector store: %s", exc)
    finally:
        if previous is not None:
            set_current_path_manager(previous)
    return roots


def list_subject_paths(root: Path, raw: Dict[str, Any], project_name: str) -> Dict[str, str]:
    """``{subject_path: label}`` — the selectable subject areas, two levels deep.

    Rendered as a tree: a root line followed by its indented children. Picking
    a root takes its whole subtree, picking a child narrows to that branch.
    The menu stops at :data:`SUBJECT_MENU_MAX_DEPTH` because deeper trees make
    the screen unscannable while a second-level node is already a coherent
    area (``运营/活动``, ``数据分析/指标统计``).

    The vector store's subject tree is the authoritative menu; counts come
    from the artifacts that would actually travel, so a label says what
    picking that node costs. Paths present only in the artifacts (KB never
    built on this machine) are still offered.
    """
    tree_nodes = _vector_db_subject_roots(root, raw, project_name)
    sql_counts = _count_by_subject_prefix(_summary_subject_values(root))
    metric_counts = _count_by_subject_prefix(_metric_subject_values(root))

    paths = {p for p in set(tree_nodes) | set(sql_counts) | set(metric_counts) if p}
    # Parents of any offered child are offered too, so the tree is connected.
    paths |= {path.split("/", 1)[0] for path in paths}

    labels: Dict[str, str] = {}
    for path in sorted(paths, key=lambda value: (value.split("/")[0], value.count("/"), value)):
        depth = path.count("/")
        leaf = path.split("/")[-1]
        parts = []
        if metric_counts.get(path):
            parts.append(f"{metric_counts[path]} metrics")
        if sql_counts.get(path):
            parts.append(f"{sql_counts[path]} reference SQL")
        if not parts:
            parts.append("no packaged entries")
        description = tree_nodes.get(path) or ""
        suffix = f" — {description}" if description else ""
        indent = "  └ " if depth else ""
        labels[path] = f"{indent}{leaf} ({', '.join(parts)}){suffix}"
    return labels


def list_packageable_plugins(root: Path) -> Dict[str, str]:
    """``{plugin_name: label}`` — plugins this project can ship install lines for.

    The candidate set is the union of the project's activation list
    (``.datus/config.yml`` ``plugins:``) and what is actually installed in the
    managed store, so a project that never wrote an activation list can still
    select from its installed plugins.
    """
    labels: Dict[str, str] = {}
    installed = _installed_plugin_versions()
    override = load_project_override(cwd=str(root))
    activated = {name for name, act in (override.plugins or {}).items() if act.enabled} if override else set()

    for name in sorted(activated | set(installed)):
        distribution, version = installed.get(name, (name, ""))
        marks = []
        if name in activated:
            marks.append("activated")
        marks.append(f"{distribution}=={version}" if version else f"{distribution} (version unknown)")
        labels[name] = f"{name} — {', '.join(marks)}"
    return labels


def _installed_plugin_versions() -> Dict[str, Tuple[str, str]]:
    """``{name: (distribution, version)}`` from the managed plugin store."""
    versions: Dict[str, Tuple[str, str]] = {}
    try:
        from datus.plugins import store

        for meta in store.iter_installed():
            name = str(meta.get("name") or "")
            if name:
                versions[name] = (str(meta.get("distribution") or name), str(meta.get("version") or ""))
    except Exception as exc:  # pragma: no cover - store errors must not kill packaging
        logger.warning("package: cannot read installed plugin metadata: %s", exc)
    return versions


def resolve_default_datasource(root: Path, raw: Dict[str, Any]) -> Optional[str]:
    """Datasource the reference-SQL rebuild runs against: project pin, then
    the ``default: true`` entry, then a single-datasource shortcut."""
    override = load_project_override(cwd=str(root))
    if override is not None and override.default_datasource:
        return override.default_datasource
    datasources = ((raw.get("services") or {}).get("datasources")) or {}
    if not isinstance(datasources, dict) or not datasources:
        return None
    for name, entry in datasources.items():
        if isinstance(entry, dict) and entry.get("default"):
            return str(name)
    return str(next(iter(datasources))) if len(datasources) == 1 else None


def list_metric_datasources(root: Path) -> List[str]:
    base = root / "subject" / "semantic_models"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


# kind → (top-level dir name, per-prefix walker allowlist). The single map
# both slug listing and artifact staging branch on.
_ARTIFACT_KIND_DIRS: Dict[str, Tuple[str, Dict[str, Tuple[Tuple[str, ...], bool]]]] = {
    "report": ("reports", _REPORT_ARTIFACT_DIRS),
    "dashboard": ("dashboards", _DASHBOARD_ARTIFACT_DIRS),
}


def list_artifact_slugs(root: Path, kind: str) -> List[str]:
    """Slugs with a ``manifest.json`` under ``reports/`` / ``dashboards/``."""
    kind_dir, _ = _ARTIFACT_KIND_DIRS[kind]
    base = root / kind_dir
    if not base.is_dir():
        return []
    slugs = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and ARTIFACT_SLUG_RE.fullmatch(entry.name) and (entry / "manifest.json").is_file():
            slugs.append(entry.name)
    return slugs


# --------------------------------------------------------------------------- #
# Step 1 — generic collection & filtering                                     #
# --------------------------------------------------------------------------- #


def _compile_patterns(patterns: Sequence[str], label: str) -> List["re.Pattern[str]"]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise PackageError(f"invalid {label} regex {pattern!r}: {exc}") from exc
    return compiled


def collect_project_files(
    root: Path,
    include: Sequence[str],
    exclude: Sequence[str],
    output_path: Path,
) -> Tuple[List[StagedEntry], List[str]]:
    """Walk the project tree honoring built-in and user filters.

    Matching is ``re.search`` against the POSIX relative path. Built-in
    exclusions always win; the ``.datus`` guard in particular can never be
    bypassed by user patterns.
    """
    include_res = _compile_patterns(include, "--include")
    exclude_res = _compile_patterns(exclude, "--exclude")
    root = root.resolve()
    output_resolved = output_path.resolve()
    entries: List[StagedEntry] = []
    warnings: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        depth_parts = rel_dir.parts

        pruned = []
        for d in list(dirnames):
            child_parts = depth_parts + (d,)
            top = child_parts[0]
            if d in _ANY_DEPTH_EXCLUDED_DIRS:
                pruned.append(d)
            elif len(child_parts) == 1 and (
                top in _TOP_LEVEL_EXCLUDED_DIRS or top in _SELECTOR_OWNED_TOP_DIRS or top.startswith("output")
            ):
                pruned.append(d)
            elif child_parts[:2] in (("subject", "semantic_models"), ("subject", "sql_summaries")):
                # Selector-owned: metric / reference-SQL selection stages
                # these subtrees, filtered by the chosen subject roots.
                pruned.append(d)
            elif (Path(dirpath) / d).is_symlink():
                warnings.append(f"skipped symlinked directory: {(rel_dir / d).as_posix()}")
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)
        dirnames.sort()

        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            rel = (rel_dir / fname).as_posix() if depth_parts else fname
            if _is_junk_path(Path(fname)):
                continue
            if not depth_parts and fname in _TOP_LEVEL_EXCLUDED_FILES:
                continue
            if rel in _GENERATED_CONF_RELPATHS or rel == PROJECT_CONFIG_REL:
                continue
            try:
                if fpath.resolve() == output_resolved:
                    continue
            except OSError:
                continue
            if include_res and not any(p.search(rel) for p in include_res):
                continue
            if any(p.search(rel) for p in exclude_res):
                continue
            if fpath.is_symlink():
                try:
                    resolved = fpath.resolve(strict=True)
                except OSError:
                    warnings.append(f"skipped broken symlink: {rel}")
                    continue
                if not resolved.is_relative_to(root):
                    warnings.append(f"skipped symlink escaping project root: {rel}")
                    continue
                entries.append(StagedEntry(arcname=rel, source=resolved))
                continue
            if not fpath.is_file():
                continue
            entries.append(StagedEntry(arcname=rel, source=fpath))
    return entries, warnings


# --------------------------------------------------------------------------- #
# Step 2 — component selectors                                                #
# --------------------------------------------------------------------------- #


def _resolve_selection(available: Sequence[str], requested: Optional[Sequence[str]], label: str) -> List[str]:
    """``None`` selects everything; explicit names must all exist."""
    if requested is None:
        return list(available)
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise PackageError(f"unknown {label}(s): {', '.join(unknown)}; available: {', '.join(available) or '-'}")
    return list(requested)


def select_subagents(
    raw: Dict[str, Any],
    requested: Optional[Sequence[str]],
    source_home: Path,
) -> Tuple[List[str], List[StagedEntry], List[str]]:
    """Pick agentic_nodes entries and stage their prompt templates."""
    kept = _resolve_selection(list(list_subagents(raw)), requested, "subagent")

    entries: List[StagedEntry] = []
    warnings: List[str] = []
    nodes = raw.get("agentic_nodes") or {}
    template_dir = source_home / "template"
    for name in kept:
        entry = nodes.get(name) or {}
        base = str(entry.get("system_prompt") or name)
        version = str(entry.get("prompt_version") or "1.0")
        template = template_dir / f"{base}_system_{version}.j2"
        if template.is_file():
            entries.append(StagedEntry(arcname=f"template/{template.name}", source=template))
        elif entry.get("system_prompt"):
            # Built-in node names fall back to packaged templates; a custom
            # system_prompt with no template file will fail on the receiver.
            warnings.append(f"subagent {name!r}: template {template.name} not found under {template_dir}")
    return kept, entries, warnings


def select_skills(
    root: Path,
    requested: Optional[Sequence[str]],
) -> Tuple[List[str], List[StagedEntry]]:
    """Stage selected skills under ``.datus/skills/<name>/`` in the zip."""
    available = list_skills(root)
    kept = _resolve_selection(sorted(available), requested, "skill")

    entries: List[StagedEntry] = []
    for name in kept:
        skill_dir = available[name]
        for path in sorted(skill_dir.rglob("*")):
            if path.is_file() and not _is_junk_path(path.relative_to(skill_dir)):
                rel = path.relative_to(skill_dir).as_posix()
                entries.append(StagedEntry(arcname=f".datus/skills/{name}/{rel}", source=path))
    return kept, entries


def select_metrics(
    root: Path,
    requested: Optional[Sequence[str]],
    selected_subjects: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[StagedEntry], Dict[str, Tuple[List[str], List[str]]], List[str]]:
    """Stage ``subject/semantic_models/{ds}/**`` for the selected datasources.

    ``selected_subjects`` (``None`` = no subject filtering) narrows the metric
    documents to the chosen subject-tree roots; semantic-model documents are
    table definitions and always travel with their datasource.

    Returns ``(kept, entries, per_ds, warnings)`` where ``per_ds`` maps
    datasource → ``(semantic_yaml_relpaths, metrics_yaml_relpaths)`` for the
    rebuild script.
    """
    kept = _resolve_selection(list_metric_datasources(root), requested, "metric datasource")

    warnings: List[str] = []
    entries: List[StagedEntry] = []
    per_ds: Dict[str, Tuple[List[str], List[str]]] = {}
    for ds in kept:
        ds_dir = root / "subject" / "semantic_models" / ds
        semantic_files: List[str] = []
        metrics_files: List[str] = []
        for path in sorted(ds_dir.rglob("*")):
            if not path.is_file() or _is_junk_path(path.relative_to(ds_dir)):
                continue
            rel = path.relative_to(root).as_posix()
            is_metric_file = "metrics" in path.relative_to(ds_dir).parts[:-1]
            if is_metric_file and selected_subjects is not None:
                # Metric docs carry their subject as a ``subject_tree:`` tag;
                # a file travels when at least one of its metrics falls under
                # a selected subject root. Semantic-model docs are table
                # definitions, not subject-scoped, so they always travel.
                doc_subjects = _metric_doc_subjects(path)
                if not doc_subjects:
                    # Untagged metrics belong to no subject area and would
                    # match no selection — ship them (with a warning) rather
                    # than dropping them from every filtered package.
                    warnings.append(f"{rel}: no subject_tree tag — packaged regardless of the subject selection")
                elif not any(_subject_matches(s, set(selected_subjects)) for s in doc_subjects):
                    continue
                else:
                    # One file spans several subjects, so drop the documents
                    # outside the selection instead of shipping the lot.
                    filtered = filter_metric_yaml(path, selected_subjects)
                    if filtered is not None:
                        entries.append(StagedEntry(arcname=rel, content=filtered))
                        metrics_files.append(rel)
                        continue
            entries.append(StagedEntry(arcname=rel, source=path))
            if path.suffix.lower() in (".yml", ".yaml"):
                # ``metrics/*_metrics.yml`` feeds --components metrics; the
                # rest are semantic-model documents.
                if is_metric_file:
                    metrics_files.append(rel)
                else:
                    semantic_files.append(rel)
        per_ds[ds] = (semantic_files, metrics_files)
    return kept, entries, per_ds, warnings


def select_reference_sql(root: Path, selected_subjects: Sequence[str]) -> Tuple[List[StagedEntry], int, List[str]]:
    """Stage the summary YAML under the selected subject roots.

    ``subject/sql_summaries/*.yaml`` is what the receiver re-indexes (see
    ``bootstrap-kb --from_summaries``), so the summaries — not the raw
    ``.sql`` corpus — are what the subject selection gates. The raw corpus
    ships as ordinary project content via the generic walk.

    A summary with no ``subject_tree`` belongs to no subject area and would
    match no selection; it ships anyway (with a warning) rather than
    disappearing from every package.
    """
    base = root / _SQL_SUMMARIES_REL
    if not base.is_dir():
        return [], 0, []
    wanted = set(selected_subjects)
    entries: List[StagedEntry] = []
    warnings: List[str] = []
    for path in sorted(base.rglob("*.y*ml")):
        if not path.is_file() or _is_junk_path(path.relative_to(base)):
            continue
        subject_path = str(_read_yaml_mapping(path).get("subject_tree") or "")
        if not _subject_segments(subject_path):
            warnings.append(f"{path.name}: no subject_tree — packaged regardless of the subject selection")
        elif not _subject_matches(subject_path, wanted):
            continue
        entries.append(StagedEntry(arcname=path.relative_to(root).as_posix(), source=path))
    return entries, len(entries), warnings


def _artifact_walk(artifact_dir: Path, dirs_spec: Dict[str, Tuple[Tuple[str, ...], bool]]) -> List[Path]:
    """Allowlist walk mirroring ``report_service._iter_artifact_files``."""
    out: List[Path] = []
    resolved_root = artifact_dir.resolve()
    for sub, (suffixes, recursive) in dirs_spec.items():
        base = artifact_dir / sub
        if not base.is_dir():
            continue
        iterator = base.rglob("*") if recursive else base.iterdir()
        for path in iterator:
            if not path.is_file() or _is_junk_path(path.relative_to(base)):
                continue
            name_lower = path.name.lower()
            if not any(name_lower.endswith(sfx) for sfx in suffixes):
                continue
            try:
                path.resolve().relative_to(resolved_root)
            except ValueError:
                continue  # symlink escaping the artifact — drop
            out.append(path)
    return sorted(out)


def select_artifacts(
    root: Path,
    kind: str,
    requested: Optional[Sequence[str]],
    report_dist: Optional[Path],
) -> Tuple[List[str], List[StagedEntry], List[str]]:
    """Stage ``reports/<slug>/`` / ``dashboards/<slug>/`` subtrees.

    Reuses the canonical per-prefix allowlists from the API services (no
    fourth copy), plus ``manifest.json`` / ``index.html`` / ``_assets``.
    When ``report_dist`` is set, report ``index.html`` files rendered
    against the CDN are rewritten in memory to relative ``_assets/`` URLs
    and the two dist files are staged — the source project is never touched.
    """
    kind_dir, dirs_spec = _ARTIFACT_KIND_DIRS[kind]
    kept = _resolve_selection(list_artifact_slugs(root, kind), requested, f"{kind} slug")

    entries: List[StagedEntry] = []
    warnings: List[str] = []
    for slug in kept:
        artifact_dir = root / kind_dir / slug
        for path in _artifact_walk(artifact_dir, dirs_spec):
            rel = path.relative_to(root).as_posix()
            entries.append(StagedEntry(arcname=rel, source=path))
        manifest = artifact_dir / "manifest.json"
        if manifest.is_file():
            entries.append(StagedEntry(arcname=f"{kind_dir}/{slug}/manifest.json", source=manifest))
        assets_dir = artifact_dir / "_assets"
        staged_assets = False
        for asset_name in (_DIST_CSS_NAME, _DIST_JS_NAME):
            asset = assets_dir / asset_name
            if asset.is_file():
                entries.append(StagedEntry(arcname=f"{kind_dir}/{slug}/_assets/{asset_name}", source=asset))
                staged_assets = True
        index_html = artifact_dir / "index.html"
        if index_html.is_file():
            entries.extend(_stage_index_html(index_html, kind, kind_dir, slug, report_dist, staged_assets, warnings))
    return kept, entries, warnings


def _stage_index_html(
    index_html: Path,
    kind: str,
    kind_dir: str,
    slug: str,
    report_dist: Optional[Path],
    already_offline: bool,
    warnings: List[str],
) -> List[StagedEntry]:
    """Stage index.html, rewriting CDN URLs to ``_assets/`` when a dist is given."""
    arc = f"{kind_dir}/{slug}/index.html"
    if kind != "report" or report_dist is None or already_offline:
        return [StagedEntry(arcname=arc, source=index_html)]
    from datus.agent.node.visual_artifact._artifact_html_renderer import CDN_BUNDLE_CSS, CDN_BUNDLE_JS

    try:
        html = index_html.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.append(f"{arc}: unreadable ({exc}); staged as-is")
        return [StagedEntry(arcname=arc, source=index_html)]
    if CDN_BUNDLE_CSS not in html and CDN_BUNDLE_JS not in html:
        warnings.append(f"{arc}: no CDN bundle URLs found; staged as-is")
        return [StagedEntry(arcname=arc, source=index_html)]
    missing = [name for name in (_DIST_CSS_NAME, _DIST_JS_NAME) if not (report_dist / name).is_file()]
    if missing:
        # The CLI validates the dist dir up front, but the builder API can be
        # handed any path — never rewrite to assets we cannot actually ship.
        warnings.append(f"{arc}: dist {report_dist} is missing {', '.join(missing)}; staged as-is (CDN URLs kept)")
        return [StagedEntry(arcname=arc, source=index_html)]
    html = html.replace(CDN_BUNDLE_CSS, f"_assets/{_DIST_CSS_NAME}").replace(CDN_BUNDLE_JS, f"_assets/{_DIST_JS_NAME}")
    return [
        StagedEntry(arcname=arc, content=html.encode("utf-8")),
        StagedEntry(arcname=f"{kind_dir}/{slug}/_assets/{_DIST_CSS_NAME}", source=report_dist / _DIST_CSS_NAME),
        StagedEntry(arcname=f"{kind_dir}/{slug}/_assets/{_DIST_JS_NAME}", source=report_dist / _DIST_JS_NAME),
    ]


# --------------------------------------------------------------------------- #
# Step 3 — sanitized config generation                                        #
# --------------------------------------------------------------------------- #


def _is_secret_key(key: str) -> bool:
    # Split camelCase before lowering so ``appSecret`` matches ``app_secret``.
    decamel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    normalized = "_".join(part for part in re.split(r"[^a-z0-9]+", decamel.lower()) if part)
    if normalized in _SECRET_KEY_NAMES:
        return True
    padded = f"_{normalized}_"
    return any(f"_{name}_" in padded for name in _SECRET_KEY_NAMES)


def _sanitize_var_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").upper()
    return cleaned or "X"


class _PlaceholderAllocator:
    """Generates deterministic ``${VAR}`` names, dedup-ing by source value."""

    def __init__(self) -> None:
        self.bindings: List[EnvVarBinding] = []
        self._by_value: Dict[str, str] = {}
        self._used: Dict[str, str] = {}
        self._seen: Set[Tuple[str, str]] = set()

    def _record(self, var: str, config_path: str, preexisting: bool) -> None:
        key = (var, config_path)
        if key not in self._seen:
            self._seen.add(key)
            self.bindings.append(EnvVarBinding(var=var, config_path=config_path, preexisting=preexisting))

    def keep_preexisting(self, value: str, config_path: str) -> None:
        match = _PLACEHOLDER_RE.match(value)
        if match and match.group(1) not in self._used:
            # Placeholders we allocated ourselves this run re-surface when a
            # later (recursive) pass revisits the same field — skip those.
            self._record(match.group(1), config_path, preexisting=True)

    def allocate(self, value: str, preferred_var: str, config_path: str) -> str:
        existing = self._by_value.get(value)
        if existing is not None:
            self._record(existing, config_path, preexisting=False)
            return f"${{{existing}}}"
        var = preferred_var
        suffix = 2
        while var in self._used and self._used[var] != value:
            var = f"{preferred_var}_{suffix}"
            suffix += 1
        self._used[var] = value
        self._by_value[value] = var
        self._record(var, config_path, preexisting=False)
        return f"${{{var}}}"

    def harvest(self, tree: Any, config_path: str) -> None:
        """Record every ``${VAR}`` occurrence (whole-value or embedded) as a
        preexisting binding — non-secret fields (host/port/base URLs, …)
        never pass through the secret rewriters, but the receiver still has
        to export them for the config to resolve."""
        if isinstance(tree, dict):
            for key, value in tree.items():
                child = f"{config_path}.{key}" if config_path else str(key)
                self.harvest(value, child)
        elif isinstance(tree, list):
            for idx, item in enumerate(tree):
                self.harvest(item, f"{config_path}[{idx}]")
        elif isinstance(tree, str):
            for var in _PLACEHOLDER_ANY_RE.findall(tree):
                if var not in self._used:
                    self._record(var, config_path, preexisting=True)


def _rewrite_secret_value(
    container: Dict[str, Any],
    key: str,
    preferred_var: str,
    config_path: str,
    alloc: _PlaceholderAllocator,
) -> None:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        return
    if _PLACEHOLDER_RE.match(value):
        alloc.keep_preexisting(value, config_path)
        return
    container[key] = alloc.allocate(value, preferred_var, config_path)


def _rewrite_uri_password(
    container: Dict[str, Any],
    key: str,
    preferred_var: str,
    config_path: str,
    alloc: _PlaceholderAllocator,
) -> None:
    value = container.get(key)
    if not isinstance(value, str) or "@" not in value or "://" not in value:
        return
    match = _URI_CREDENTIAL_RE.match(value)
    if match:
        pwd = match.group("pwd")
        if _PLACEHOLDER_RE.match(pwd):
            alloc.keep_preexisting(pwd, config_path)
            return
        placeholder = alloc.allocate(pwd, preferred_var, config_path)
        container[key] = f"{match.group('prefix')}{placeholder}{match.group('suffix')}"
    elif re.search(r"://[^/@]+:[^@]+@", value):
        # A password component IS present but the URI doesn't parse —
        # placeholder the whole value rather than risk shipping the secret.
        _rewrite_secret_value(container, key, preferred_var, config_path, alloc)
    # else: user-only URIs (``scheme://user@host/db``) carry no password —
    # leave them intact so the receiver keeps host/port/database.


def _rewrite_secret_named_keys(
    tree: Any,
    scope_var_prefix: str,
    config_path: str,
    alloc: _PlaceholderAllocator,
    *,
    all_string_leaves: bool = False,
) -> None:
    """Recursively placeholder secret-named keys (or every string leaf)."""
    if isinstance(tree, dict):
        for key, value in list(tree.items()):
            child_path = f"{config_path}.{key}"
            if isinstance(value, (dict, list)):
                _rewrite_secret_named_keys(
                    value,
                    f"{scope_var_prefix}_{_sanitize_var_component(key)}",
                    child_path,
                    alloc,
                    all_string_leaves=all_string_leaves,
                )
            elif isinstance(value, str) and (all_string_leaves or _is_secret_key(key)):
                _rewrite_secret_value(
                    tree, key, f"{scope_var_prefix}_{_sanitize_var_component(key)}", child_path, alloc
                )
    elif isinstance(tree, list):
        for idx, item in enumerate(tree):
            _rewrite_secret_named_keys(
                item,
                f"{scope_var_prefix}_{idx}",
                f"{config_path}[{idx}]",
                alloc,
                all_string_leaves=all_string_leaves,
            )


def _sanitize_agent_tree(data: Dict[str, Any], alloc: _PlaceholderAllocator, warnings: List[str]) -> None:
    """Walk the known secret paths of the raw ``agent:`` dict, in place.

    Schema-driven on purpose: the raw values may be plaintext or already
    ``${VAR}`` and values alone cannot tell — every known secret path is
    overwritten. New config sections carrying credentials must be added
    here; the final content scan is the safety net for omissions.
    """
    providers = data.get("providers")
    if isinstance(providers, dict):
        for name, entry in providers.items():
            if isinstance(entry, dict):
                # Conventional names (OPENAI_API_KEY, …) — matches agent.yml.example.
                _rewrite_secret_value(
                    entry, "api_key", f"{_sanitize_var_component(name)}_API_KEY", f"providers.{name}.api_key", alloc
                )

    models = data.get("models")
    if isinstance(models, dict):
        for name, entry in models.items():
            if not isinstance(entry, dict):
                continue
            prefix = f"DATUS_MODEL_{_sanitize_var_component(name)}"
            _rewrite_secret_value(entry, "api_key", f"{prefix}_API_KEY", f"models.{name}.api_key", alloc)
            headers = entry.get("default_headers")
            if isinstance(headers, dict):
                for header in list(headers):
                    _rewrite_secret_value(
                        headers,
                        header,
                        f"{prefix}_HEADER_{_sanitize_var_component(header)}",
                        f"models.{name}.default_headers.{header}",
                        alloc,
                    )

    services = data.get("services")
    if isinstance(services, dict):
        datasources = services.get("datasources")
        if isinstance(datasources, dict):
            for name, entry in datasources.items():
                if not isinstance(entry, dict):
                    continue
                prefix = f"DATUS_DS_{_sanitize_var_component(name)}"
                path = f"services.datasources.{name}"
                for fld in ("password", "username", "account", "private_key_file_pwd"):
                    _rewrite_secret_value(
                        entry, fld, f"{prefix}_{_sanitize_var_component(fld)}", f"{path}.{fld}", alloc
                    )
                _rewrite_uri_password(entry, "uri", f"{prefix}_URI_PASSWORD", f"{path}.uri", alloc)
                # Free-form extras (private_key PEM bodies, access keys, …).
                _rewrite_secret_named_keys(entry, prefix, path, alloc)
        for section, scope in (("bi_platforms", "DATUS_BI"), ("schedulers", "DATUS_SCHEDULER")):
            block = services.get(section)
            if not isinstance(block, dict):
                continue
            for name, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                prefix = f"{scope}_{_sanitize_var_component(name)}"
                path = f"services.{section}.{name}"
                for fld in ("password", "username", "api_key", "token"):
                    _rewrite_secret_value(
                        entry, fld, f"{prefix}_{_sanitize_var_component(fld)}", f"{path}.{fld}", alloc
                    )
                _rewrite_secret_named_keys(entry, prefix, path, alloc)
        semantic = services.get("semantic_layer")
        if isinstance(semantic, dict):
            _rewrite_secret_named_keys(semantic, "DATUS_SEMANTIC", "services.semantic_layer", alloc)
        mcp = services.get("mcp_servers")
        if isinstance(mcp, dict):
            for name, entry in mcp.items():
                if not isinstance(entry, dict):
                    continue
                prefix = f"DATUS_MCP_{_sanitize_var_component(name)}"
                headers = entry.get("headers")
                if isinstance(headers, dict):
                    for header in list(headers):
                        _rewrite_secret_value(
                            headers,
                            header,
                            f"{prefix}_HEADER_{_sanitize_var_component(header)}",
                            f"services.mcp_servers.{name}.headers.{header}",
                            alloc,
                        )
                env = entry.get("env")
                if isinstance(env, dict):
                    _rewrite_secret_named_keys(env, f"{prefix}_ENV", f"services.mcp_servers.{name}.env", alloc)

    plugins = data.get("plugins")
    if isinstance(plugins, dict):
        _sanitize_plugin_profiles(plugins, alloc, warnings)

    channels = data.get("channels")
    if isinstance(channels, dict):
        for name, entry in channels.items():
            if isinstance(entry, dict):
                _rewrite_secret_named_keys(
                    entry, f"DATUS_CHANNEL_{_sanitize_var_component(name)}", f"channels.{name}", alloc
                )
                extra = entry.get("extra")
                if isinstance(extra, dict):
                    for fld in ("bot_token", "app_token", "app_secret", "app_id"):
                        _rewrite_secret_value(
                            extra,
                            fld,
                            f"DATUS_CHANNEL_{_sanitize_var_component(name)}_{_sanitize_var_component(fld)}",
                            f"channels.{name}.extra.{fld}",
                            alloc,
                        )

    observability = data.get("observability")
    if isinstance(observability, dict):
        _rewrite_secret_named_keys(observability, "DATUS_TRACING", "observability", alloc)
        tracing = observability.get("tracing")
        if isinstance(tracing, dict):
            adapters = tracing.get("adapters")
            if isinstance(adapters, list):
                for idx, adapter in enumerate(adapters):
                    if isinstance(adapter, dict) and isinstance(adapter.get("headers"), dict):
                        headers = adapter["headers"]
                        for header in list(headers):
                            _rewrite_secret_value(
                                headers,
                                header,
                                f"DATUS_TRACING_{idx}_HEADER_{_sanitize_var_component(header)}",
                                f"observability.tracing.adapters[{idx}].headers.{header}",
                                alloc,
                            )

    document = data.get("document")
    if isinstance(document, dict):
        _rewrite_secret_value(document, "tavily_api_key", "TAVILY_API_KEY", "document.tavily_api_key", alloc)
        for name, entry in document.items():
            if isinstance(entry, dict):
                _rewrite_secret_value(
                    entry,
                    "github_token",
                    "GITHUB_TOKEN",
                    f"document.{name}.github_token",
                    alloc,
                )

    api = data.get("api")
    if isinstance(api, dict):
        auth = api.get("auth_provider")
        if isinstance(auth, dict) and isinstance(auth.get("kwargs"), dict):
            _rewrite_secret_named_keys(
                auth["kwargs"], "DATUS_API_AUTH", "api.auth_provider.kwargs", alloc, all_string_leaves=True
            )


def _sanitize_plugin_profiles(plugins: Dict[str, Any], alloc: _PlaceholderAllocator, warnings: List[str]) -> None:
    """``plugins.<name>.<profile>.<field>`` — schema-driven when available.

    With a manifest ``config_schema``, only ``secret``-flagged fields are
    rewritten. Without one we cannot tell which fields are credentials, so
    every string leaf becomes a placeholder (zero-leak wins) with a warning.
    """
    try:
        from datus.plugins.registry import plugin_config_schema
    except Exception:  # pragma: no cover - registry is part of the base package
        plugin_config_schema = None  # type: ignore[assignment]

    for plugin_name, profiles in plugins.items():
        if not isinstance(profiles, dict):
            continue
        secret_fields: Optional[Set[str]] = None
        if plugin_config_schema is not None:
            # A broken plugin manifest must degrade to "no schema" (all string
            # leaves become placeholders), not crash the build.
            try:
                specs = plugin_config_schema(str(plugin_name))
                if specs:
                    secret_fields = {spec["name"] for spec in specs if isinstance(spec, dict) and spec.get("secret")}
            except Exception as exc:
                logger.warning("package: plugin_config_schema(%r) failed: %s", plugin_name, exc)
                secret_fields = None
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            prefix = f"DATUS_PLUGIN_{_sanitize_var_component(plugin_name)}_{_sanitize_var_component(profile_name)}"
            path = f"plugins.{plugin_name}.{profile_name}"
            if secret_fields is None:
                warnings.append(
                    f"plugin {plugin_name!r} has no config schema; all string fields of profile "
                    f"{profile_name!r} were replaced with placeholders"
                )
                _rewrite_secret_named_keys(profile, prefix, path, alloc, all_string_leaves=True)
            else:
                # Sorted: allocation order decides collision suffixes (_2, _3),
                # so set order would rename variables between builds.
                for dotted in sorted(secret_fields):
                    container: Any = profile
                    parts = dotted.split(".")
                    for part in parts[:-1]:
                        container = container.get(part) if isinstance(container, dict) else None
                    if isinstance(container, dict):
                        _rewrite_secret_value(
                            container,
                            parts[-1],
                            f"{prefix}_{_sanitize_var_component(dotted)}",
                            f"{path}.{dotted}",
                            alloc,
                        )
                # Belt & braces: secret-named keys outside the schema.
                _rewrite_secret_named_keys(profile, prefix, path, alloc)


def generate_agent_yml(
    raw: Dict[str, Any],
    project_name: str,
    kept_subagents: Sequence[str],
    alloc: _PlaceholderAllocator,
) -> Tuple[bytes, List[str]]:
    """Regenerate ``conf/agent.yml`` from the raw dict — the technical core.

    The raw (unexpanded) YAML dict is the only lossless base: AgentConfig
    is a lossy projection (``to_dict()`` covers ~19 of ~40 sections).
    Unknown sections pass through untouched.
    """
    warnings: List[str] = []
    data = copy.deepcopy(raw)
    data["home"] = "."
    data["project_name"] = project_name
    # An absolute project_root would pin the source machine's path.
    data.pop("project_root", None)

    nodes = data.get("agentic_nodes")
    if isinstance(nodes, dict):
        filtered = {name: nodes[name] for name in kept_subagents if name in nodes}
        if filtered:
            data["agentic_nodes"] = filtered
        else:
            data.pop("agentic_nodes", None)

    _sanitize_agent_tree(data, alloc, warnings)
    alloc.harvest(data, "")
    text = yaml.safe_dump({"agent": data}, allow_unicode=True, sort_keys=False)
    return text.encode("utf-8"), warnings


def generate_mcp_json(source_home: Path, alloc: _PlaceholderAllocator) -> Optional[bytes]:
    """Sanitized copy of ``{home}/conf/.mcp.json`` (headers → placeholders)."""
    source = source_home / "conf" / ".mcp.json"
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("package: unreadable %s (%s); skipping", source, exc)
        return None
    if not isinstance(payload, dict):
        return None
    servers = payload.get("mcpServers")
    if isinstance(servers, dict):
        for name, entry in servers.items():
            if not isinstance(entry, dict):
                continue
            prefix = f"DATUS_MCP_{_sanitize_var_component(name)}"
            headers = entry.get("headers")
            if isinstance(headers, dict):
                for header in list(headers):
                    _rewrite_secret_value(
                        headers,
                        header,
                        f"{prefix}_HEADER_{_sanitize_var_component(header)}",
                        f".mcp.json:{name}.headers.{header}",
                        alloc,
                    )
            env = entry.get("env")
            if isinstance(env, dict):
                _rewrite_secret_named_keys(env, f"{prefix}_ENV", f".mcp.json:{name}.env", alloc)
    alloc.harvest(payload, ".mcp.json")
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def generate_project_config(root: Path, project_name: str) -> bytes:
    """Regenerate ``.datus/config.yml`` with the pinned project name."""
    source = root / PROJECT_CONFIG_REL
    payload: Dict[str, Any] = {}
    if source.is_file():
        try:
            loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("package: unreadable %s (%s); regenerating minimal config", source, exc)
    payload["project_name"] = project_name
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return text.encode("utf-8")


# --------------------------------------------------------------------------- #
# Step 4 — generated deliverables                                             #
# --------------------------------------------------------------------------- #


def generate_requirements(packages: Sequence[DatusPackage]) -> Tuple[bytes, List[str]]:
    lines = [f"{pkg.name}=={pkg.version}" for pkg in packages]
    editable = [pkg.name for pkg in packages if pkg.editable]
    return ("\n".join(lines) + "\n").encode("utf-8"), editable


def generate_rebuild_kb_script(
    per_ds: Dict[str, Tuple[List[str], List[str]]],
    reference_sql_count: int = 0,
    reference_sql_datasource: Optional[str] = None,
) -> Optional[bytes]:
    """Per-file bootstrap-kb loop: semantic models first, then metrics.

    ``--semantic_yaml`` accepts exactly one file, and the ``metrics``
    component only accepts files with ``metric:`` documents — hence one
    invocation per file, semantic before metrics per datasource.

    Strategy flags matter: the default ``check`` strategy only reports
    counts and ingests NOTHING. The very first semantic_model call uses
    ``overwrite`` (which truncates the whole project-scoped semantic
    store — safe exactly once, on a fresh unzip); every later call uses
    ``incremental`` so multi-file / multi-datasource selections don't
    wipe each other.
    """
    commands: List[str] = []
    overwrite_used = False
    for ds, (semantic_files, metrics_files) in per_ds.items():
        for rel in semantic_files:
            strategy = "incremental" if overwrite_used else "overwrite"
            overwrite_used = True
            commands.append(
                f"datus-agent bootstrap-kb --datasource {ds} --components semantic_model "
                f'--semantic_yaml "{rel}" --kb_update_strategy {strategy} -y'
            )
        for rel in metrics_files:
            commands.append(
                f"datus-agent bootstrap-kb --datasource {ds} --components metrics "
                f'--semantic_yaml "{rel}" --kb_update_strategy incremental -y'
            )

    notes: List[str] = []
    if reference_sql_count:
        if reference_sql_datasource:
            # --from_summaries re-indexes the packaged YAML verbatim: no LLM
            # call, no API spend, and the receiver's rows match the source
            # project's reviewed summaries exactly.
            commands.append(
                f"datus-agent bootstrap-kb --datasource {reference_sql_datasource} "
                f"--components reference_sql --from_summaries --kb_update_strategy overwrite -y"
            )
        else:
            notes += [
                "# NOTE: reference SQL shipped but no default datasource could be",
                "# resolved at pack time. Re-index the summaries manually with:",
                "#   datus-agent bootstrap-kb --datasource <ds> --components reference_sql "
                "--from_summaries --kb_update_strategy overwrite -y",
            ]

    if not commands:
        return None
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by `datus package` — rebuilds the local KB from the",
            "# sources in this package. Run from anywhere; it cd's to the package",
            "# root so the relative paths resolve.",
            "set -euo pipefail",
            'cd "$(dirname "$0")/.."',
            "",
            *notes,
            *([""] if notes else []),
            *commands,
            "",
        ]
    )
    return script.encode("utf-8")


def generate_install_plugins_script(
    root: Path,
    requested: Optional[Sequence[str]] = None,
) -> Tuple[Optional[bytes], List[str], List[str]]:
    """One ``datus plugin install`` line per selected plugin.

    Returns ``(script_or_None, kept_names, warnings)``. ``requested is None``
    keeps the project's activated plugins (falling back to everything
    installed when no activation list exists); an explicit selection wins.
    """
    available = list_packageable_plugins(root)
    if requested is None:
        override = load_project_override(cwd=str(root))
        activated = [name for name, act in (override.plugins or {}).items() if act.enabled] if override else []
        kept = sorted(activated) if activated else sorted(available)
    else:
        kept = _resolve_selection(sorted(available), requested, "plugin")
    if not kept:
        return None, [], []

    warnings: List[str] = []
    versions = _installed_plugin_versions()

    commands = []
    for name in sorted(kept):
        distribution, version = versions.get(name, (name, ""))
        spec = f"{distribution}=={version}" if version else distribution
        if not version:
            warnings.append(f"plugin {name!r}: installed version unknown; install line left unpinned")
        # --force replaces an already-installed plugin, so re-running the
        # script (or init.sh) is idempotent instead of erroring out.
        commands.append(f"datus plugin install {spec} --force")
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by `datus package` — installs the managed plugins this",
            "# project activates (see .datus/config.yml `plugins:`).",
            "set -euo pipefail",
            "",
            *commands,
            "",
        ]
    )
    return script.encode("utf-8"), kept, warnings


def generate_init_script(
    env_vars: Sequence[EnvVarBinding],
    has_plugin_script: bool,
    has_rebuild_script: bool,
) -> bytes:
    """One-command setup: dependencies → plugins → knowledge base.

    The per-step scripts stay in the package because they are worth
    re-running on their own (rebuilding the KB after editing subject YAML,
    reinstalling a plugin); ``init.sh`` is just the ordered path through
    them for a fresh unzip. It is re-runnable: pip is idempotent, plugin
    installs pass ``--force``, and the KB steps overwrite.
    """
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by `datus package` — first-run setup for this package.",
        "# Safe to re-run: every step overwrites or is idempotent.",
        "set -euo pipefail",
        'cd "$(dirname "$0")/.."',
        "",
    ]
    required = sorted({binding.var for binding in env_vars})
    if required:
        # Warn rather than fail: dependency install needs nothing, and the
        # config's ${VAR:-default} fallbacks may cover the rest.
        lines += [
            "missing=()",
            f"for var in {' '.join(required)}; do",
            '  [ -n "${!var:-}" ] || missing+=("$var")',
            "done",
            "if [ ${#missing[@]} -gt 0 ]; then",
            '  echo "WARNING: unset environment variables: ${missing[*]}" >&2',
            '  echo "         See README.md — steps needing them may fail." >&2',
            "fi",
            "",
        ]
    lines += [
        # Always target an explicit interpreter: a bare ``pip`` resolves
        # through PATH and can belong to a different Python than the
        # virtualenv that will run datus, installing the dependencies where
        # they are never imported. uv comes first because uv-created
        # virtualenvs have no ``pip`` module at all.
        'PYTHON="${PYTHON:-python3}"',
        'echo "==> Installing dependencies for $PYTHON"',
        "if command -v uv >/dev/null 2>&1; then",
        '  uv pip install --python "$PYTHON" -r requirements.txt',
        'elif "$PYTHON" -m pip --version >/dev/null 2>&1; then',
        '  "$PYTHON" -m pip install -r requirements.txt',
        "else",
        '  echo "ERROR: neither uv nor pip is available for $PYTHON." >&2',
        '  echo "       Install one, or set PYTHON=/path/to/python and re-run." >&2',
        "  exit 1",
        "fi",
        "",
    ]
    if has_plugin_script:
        lines += ['echo "==> Installing plugins"', "bash scripts/install_plugins.sh", ""]
    if has_rebuild_script:
        lines += ['echo "==> Rebuilding the knowledge base"', "bash scripts/rebuild_kb.sh", ""]
    # Single-quoted: backticks inside a double-quoted echo would be command
    # substitution, and this line names the ``datus`` command.
    lines += ["echo '==> Done. Start with: datus-api   (or datus for the interactive console)'", ""]
    return "\n".join(lines).encode("utf-8")


def generate_readme(
    project_name: str,
    env_vars: Sequence[EnvVarBinding],
    has_rebuild_script: bool,
    has_plugin_script: bool,
    report_count: int,
    dashboard_count: int,
    offline_reports: bool,
) -> bytes:
    """Receiver-facing quickstart — env list first, everything else after."""
    lines: List[str] = [
        f"# {project_name}",
        "",
        "Self-contained Datus project package. Everything runs inside this",
        "directory; your `~/.datus` is not touched.",
        "",
        "## Quick start",
        "",
        "```bash",
    ]
    if env_vars:
        sample = " ".join(f"{binding.var}=..." for binding in _unique_vars(env_vars)[:2])
        lines.append(f"export {sample}   # full list below")
    lines += [
        "bash scripts/init.sh   # dependencies"
        + (" + plugins" if has_plugin_script else "")
        + (" + knowledge base" if has_rebuild_script else ""),
        "datus-api              # or `datus` for the interactive console",
        "```",
        "",
        "`init.sh` is safe to re-run. The steps it drives are also available",
        "on their own:",
        "",
        "```bash",
        "pip install -r requirements.txt",
    ]
    if has_plugin_script:
        lines.append("bash scripts/install_plugins.sh   # datus plugin install --force, per plugin")
    if has_rebuild_script:
        lines.append("bash scripts/rebuild_kb.sh        # re-index metrics / semantic models / reference SQL")
    lines.append("```")
    lines += ["", "## Required environment variables", ""]
    if env_vars:
        lines += ["| Variable | Used by |", "|---|---|"]
        for record in group_env_vars(env_vars):
            lines.append(f"| `{record.var}` | {', '.join(record.config_paths)} |")
    else:
        lines.append("None — this package carries no credential-bearing config.")
    lines += [
        "",
        "`.env` files are NOT auto-loaded when datus-agent is installed via pip.",
        "Export the variables in your shell, or run `set -a; source .env; set +a`.",
        "",
        "## Notes",
        "",
        "- `conf/agent.yml` was regenerated at pack time: `home: .` keeps all",
        f"  runtime state inside this directory and `project_name: {project_name}`",
        "  keeps session/index shards stable wherever you unzip.",
        "- No secrets ship in this package; every credential field is a",
        "  `${VAR}` placeholder resolved from your environment.",
    ]
    if report_count:
        if offline_reports:
            lines.append("- Report `index.html` files are self-contained — open them via `file://`.")
        else:
            lines.append("- Report `index.html` files load their viewer from a CDN; opening them needs network access.")
    if dashboard_count:
        lines.append(
            "- Dashboard `index.html` files need a running `datus --web` query endpoint; `file://` alone won't work."
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def group_env_vars(env_vars: Sequence[EnvVarBinding]) -> List[EnvVarRequirement]:
    """Collapse the ``(var, config_path)`` binding list into one record per var.

    ``_PlaceholderAllocator`` records a binding per ``(var, config_path)`` pair,
    so one variable can appear several times — a single API key referenced from
    both ``providers.*`` and ``models.*``, say. Both the README table and the
    package manifest want the per-variable view; sharing this keeps the two from
    drifting apart.

    ``preexisting`` is aggregated with ``all``: a var counts as preexisting only
    if EVERY site already carried a ``${VAR}``. If any site held a literal the
    packer rewrote, the receiver is being asked to supply something that used to
    be a plaintext secret — the stronger claim, and the one that must not be
    hidden by an unrelated harvested occurrence.
    """
    grouped: Dict[str, List[EnvVarBinding]] = {}
    for binding in env_vars:
        grouped.setdefault(binding.var, []).append(binding)
    return [
        EnvVarRequirement(
            var=var,
            config_paths=sorted({binding.config_path for binding in grouped[var]}),
            preexisting=all(binding.preexisting for binding in grouped[var]),
        )
        for var in sorted(grouped)
    ]


def _unique_vars(env_vars: Sequence[EnvVarBinding]) -> List[EnvVarBinding]:
    seen = set()
    out = []
    for binding in env_vars:
        if binding.var not in seen:
            seen.add(binding.var)
            out.append(binding)
    return out


def build_package_manifest(
    entries: Sequence[StagedEntry],
    project_name: str,
    selections: Dict[str, Any],
    env_vars: Sequence[EnvVarBinding],
    editable_packages: Sequence[str],
) -> bytes:
    files = [
        {
            "path": entry.arcname,
            "sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
            "size": entry.size(),
            # Per-file provenance: generated at pack time vs copied from the
            # source project tree.
            "source": "generated" if entry.content is not None else "project",
        }
        for entry in sorted(entries, key=lambda e: e.arcname)
    ]
    manifest = {
        "format": PACKAGE_FORMAT,
        "format_version": PACKAGE_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": f"datus/{datus.__version__}",
        "project_name": project_name,
        "selections": selections,
        # One record per variable, each naming the config paths that reference
        # it. The receiver has to bind every ``${VAR}`` to something; without the
        # paths they can only guess what a given variable is for.
        "env_vars": [record.as_manifest_record() for record in group_env_vars(env_vars)],
        "editable_source_packages": sorted(editable_packages),
        "files": files,
    }
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- #
# Step 5 — final secret scan + zip                                            #
# --------------------------------------------------------------------------- #


def _scan_generated_config(arcname: str, text: str) -> List[SecretFinding]:
    """Self-check over files WE generated: any secret-named key must carry a
    placeholder (or be empty). Catches secret paths missing from the table."""
    findings: List[SecretFinding] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(value, (dict, list)):
                    _walk(value, child)
                elif isinstance(value, str) and value.strip() and _is_secret_key(str(key)):
                    if not _PLACEHOLDER_RE.match(value.strip()):
                        findings.append(SecretFinding(arcname=arcname, locator=child, kind="plaintext_secret_key"))
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _walk(item, f"{path}[{idx}]")

    try:
        parsed = json.loads(text) if arcname.endswith(".json") else yaml.safe_load(text)
    except (ValueError, yaml.YAMLError):
        return findings
    _walk(parsed, "")
    return findings


def scan_for_secrets(entries: Sequence[StagedEntry]) -> Tuple[List[SecretFinding], List[str]]:
    """Return ``(findings, warnings)``; warnings flag partial coverage.

    The scan reads at most ``_SCAN_READ_CAP_BYTES`` per file and skips
    binary-sniffed content — both limits are surfaced as warnings so a
    partial scan never silently reads as a full one.
    """
    findings: List[SecretFinding] = []
    warnings: List[str] = []
    generated_conf = {"conf/agent.yml", "conf/.mcp.json", PROJECT_CONFIG_REL}
    for entry in entries:
        try:
            head = entry.read_bytes(cap=_SCAN_READ_CAP_BYTES)
        except OSError as exc:
            findings.append(SecretFinding(arcname=entry.arcname, locator=str(exc), kind="unreadable"))
            continue
        if entry.size() > _SCAN_READ_CAP_BYTES:
            warnings.append(
                f"secret scan truncated for {entry.arcname}: only the first "
                f"{_SCAN_READ_CAP_BYTES // (1024 * 1024)} MB of {entry.size()} bytes were scanned"
            )
        if b"\x00" in head[:_BINARY_SNIFF_BYTES]:
            continue
        for kind, pattern in _SECRET_CONTENT_PATTERNS:
            match = pattern.search(head)
            if match:
                findings.append(SecretFinding(arcname=entry.arcname, locator=f"offset {match.start()}", kind=kind))
        if entry.arcname in generated_conf and entry.content is not None:
            findings.extend(_scan_generated_config(entry.arcname, entry.content.decode("utf-8", errors="replace")))
    return findings, warnings


def write_zip(entries: Sequence[StagedEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # One timestamp for every member: entries written seconds apart must not
    # differ, so two runs over identical content diverge only by build time.
    stamp = datetime.now(timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in sorted(entries, key=lambda e: e.arcname):
            data = entry.read_bytes()
            info = zipfile.ZipInfo(entry.arcname, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if entry.executable else 0o644) << 16
            zf.writestr(info, data)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


def default_output_path(root: Path, project_name: str) -> Path:
    return root / f"{project_name}.zip"


def build_package(options: PackageOptions) -> PackageResult:
    """Run the full pipeline (collect → select → generate → scan → zip)."""
    try:
        return _build_package(options)
    except PackageError as exc:
        return PackageResult(ok=False, error=str(exc))
    except (OSError, zipfile.BadZipFile) as exc:
        logger.error("package build failed: %s", exc)
        return PackageResult(ok=False, error=str(exc))


def _build_package(options: PackageOptions) -> PackageResult:
    root = options.root.resolve()
    raw = load_raw_agent_config()
    if raw is None:
        raise PackageError("no agent configuration found (conf/agent.yml or ~/.datus/conf/agent.yml)")
    project_name = resolve_effective_project_name(root, raw)
    source_home = resolve_source_home(raw, root)
    output_path = (options.output or default_output_path(root, project_name)).resolve()

    warnings: List[str] = []
    entries_by_arc: Dict[str, StagedEntry] = {}

    def _add(new_entries: Sequence[StagedEntry]) -> None:
        for entry in new_entries:
            if entry.arcname in entries_by_arc:
                warnings.append(f"duplicate staging of {entry.arcname}; keeping the later entry")
            entries_by_arc[entry.arcname] = entry

    walk_entries, walk_warnings = collect_project_files(root, options.include, options.exclude, output_path)
    warnings.extend(walk_warnings)
    _add(walk_entries)

    kept_subagents, template_entries, sub_warnings = select_subagents(raw, options.subagents, source_home)
    warnings.extend(sub_warnings)
    _add(template_entries)

    kept_skills, skill_entries = select_skills(root, options.skills)
    _add(skill_entries)

    available_subjects = list(list_subject_paths(root, raw, project_name))
    kept_subjects = _resolve_selection(available_subjects, options.subjects, "subject")

    kept_metrics, metric_entries, per_ds, metric_warnings = select_metrics(
        root, options.metrics, None if options.subjects is None else kept_subjects
    )
    warnings.extend(metric_warnings)
    _add(metric_entries)

    reference_sql_entries, reference_sql_count, summary_warnings = select_reference_sql(root, kept_subjects)
    warnings.extend(summary_warnings)
    _add(reference_sql_entries)

    kept_reports, report_entries, report_warnings = select_artifacts(
        root, "report", options.reports, options.report_dist
    )
    warnings.extend(report_warnings)
    _add(report_entries)
    kept_dashboards, dashboard_entries, dash_warnings = select_artifacts(root, "dashboard", options.dashboards, None)
    warnings.extend(dash_warnings)
    _add(dashboard_entries)

    alloc = _PlaceholderAllocator()
    agent_yml, gen_warnings = generate_agent_yml(raw, project_name, kept_subagents, alloc)
    warnings.extend(gen_warnings)
    _add([StagedEntry(arcname="conf/agent.yml", content=agent_yml)])

    mcp_json = generate_mcp_json(source_home, alloc)
    if mcp_json is not None:
        _add([StagedEntry(arcname="conf/.mcp.json", content=mcp_json)])

    _add([StagedEntry(arcname=PROJECT_CONFIG_REL, content=generate_project_config(root, project_name))])

    packages = enumerate_datus_packages()
    requirements, editable = generate_requirements(packages)
    if editable:
        # Editable installs never block the build — surface the caveat as a
        # warning and pin the reported version as-is.
        warnings.append(
            f"editable/source installs pinned as-is: {', '.join(editable)} — the PyPI "
            "wheel for the pinned version may differ from your source tree"
        )
    _add([StagedEntry(arcname="requirements.txt", content=requirements)])

    rebuild = generate_rebuild_kb_script(per_ds, reference_sql_count, resolve_default_datasource(root, raw))
    if rebuild is not None:
        _add([StagedEntry(arcname="scripts/rebuild_kb.sh", content=rebuild, executable=True)])

    plugin_script, kept_plugins, plugin_warnings = generate_install_plugins_script(root, options.plugins)
    warnings.extend(plugin_warnings)
    if plugin_script is not None:
        _add([StagedEntry(arcname="scripts/install_plugins.sh", content=plugin_script, executable=True)])

    _add(
        [
            StagedEntry(
                arcname="scripts/init.sh",
                content=generate_init_script(alloc.bindings, plugin_script is not None, rebuild is not None),
                executable=True,
            )
        ]
    )

    readme = generate_readme(
        project_name,
        alloc.bindings,
        has_rebuild_script=rebuild is not None,
        has_plugin_script=plugin_script is not None,
        report_count=len(kept_reports),
        dashboard_count=len(kept_dashboards),
        offline_reports=options.report_dist is not None,
    )
    _add([StagedEntry(arcname="README.md", content=readme)])

    # Disk-backed entries can vanish between collection and finalize (editor
    # swap files, concurrent cleanups). Drop them with a warning instead of
    # failing the whole build on a FileNotFoundError deep in scan/zip.
    staged = []
    for entry in entries_by_arc.values():
        if entry.source is not None and not entry.source.is_file():
            warnings.append(f"skipped {entry.arcname}: file vanished during packaging")
            continue
        staged.append(entry)

    for entry in staged:
        if entry.source is not None and entry.size() > _LARGE_FILE_WARN_BYTES:
            warnings.append(f"large file packaged: {entry.arcname} ({entry.size() // (1024 * 1024)} MB)")

    selections = {
        "subagents": kept_subagents,
        "skills": kept_skills,
        "metrics": kept_metrics,
        "subjects": kept_subjects,
        "reference_sql_entries": reference_sql_count,
        "plugins": kept_plugins,
        "reports": kept_reports,
        "dashboards": kept_dashboards,
        "include": list(options.include),
        "exclude": list(options.exclude),
        "report_dist": str(options.report_dist) if options.report_dist else None,
    }

    findings, scan_warnings = scan_for_secrets(staged)
    warnings.extend(scan_warnings)
    if findings:
        return PackageResult(
            ok=False,
            error="secret scan failed — real credential material detected in the staged package",
            warnings=warnings,
            secret_findings=findings,
            env_vars=alloc.bindings,
        )

    manifest = build_package_manifest(staged, project_name, selections, alloc.bindings, editable)
    staged.append(StagedEntry(arcname=PACKAGE_MANIFEST_NAME, content=manifest))

    write_zip(staged, output_path)
    total = sum(entry.size() for entry in staged)
    logger.info("package built: %s (%d files, %d bytes)", output_path, len(staged), total)
    return PackageResult(
        ok=True,
        zip_path=str(output_path),
        file_count=len(staged),
        total_bytes=total,
        env_vars=alloc.bindings,
        warnings=warnings,
        selections=selections,
    )


__all__ = [
    "EnvVarBinding",
    "PackageError",
    "PackageOptions",
    "PackageResult",
    "SecretFinding",
    "EnvVarRequirement",
    "StagedEntry",
    "build_package",
    "default_output_path",
    "group_env_vars",
    "list_artifact_slugs",
    "list_metric_datasources",
    "list_packageable_plugins",
    "list_subject_paths",
    "list_skills",
    "list_subagents",
    "load_raw_agent_config",
    "resolve_effective_project_name",
]
