# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Bring the Knowledge Base and subject tree back in line with semantic YAML.

The YAML under ``subject/semantic_models/<datasource>/`` is the source of truth;
metric and semantic-dataset rows, and the subject directories metrics hang
from, are projections of it. Anything that changes files outside the
validating save path — an IDE delete or rename, an agent's file tools, a shell
command — leaves those projections stale. ``reconcile_semantic_artifacts`` is
the one repair entry for all of them:

* every file named in ``paths`` (or found under a named directory) is
  re-projected through ``sync_osi_to_db``;
* rows whose ``yaml_path`` no longer exists are dropped, whichever level was
  deleted — a model file, a datasource directory, ``semantic_models`` or
  ``subject`` itself;
* subject directories emptied by either step are removed, walking up until an
  ancestor still holds something. Directories that were already empty are
  left alone: an empty directory created in the Explorer is real structure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set

from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.configuration.agent_config import AgentConfig

logger = get_logger(__name__)


@dataclass
class SemanticReconcileResult:
    synced_files: List[str] = field(default_factory=list)
    pruned_files: List[str] = field(default_factory=list)
    removed_subject_paths: List[List[str]] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)


class _DatasourceStores:
    """The unscoped stores one datasource's projections live in.

    Unscoped on purpose: a row whose file is gone is garbage for every
    sub-agent, and a directory is only empty when no one's entries remain.
    """

    def __init__(self, agent_config: "AgentConfig", datasource_id: str):
        from datus.storage.metric.store import MetricRAG
        from datus.storage.reference_sql.store import ReferenceSqlRAG
        from datus.storage.registry import get_subject_tree_store
        from datus.storage.semantic_dataset.store import SemanticDatasetRAG

        self.agent_config = agent_config
        self.metric_rag = MetricRAG(agent_config, datasource_id=datasource_id)
        self.dataset_rag = SemanticDatasetRAG(agent_config, datasource_id=datasource_id)
        self.reference_sql_rag = ReferenceSqlRAG(agent_config, datasource_id=datasource_id)
        self.tree = get_subject_tree_store(project=agent_config.project_name, datasource_id=datasource_id)

    def metric_node_ids(self, yaml_path: str) -> Set[int]:
        from datus.storage.subject_tree.store import SUBJECT_ID_COLUMN_NAME

        return {
            row[SUBJECT_ID_COLUMN_NAME]
            for row in self.metric_rag.list_artifact_rows(yaml_path)
            if row.get(SUBJECT_ID_COLUMN_NAME) is not None
        }

    def prune_missing(
        self, keep: Set[str], under: Path, failures: Optional[Dict[str, str]] = None
    ) -> tuple[List[str], Set[int]]:
        """Drop rows of artifacts no longer on disk; return them and the nodes they left.

        Only rows whose ``yaml_path`` lies under ``under`` are candidates. A row
        recorded under another root (a relocated mount, a relative legacy path)
        would otherwise read as deleted and take the whole project with it.
        """
        root = under.expanduser().resolve(strict=False)
        pruned: Set[str] = set()
        node_ids: Set[int] = set()
        for rag in (self.dataset_rag, self.metric_rag):
            for yaml_path in rag.list_artifact_paths():
                normalized = Path(_normalized(yaml_path))
                if not Path(yaml_path).is_absolute() or not _is_relative_to(normalized, root):
                    continue
                if str(normalized) in keep or Path(yaml_path).exists():
                    continue
                try:
                    if rag is self.metric_rag:
                        node_ids |= self.metric_node_ids(yaml_path)
                    rag.delete_artifact_rows(yaml_path)
                except Exception as exc:  # noqa: BLE001 - one artifact must not block the rest
                    logger.exception(f"Failed to prune rows for deleted semantic model '{yaml_path}'")
                    if failures is not None:
                        failures[yaml_path] = str(exc)
                    continue
                pruned.add(yaml_path)
        return sorted(pruned), node_ids

    def projection_matches(self, yaml_path: str) -> bool:
        """Whether the KB holds what ``yaml_path`` declares, by metric names and dataset presence.

        Cheap enough to run on every file of a first reconcile; catches metrics
        added or removed behind the save path, not an expression edited in place.
        """
        declared = _declared_objects(yaml_path)
        if declared is None:
            return True  # Unreadable: a sync would only fail the same way.
        metric_names, declares_datasets = declared
        projected = {str(row.get("name") or "").strip() for row in self.metric_rag.list_artifact_rows(yaml_path)}
        if projected - {""} != metric_names:
            return False
        return not declares_datasets or bool(self.dataset_rag.list_artifact_rows(yaml_path))

    def remove_emptied_nodes(self, node_ids: Iterable[int]) -> List[List[str]]:
        removed: List[List[str]] = []
        for node_id in node_ids:
            current: Optional[int] = node_id
            while current is not None:
                node = self.tree.get_node(current)
                if not node or not self._is_empty(current):
                    break
                path = self.tree.get_full_path(current)
                parent_id = node.get("parent_id")
                self.tree.delete_node(current, cascade=False)
                removed.append(path)
                current = parent_id
        return removed

    def _is_empty(self, node_id: int) -> bool:
        if self.tree.get_children(node_id):
            return False
        # Node ids are per datasource, so the entry lookups must be too.
        metrics = self.metric_rag.storage.list_entries(
            node_id, limit=1, extra_conditions=self.metric_rag._sub_agent_conditions()
        )
        if metrics:
            return False
        return not self.reference_sql_rag.reference_sql_storage.list_entries(
            node_id, limit=1, extra_conditions=self.reference_sql_rag._sub_agent_conditions()
        )


def reconcile_semantic_artifacts(
    agent_config: "AgentConfig",
    paths: Iterable[str] = (),
) -> SemanticReconcileResult:
    """Re-project changed semantic YAML and drop what deleted files projected.

    Args:
        agent_config: The project's config; each datasource is reconciled on a
            copy switched to it, so the shared config is never mutated.
        paths: Project-relative or absolute paths that were created, modified
            or renamed to. Deleted paths need not be listed — anything whose
            file is gone is pruned regardless. Paths outside
            ``subject/semantic_models`` are ignored unless they contain it.
    """
    from datus.agent.node.semantic_authoring import osi_semantic_models_root

    result = SemanticReconcileResult()
    root = osi_semantic_models_root(agent_config)
    datasources = list(getattr(getattr(agent_config, "services", None), "datasources", {}) or {})
    if root is None or not datasources:
        return result
    root = root.expanduser().resolve(strict=False)

    files_by_datasource = _files_to_sync(agent_config, root, paths, datasources, result)
    for datasource in datasources:
        try:
            _reconcile_datasource(
                _config_for(agent_config, datasource),
                datasource,
                root,
                files_by_datasource.get(datasource, []),
                result,
            )
        except Exception as exc:  # noqa: BLE001 - one datasource must not block the rest
            logger.exception(f"Failed to reconcile semantic artifacts for datasource '{datasource}'")
            result.failures[datasource] = str(exc)
    return result


def _reconcile_datasource(
    agent_config: "AgentConfig",
    datasource: str,
    root: Path,
    explicit: List[Path],
    result: SemanticReconcileResult,
) -> None:
    from datus.storage.semantic_model.artifact_file import semantic_artifact_lock
    from datus.storage.semantic_model.semantic_model_init import reject_non_dosi_semantic_yaml, semantic_yaml_files
    from datus.storage.semantic_model.sync_state import (
        file_digest,
        forget_digests,
        load_digests,
        record_digests,
        state_key,
    )
    from datus.tools.func_tool.generation_tools import GenerationTools

    stores = _DatasourceStores(agent_config, datasource)
    on_disk = [path.resolve(strict=False) for path in semantic_yaml_files(root / datasource)]
    digests = load_digests(agent_config, datasource)
    # Files edited behind the save path (agent tools, shell, git) show up as a
    # digest that no longer matches what was last projected.
    if digests is not None:
        changed = [path for path in on_disk if digests.get(state_key(path)) != file_digest(path)]
    else:
        # No digests (first run, or the state was lost): trust only projections
        # that still match what their file declares, and re-project the rest.
        changed = [path for path in on_disk if not stores.projection_matches(str(path))]

    touched_nodes: Set[int] = set()
    failed: Set[str] = set()
    tools: Optional[GenerationTools] = None
    for path in sorted(set(explicit) | set(changed)):
        yaml_path = str(path)
        rejection = reject_non_dosi_semantic_yaml(yaml_path, agent_config)
        if rejection:
            result.failures[yaml_path] = rejection
            failed.add(yaml_path)
            continue
        # Every open browser reconciles on the same fs:changed; holding the
        # file's lock and re-checking its digest folds them into one sync.
        with semantic_artifact_lock(path):
            latest = load_digests(agent_config, datasource)
            if latest is not None and latest.get(state_key(path)) == file_digest(path):
                continue
            # A metric moved to another subject_path leaves its old directory behind.
            touched_nodes |= stores.metric_node_ids(yaml_path)
            tools = tools or GenerationTools(agent_config=agent_config, authoring_format="osi")
            try:
                sync = tools.sync_osi_to_db(yaml_path, include_semantic_objects=True, include_metrics=True)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                logger.exception(f"Failed to sync semantic YAML file '{yaml_path}'")
                sync = {"success": False, "error": str(exc)}
        if sync.get("success"):
            result.synced_files.append(yaml_path)
        else:
            result.failures[yaml_path] = str(sync.get("error") or "Unknown error")
            failed.add(yaml_path)

    if digests is None:
        # Re-projecting every file of every project at once on upgrade is what
        # the baseline avoids; the mismatching ones were synced above.
        baseline = {str(path): file_digest(path) for path in on_disk if str(path) not in failed}
        record_digests(agent_config, datasource, {path: digest for path, digest in baseline.items() if digest})

    pruned, pruned_nodes = stores.prune_missing(keep=set(), under=root, failures=result.failures)
    result.pruned_files.extend(pruned)
    touched_nodes |= pruned_nodes
    present = {state_key(path) for path in on_disk}
    forget_digests(
        agent_config,
        datasource,
        [path for path in (digests or {}) if path not in present] + pruned,
    )
    result.removed_subject_paths.extend(stores.remove_emptied_nodes(touched_nodes))


def _files_to_sync(
    agent_config: "AgentConfig",
    root: Path,
    paths: Iterable[str],
    datasources: List[str],
    result: SemanticReconcileResult,
) -> Dict[str, List[Path]]:
    from datus.storage.semantic_model.semantic_model_init import semantic_yaml_files

    project_root = Path(str(agent_config.project_root)).expanduser().resolve(strict=False)
    found: Dict[str, Set[Path]] = {}
    for raw in paths:
        target = Path(str(raw)).expanduser()
        target = (target if target.is_absolute() else project_root / target).resolve(strict=False)
        if not target.exists():
            continue
        if _is_relative_to(root, target):
            # An ancestor of the root (``subject`` renamed into place, say).
            scopes = [root / datasource for datasource in datasources]
        elif _is_relative_to(target, root) and target != root:
            scopes = [target]
        else:
            continue
        for scope in scopes:
            if not scope.exists():
                continue
            datasource = scope.relative_to(root).parts[0]
            if datasource not in datasources:
                result.failures[str(scope)] = f"{datasource!r} is not a datasource of this project"
                continue
            for file in semantic_yaml_files(scope):
                # A file directly under the root belongs to no datasource.
                if len(file.relative_to(root).parts) > 1:
                    found.setdefault(datasource, set()).add(file.resolve(strict=False))
    return {datasource: sorted(files) for datasource, files in found.items()}


def _declared_objects(yaml_path: str) -> Optional[tuple[Set[str], bool]]:
    """``(metric names, declares any dataset)`` of an OSI file, or ``None`` if unreadable."""
    import yaml

    try:
        doc = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    models = doc.get("semantic_model") if isinstance(doc, dict) else None
    if not isinstance(models, list):
        return None
    metrics: Set[str] = set()
    datasets = False
    for model in models:
        if not isinstance(model, dict):
            continue
        datasets = datasets or bool(model.get("datasets"))
        for metric in model.get("metrics") or []:
            if isinstance(metric, dict) and str(metric.get("name") or "").strip():
                metrics.add(str(metric["name"]).strip())
    return metrics, datasets


def _config_for(agent_config: "AgentConfig", datasource: str) -> "AgentConfig":
    if getattr(agent_config, "current_datasource", "") == datasource:
        return agent_config
    clone = copy.copy(agent_config)
    clone.current_datasource = datasource
    return clone


def _normalized(path: str) -> str:
    return str(Path(path).resolve(strict=False))


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True
