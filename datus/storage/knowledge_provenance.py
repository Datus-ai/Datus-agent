# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

_REFERENCE_SQL = "reference_sql"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def is_knowledge_provenance_enabled(agent_config: AgentConfig) -> bool:
    """Return whether benchmark/evaluation provenance sidecar writes are enabled."""
    raw = getattr(agent_config, "knowledge_base", {}) or {}
    if not isinstance(raw, dict):
        return False

    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        return False
    return _coerce_bool(provenance.get("enabled"), False)


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", ";").split(";")]
        return [part for part in parts if part]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class KnowledgeProvenanceStore:
    """File-backed provenance sidecar for benchmark/evaluation-only metadata.

    The sidecar intentionally avoids changing primary vector-table schemas. It is
    disabled by default and only used when ``agent.knowledge_base.provenance`` is
    enabled, so existing user data remains untouched.
    """

    def __init__(self, agent_config: AgentConfig, file_path: Optional[Path] = None):
        self.agent_config = agent_config
        self.file_path = file_path or (agent_config.path_manager.project_data_dir / "knowledge_provenance.json")

    def _load_rows(self) -> List[Dict[str, Any]]:
        if not self.file_path.exists():
            return []
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load knowledge provenance sidecar %s: %s", self.file_path, exc)
            return []
        if not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(self.file_path.parent), delete=False) as handle:
            handle.write(payload)
            handle.write("\n")
            tmp_path = Path(handle.name)
        tmp_path.replace(self.file_path)

    @staticmethod
    def _row_key(row: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("artifact_type") or ""),
            str(row.get("artifact_id") or ""),
            str(row.get("source_id") or ""),
            str(row.get("source_context_id") or ""),
        )

    def upsert_many(self, rows: Iterable[Dict[str, Any]]) -> int:
        existing = self._load_rows()
        by_key = {self._row_key(row): dict(row) for row in existing}
        now = _now_iso()
        written = 0

        for row in rows:
            artifact_type = str(row.get("artifact_type") or "").strip()
            artifact_id = str(row.get("artifact_id") or "").strip()
            if not artifact_type or not artifact_id:
                continue

            normalized = dict(row)
            normalized["artifact_type"] = artifact_type
            normalized["artifact_id"] = artifact_id
            normalized["source_id"] = str(normalized.get("source_id") or "")
            normalized["source_context_id"] = str(normalized.get("source_context_id") or "")
            normalized["source_type"] = str(normalized.get("source_type") or "")
            metadata = normalized.get("source_metadata")
            normalized["source_metadata"] = metadata if isinstance(metadata, dict) else {}

            key = self._row_key(normalized)
            created_at = by_key.get(key, {}).get("created_at") or now
            normalized["created_at"] = created_at
            normalized["updated_at"] = now
            by_key[key] = normalized
            written += 1

        if written:
            self._write_rows(sorted(by_key.values(), key=self._row_key))
        return written

    def find_by_artifact_ids(self, artifact_type: str, artifact_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids = {str(artifact_id) for artifact_id in artifact_ids if artifact_id}
        if not ids:
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for row in self._load_rows():
            if row.get("artifact_type") != artifact_type or row.get("artifact_id") not in ids:
                continue
            artifact_id = str(row.get("artifact_id"))
            entry = result.setdefault(
                artifact_id,
                {"source_ids": [], "source_context_ids": [], "source_metadata": []},
            )
            source_id = str(row.get("source_id") or "")
            if source_id and source_id not in entry["source_ids"]:
                entry["source_ids"].append(source_id)
            context_id = str(row.get("source_context_id") or "")
            if context_id and context_id not in entry["source_context_ids"]:
                entry["source_context_ids"].append(context_id)
            metadata = row.get("source_metadata")
            if isinstance(metadata, dict) and metadata and metadata not in entry["source_metadata"]:
                entry["source_metadata"].append(metadata)
        return result


def build_reference_sql_provenance_rows(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build sidecar rows from processed reference SQL bootstrap items."""
    from datus.storage.reference_sql.init_utils import gen_reference_sql_id

    rows: List[Dict[str, Any]] = []
    for item in items:
        sql = item.get("sql") or ""
        artifact_id = item.get("id") or gen_reference_sql_id(sql)
        if not artifact_id:
            continue

        source_id = str(item.get("source_id") or "")
        context_ids = _normalize_string_list(item.get("source_context_ids") or item.get("source_context_id"))
        if not source_id and not context_ids:
            continue

        metadata = item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {}
        source_type = str(item.get("source_type") or metadata.get("source_type") or "sql_file")
        if not source_id:
            source_id = str(metadata.get("source_id") or "")
        context_values = context_ids or [""]

        for context_id in context_values:
            rows.append(
                {
                    "artifact_type": _REFERENCE_SQL,
                    "artifact_id": artifact_id,
                    "source_id": source_id,
                    "source_context_id": context_id,
                    "source_type": source_type,
                    "source_metadata": metadata,
                }
            )
    return rows


def enrich_reference_sql_results(agent_config: AgentConfig, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results or not is_knowledge_provenance_enabled(agent_config):
        return results

    ids = [str(item.get("id")) for item in results if isinstance(item, dict) and item.get("id")]
    if not ids:
        return results

    provenance = KnowledgeProvenanceStore(agent_config).find_by_artifact_ids(_REFERENCE_SQL, ids)
    if not provenance:
        return results

    enriched: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            enriched.append(item)
            continue
        artifact_id = str(item.get("id") or "")
        metadata = provenance.get(artifact_id)
        if metadata:
            updated = dict(item)
            updated.update(metadata)
            enriched.append(updated)
        else:
            enriched.append(item)
    return enriched
