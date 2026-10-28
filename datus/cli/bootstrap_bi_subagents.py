# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Pure helpers and the sub-agent persistence stream for ``/bootstrap-bi``.

Everything here is plain Python — no Rich console, no progress callbacks.
The single async generator (``stream_bi_save_subagents``) wraps the two
``SubAgentManager.save_agent`` calls plus ``agent_config.agentic_nodes``
refresh into a chat-style ``task(save_subagents, …)`` group via
:func:`bootstrap_subagent.as_task_subagent`.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, Sequence

import pandas as pd

from datus.cli.bootstrap_subagent import as_task_subagent
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import (
    ActionHistory,
    ActionHistoryManager,
    ActionRole,
    ActionStatus,
)
from datus.schemas.agent_models import ScopedContext, SubAgentConfig
from datus.tools.bi_tools.dashboard_assembler import SelectedSqlCandidate
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import metadata_identifier, parse_table_name_parts
from datus.utils.sub_agent_manager import SubAgentManager

logger = get_logger(__name__)


# ── identifier / naming ───────────────────────────────────────────────


def normalize_identifier(text: str, max_words: Optional[int] = None, fallback: str = "item") -> str:
    """Normalize a free-form label into a filesystem/identifier-friendly token.

    Keeps ASCII alphanumerics and CJK ranges; lower-cases ASCII tokens; joins
    with ``_``; collapses repeated separators. Mirrors the behaviour from
    the original ``bi_dashboard._normalize_identifier`` so existing
    sub-agent names remain stable.
    """
    raw = (text or "").strip()
    if not raw:
        return fallback

    pattern = r"[A-Za-z0-9]+|[\u4E00-\u9FFF]+"
    tokens = re.findall(pattern, raw)
    if max_words is not None and len(tokens) > max_words:
        tokens = tokens[:max_words]
    if not tokens:
        return fallback

    normalized = [tok.lower() if tok.isascii() else tok for tok in tokens]
    out = "_".join(p for p in normalized if p)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or fallback


def build_sub_agent_name(platform: str, dashboard_name: str) -> str:
    platform_token = normalize_identifier(platform, fallback="bi")
    dashboard_token = normalize_identifier(dashboard_name, max_words=3, fallback="dashboard")
    name = f"{platform_token}_{dashboard_token}".strip("_")
    if not name or not name[0].isalpha():
        name = f"dashboard_{name}" if name else "dashboard_agent"
    return name


def parse_subject_path_for_metrics(tags: List[str]) -> Optional[str]:
    """Extract the dotted subject path from a metric YAML's locked tags."""
    if not tags:
        return None
    for tag in tags:
        if tag.startswith("subject_tree:"):
            parts = [p.strip() for p in tag[13:].strip().split("/") if p.strip()]
            if parts:
                return ".".join(parts)
    return None


# ── value scrubbing ───────────────────────────────────────────────────


def dedupe_values(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def clean_comment_text(text: str) -> str:
    return " ".join(str(text).split())


# ── table-name qualification ──────────────────────────────────────────


def qualify_table_names(
    table_names: List[str],
    agent_config: AgentConfig,
    *,
    catalog: str,
    database: str,
    schema: str,
) -> List[str]:
    """Fill missing catalog/database/schema fields per the dialect's hierarchy.

    The caller is responsible for resolving the live (catalog, database,
    schema) tuple — this helper does not depend on a CLI context.
    """
    dialect = agent_config.db_type or ""
    qualified: List[str] = []
    for name in table_names:
        if not (name or "").strip():
            continue
        parts = parse_table_name_parts(name, dialect)
        if not parts.get("catalog_name") and catalog:
            parts["catalog_name"] = catalog
        if not parts.get("database_name") and database:
            parts["database_name"] = database
        if not parts.get("schema_name") and schema:
            parts["schema_name"] = schema
        qualified.append(
            metadata_identifier(
                catalog_name=parts.get("catalog_name", ""),
                database_name=parts.get("database_name", ""),
                schema_name=parts.get("schema_name", ""),
                table_name=parts.get("table_name", ""),
                dialect=dialect,
            )
        )
    return qualified


# ── file materialization ──────────────────────────────────────────────


def _build_sql_file_name(platform: str, dashboard_name: str) -> str:
    platform_token = normalize_identifier(platform, fallback="bi")
    dashboard_token = normalize_identifier(dashboard_name or "", max_words=3, fallback="dashboard")
    parts = [p for p in (platform_token, dashboard_token) if p] + [datetime.now().strftime("%Y%m%d%H%M")]
    return "_".join(parts)


def ensure_file_name(
    agent_config: AgentConfig,
    platform: str,
    dashboard_name: str,
    suffix: str = ".sql",
) -> Path:
    sql_root = agent_config.path_manager.dashboard_path() / platform
    sql_root.mkdir(parents=True, exist_ok=True)
    return sql_root / f"{_build_sql_file_name(platform, dashboard_name)}{suffix}"


def build_sql_comment_lines(sql_item: SelectedSqlCandidate, dashboard_name: str) -> List[str]:
    lines = [
        f"-- Dashboard={clean_comment_text(dashboard_name or '')};",
        f"-- Chart={clean_comment_text(sql_item.chart_name or str(sql_item.chart_id))};",
    ]
    if sql_item.description:
        lines.append(f"-- Description={clean_comment_text(sql_item.description)};")
    return lines


def write_chart_sql_files(
    reference_sqls: Sequence[SelectedSqlCandidate],
    *,
    platform: str,
    dashboard_name: str,
    agent_config: AgentConfig,
) -> Optional[Path]:
    """Materialize selected chart SQL into a single ``.sql`` file.

    Returns the file path on success, ``None`` when there is nothing to
    write. Each chart's SQLs are grouped together with a leading comment
    block describing the originating dashboard and chart.
    """
    if not reference_sqls:
        return None

    target_file = ensure_file_name(agent_config, platform, dashboard_name)

    grouped: dict[str, List[SelectedSqlCandidate]] = {}
    for item in reference_sqls:
        grouped.setdefault(str(item.chart_id), []).append(item)

    with open(target_file, "w", encoding="utf-8") as target_f:
        for items in grouped.values():
            lines: List[str] = []
            for sql_item in items:
                lines.extend(build_sql_comment_lines(sql_item, dashboard_name))
                sql_text = (sql_item.sql or "").strip()
                if sql_text:
                    if not sql_text.endswith(";"):
                        sql_text += ";"
                    lines.append(sql_text)
                    lines.append("")
            if lines:
                target_f.write("\n".join(lines))
    return target_file


def write_metrics_csv(
    sqls: Sequence[SelectedSqlCandidate],
    *,
    platform: str,
    dashboard_name: str,
    agent_config: AgentConfig,
) -> Path:
    """Write a metrics CSV (``question, sql`` columns) and return its path.

    Idempotent — if the target file already exists it is left untouched
    (semantic_model and metrics streams share the same file).
    """
    target_file = ensure_file_name(agent_config, platform, dashboard_name, suffix=".csv")
    if target_file.exists():
        return target_file

    rows: List[dict[str, Any]] = []
    for sql_item in sqls:
        question = (
            f"Dashboard={clean_comment_text(dashboard_name or '')};"
            f"Chart={clean_comment_text(sql_item.chart_name or str(sql_item.chart_id))};"
        )
        if sql_item.description:
            question += f"Description={clean_comment_text(sql_item.description)};"
        rows.append({"question": question, "sql": sql_item.sql})

    with open(target_file, "w", encoding="utf-8") as target_f:
        pd.DataFrame(rows, columns=["question", "sql"]).to_csv(target_f, index=False)
    return target_file


# ── sub-agent persistence stream ──────────────────────────────────────


def stream_bi_save_subagents(
    agent_config: AgentConfig,
    *,
    sub_agent_name: str,
    description: str,
    scoped_context: ScopedContext,
    sub_agent_manager: SubAgentManager,
    cli_ref: Any = None,
) -> AsyncGenerator[ActionHistory, None]:
    """Wrap the two sub-agent yaml writes plus agentic_nodes refresh as one task group.

    Two ``SubAgentConfig`` rows are persisted (the main sub-agent and its
    ``_attribution`` companion). Each save emits a depth=1 TOOL action;
    failure yields a FAILED entry instead of raising. After both writes,
    ``agent_config.agentic_nodes`` is refreshed and the CLI's known
    sub-agent set is updated when present.
    """
    main_cfg = SubAgentConfig(
        system_prompt=sub_agent_name,
        agent_description=description,
        tools="context_search_tools,db_tools.search_table,db_tools.describe_table,db_tools.execute_sql",
        scoped_context=scoped_context,
    )
    attribution_cfg = SubAgentConfig(
        system_prompt=f"{sub_agent_name}_attribution",
        agent_description=f"Attribution analysis for {description}",
        node_class="gen_report",
        tools="semantic_tools,context_search_tools.list_subject_tree",
        scoped_context=scoped_context,
    )

    async def _inner(_mgr: ActionHistoryManager) -> AsyncGenerator[ActionHistory, None]:
        for cfg in (main_cfg, attribution_cfg):
            label = cfg.system_prompt
            try:
                sub_agent_manager.save_agent(cfg, previous_name=label)
                yield ActionHistory.create_action(
                    role=ActionRole.TOOL,
                    action_type="persist_yaml",
                    messages=f"Sub-Agent `{label}` saved",
                    input_data={"function_name": "persist_yaml", "name": label},
                    status=ActionStatus.SUCCESS,
                )
            except Exception as exc:
                logger.error("Failed to persist sub-agent %s: %s", label, exc, exc_info=True)
                yield ActionHistory.create_action(
                    role=ActionRole.TOOL,
                    action_type="persist_yaml",
                    messages=f"Sub-Agent `{label}` persist failed: {exc}",
                    input_data={"function_name": "persist_yaml", "name": label},
                    status=ActionStatus.FAILED,
                )

        # Refresh agent_config + cli's known sub-agent set.
        try:
            agents = sub_agent_manager.list_agents()
        except Exception as exc:
            logger.warning("list_agents after save failed: %s", exc)
            agents = None

        if agents is not None:
            try:
                agent_config.agentic_nodes = agents
            except Exception:
                pass
            if cli_ref is not None and getattr(cli_ref, "available_subagents", None):
                try:
                    cli_ref.available_subagents.update(name for name in agents.keys() if name != "chat")
                except Exception:
                    pass

    return as_task_subagent(
        subagent_type="save_subagents",
        description=sub_agent_name,
        inner_factory=_inner,
    )


__all__ = [
    "build_sql_comment_lines",
    "build_sub_agent_name",
    "clean_comment_text",
    "dedupe_values",
    "ensure_file_name",
    "normalize_identifier",
    "parse_subject_path_for_metrics",
    "qualify_table_names",
    "stream_bi_save_subagents",
    "write_chart_sql_files",
    "write_metrics_csv",
]
