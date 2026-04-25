# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Layer B: LLM-mode validator skill execution.

:func:`run_llm_validator` loads the given validator skill's content, assembles
a compact validation prompt (target + any Layer A pre-check context), runs a
sub-agent restricted to the read-only tool whitelist, and parses the returned
JSON report into a :class:`ValidationReport`.

The sub-agent is created inline against the parent's ``model`` instance rather
than via ``Node.new_instance`` — we don't need a full AgenticNode: no session,
no action history, no skill injection, no KB sync. Just instructions + tools +
structured output.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, List, Optional

from datus.utils.loggings import get_logger
from datus.validation.report import (
    CheckResult,
    DeliverableTarget,
    SessionTarget,
    TableTarget,
    TransferTarget,
    ValidationReport,
    describe_target,
)

if TYPE_CHECKING:
    from datus.tools.func_tool.database import DBFuncTool
    from datus.tools.skill_tools.skill_config import SkillMetadata
    from datus.tools.skill_tools.skill_registry import SkillRegistry


logger = get_logger(__name__)


# Whitelist of read-only tools exposed to validator sub-agents. Deliberately
# narrow — any write tool is explicitly excluded, as is ``SubAgentTaskTool``
# (to avoid recursive fork). See design doc §5.5.
VALIDATOR_READONLY_TOOL_NAMES = {
    # Database read tools
    "list_databases",
    "list_schemas",
    "list_tables",
    "describe_table",
    "get_table_ddl",
    "search_table",
    "read_query",
    # BI read tools (gen_dashboard validators)
    "list_dashboards",
    "get_dashboard",
    "list_charts",
    "get_chart",
    "get_chart_data",
    "list_datasets",
    "get_dataset",
    "list_bi_databases",
    # Scheduler read tools (scheduler validators). trigger_scheduler_job is
    # intentionally excluded — runtime firing is user-initiated, not automatic.
    "list_scheduler_jobs",
    "get_scheduler_job",
    "list_job_runs",
    "get_run_log",
    "list_scheduler_connections",
}


VALIDATOR_MAX_TURNS = 20


OUTPUT_CONTRACT_INSTRUCTIONS = """

## Output contract (validator hook)

After completing your analysis, emit a **single fenced JSON block** exactly
matching this schema. The hook parses only the JSON; prose outside the block
is free-form and ignored.

```json
{
  "checks": [
    {
      "name": "<short check identifier>",
      "passed": true,
      "severity": "advisory",
      "observed": {"key": "value"},
      "expected": {"key": "value"}
    }
  ],
  "blocking_issues": ["short summary string", "..."]
}
```

- ``severity`` must be ``"blocking"`` or ``"advisory"``. Prefer ``"advisory"``
  unless the violation genuinely breaks downstream consumers — the builtin
  layer already blocks on object existence and row-count invariants, so
  B-class findings default to advisory.
- ``observed`` / ``expected`` are optional dicts of whatever info helps explain the result.
- ``blocking_issues`` is a flat list of short strings naming any must-fix problems.
- If you have no findings, still emit the block with ``"checks": []``.
"""


async def run_llm_validator(
    skill: "SkillMetadata",
    registry: "SkillRegistry",
    target: DeliverableTarget,
    model: Any,
    db_func_tool: Optional["DBFuncTool"],
    precheck_context: Optional[ValidationReport] = None,
    parent_session: Optional[Any] = None,
    bi_tool: Optional[Any] = None,
    scheduler_tool: Optional[Any] = None,
) -> ValidationReport:
    """Execute a single validator skill as an isolated LLM sub-agent run.

    Args:
        skill: The validator ``SkillMetadata`` (``kind == "validator"``)
        registry: SkillRegistry used to lazy-load skill content if not cached
        target: The deliverable target to validate
        model: Parent node's LLM model (duck-typed; must expose
            ``generate_with_tools_stream``)
        db_func_tool: The parent's ``DBFuncTool`` — used to filter the
            read-only whitelist of tools to pass to the sub-agent
        precheck_context: Optional Layer A report that's already been computed
            on this target; passed into the prompt so B-class validator does
            not repeat cheap lookups.
        parent_session: Optional parent-agent session. When provided, tool-call
            and tool-result items are copied into an ephemeral in-memory
            session for the validator so it can see what the parent already
            ran (DDL text, describe_table results) without re-exploring. User
            messages and plain assistant reasoning are filtered out to avoid
            prompt injection and decision contamination.

    Returns:
        :class:`ValidationReport` tagged with ``source="skill:<name>"``.
        Malformed output or infrastructure errors surface as non-blocking
        ``warnings`` rather than exceptions, so ValidationHook can merge and
        move on.
    """
    report = ValidationReport(target=target, checks=[])

    # ── skill content ────────────────────────────────────────────────
    content = skill.content or registry.load_skill_content(skill.name)
    if not content:
        report.add_warning(
            {"type": "validator_skill_malformed", "skill_name": skill.name, "reason": "empty SKILL.md body"}
        )
        return report

    instructions = content + OUTPUT_CONTRACT_INSTRUCTIONS

    # ── tools ─────────────────────────────────────────────────────────
    tools = _select_readonly_tools(db_func_tool, bi_tool=bi_tool, scheduler_tool=scheduler_tool)

    # ── prompt ────────────────────────────────────────────────────────
    prompt = _build_prompt(target, precheck_context)

    # ── ephemeral filtered session (optional) ─────────────────────────
    # Fork parent's tool-event history into an in-memory session so the
    # validator can see what was already done without re-exploring. User text
    # and assistant reasoning are filtered out — validator stays independent
    # and immune to prompt injection via user messages.
    validator_session = await _build_validator_session(parent_session, skill.name)

    # ── execute ───────────────────────────────────────────────────────
    raw_output = ""
    try:
        async for action in model.generate_with_tools_stream(
            prompt=prompt,
            tools=tools,
            mcp_servers={},
            instruction=instructions,
            max_turns=VALIDATOR_MAX_TURNS,
            session=validator_session,
            action_history_manager=None,
            hooks=None,
            agent_name=f"validator:{skill.name}",
            interrupt_controller=None,
        ):
            out = getattr(action, "output", None)
            if isinstance(out, dict):
                raw = out.get("raw_output", "")
                if isinstance(raw, str) and raw:
                    raw_output = raw
                elif isinstance(raw, dict):
                    raw_output = json.dumps(raw)
    except Exception as e:
        logger.warning("Validator skill '%s' infrastructure error: %s", skill.name, e)
        report.add_warning({"type": "validator_runner_error", "skill_name": skill.name, "error": str(e)})
        return report

    # ── parse ─────────────────────────────────────────────────────────
    parsed = _parse_json_block(raw_output)
    if parsed is None:
        logger.warning("Validator skill '%s' returned malformed output", skill.name)
        report.add_warning(
            {
                "type": "validator_skill_malformed",
                "skill_name": skill.name,
                "reason": "no parseable JSON block in output",
                "raw_excerpt": raw_output[:200] if raw_output else "",
            }
        )
        return report

    report.checks.extend(_parse_validator_checks(parsed, skill.name))
    return report


def _parse_validator_checks(parsed: dict, skill_name: str) -> List[CheckResult]:
    """Convert a parsed validator output dict into :class:`CheckResult` list.

    Handles two forms declared by ``OUTPUT_CONTRACT_INSTRUCTIONS``:

    - ``checks: [...]`` — explicit per-check records with pass/fail & severity.
    - ``blocking_issues: [str, ...]`` — a flat list of must-fix problems.

    The second form is easy for a validator to emit (just a list of strings),
    so treat each entry as a failed blocking check. Without this, a validator
    that declares a run broken purely via ``blocking_issues`` would leave
    ``has_blocking_failure()`` at False and the hook would silently let the
    run through.
    """
    out: List[CheckResult] = []
    for raw_check in parsed.get("checks", []) or []:
        if not isinstance(raw_check, dict):
            continue
        passed = bool(raw_check.get("passed", False))
        severity_raw = raw_check.get("severity", "blocking")
        severity = severity_raw if severity_raw in ("blocking", "advisory") else "blocking"
        out.append(
            CheckResult(
                name=str(raw_check.get("name", "unnamed")),
                passed=passed,
                severity=severity,
                source=f"skill:{skill_name}",
                observed=raw_check.get("observed") if isinstance(raw_check.get("observed"), dict) else None,
                expected=raw_check.get("expected") if isinstance(raw_check.get("expected"), dict) else None,
            )
        )
    for idx, issue in enumerate(parsed.get("blocking_issues", []) or []):
        if not isinstance(issue, str) or not issue.strip():
            continue
        out.append(
            CheckResult(
                name=f"blocking_issue_{idx + 1}",
                passed=False,
                severity="blocking",
                source=f"skill:{skill_name}",
                error=issue.strip(),
            )
        )
    return out


def _select_readonly_tools(
    db_func_tool: Optional["DBFuncTool"],
    bi_tool: Optional[Any] = None,
    scheduler_tool: Optional[Any] = None,
) -> List[Any]:
    """Return the subset of each tool source's ``available_tools()`` matching
    :data:`VALIDATOR_READONLY_TOOL_NAMES`.

    DB / BI / scheduler ``available_tools()`` already exclude mutation tools
    where applicable, but we still filter by the whitelist to be safe against
    future additions. ``None`` sources are simply skipped — validators for
    pure-table subagents only need ``db_func_tool``; BI validators only need
    ``bi_tool``; etc.
    """
    allowed: List[Any] = []
    seen: set = set()
    for source in (db_func_tool, bi_tool, scheduler_tool):
        if source is None:
            continue
        try:
            tools = source.available_tools()
        except Exception as e:
            logger.warning("available_tools() failed on %s: %s", type(source).__name__, e)
            continue
        for tool in tools:
            name = getattr(tool, "name", "")
            if name in VALIDATOR_READONLY_TOOL_NAMES and name not in seen:
                allowed.append(tool)
                seen.add(name)
    return allowed


def _build_prompt(target: DeliverableTarget, precheck: Optional[ValidationReport]) -> str:
    """Construct the validator sub-agent input — compact target + precheck dump."""
    lines: List[str] = []
    if isinstance(target, TableTarget):
        lines.append("Validate the table written by the most recent tool call.")
        if target.catalog:
            lines.append(f"Catalog: {target.catalog}")
        lines.append(f"Database: {target.database}")
        if target.db_schema:
            lines.append(f"Schema: {target.db_schema}")
        lines.append(f"Table: {target.table}")
        if target.rows_affected is not None:
            lines.append(f"Rows affected (tool-reported): {target.rows_affected}")
    elif isinstance(target, TransferTarget):
        lines.append("Validate the cross-database transfer that just completed.")
        lines.append(f"Source database: {target.source.name}")
        tgt_prefix = f"{target.target.catalog}." if target.target.catalog else ""
        lines.append(f"Target: {tgt_prefix}{target.target.database}.{target.target.fqn}")
        if target.source_row_count is not None:
            lines.append(f"Source row count (tool-reported): {target.source_row_count}")
        if target.transferred_row_count is not None:
            lines.append(f"Transferred row count (tool-reported): {target.transferred_row_count}")
    elif isinstance(target, SessionTarget):
        lines.append(f"Validate the run's {len(target.targets)} deliverable(s):")
        for t in target.targets:
            if isinstance(t, TableTarget):
                prefix = f"{t.catalog}." if t.catalog else ""
                lines.append(f"- table {prefix}{t.database}.{t.fqn} (rows_affected={t.rows_affected})")
            elif isinstance(t, TransferTarget):
                prefix = f"{t.target.catalog}." if t.target.catalog else ""
                lines.append(
                    f"- transfer {t.source.name} -> {prefix}{t.target.database}.{t.target.fqn} "
                    f"(src={t.source_row_count}, tgt={t.transferred_row_count})"
                )
            else:
                lines.append(f"- {describe_target(t)}")
    else:
        lines.append(f"Validate the delivered resource: {describe_target(target)}")

    if precheck and precheck.checks:
        lines.append("")
        lines.append("Layer A pre-checks (already executed; do not repeat):")
        for c in precheck.checks:
            status = "PASS" if c.passed else "FAIL"
            obs = f" observed={c.observed}" if c.observed else ""
            lines.append(f"- [{status}] {c.name} (source={c.source}){obs}")

    lines.append("")
    lines.append(
        "Using the read-only tools available to you, execute the skill's workflow "
        "and emit the required JSON output block."
    )
    return "\n".join(lines)


async def _build_validator_session(parent_session: Optional[Any], skill_name: str) -> Optional[Any]:
    """Fork the parent's tool-event history into an ephemeral in-memory session.

    Copies only items that represent factual tool activity (tool calls issued
    by the parent, tool results, system instructions) and drops user messages
    and plain assistant text. Returns ``None`` when ``parent_session`` is
    ``None`` or filtering yields nothing — the caller passes ``None`` straight
    through to the SDK, which falls back to cold-start.
    """
    if parent_session is None:
        return None
    try:
        items = await parent_session.get_items()
    except Exception as e:
        logger.warning("Failed to read parent session for validator %s: %s", skill_name, e)
        return None
    filtered = _filter_tool_events(items)
    if not filtered:
        return None
    try:
        from datus.models.session_manager import AdvancedSQLiteSession
    except Exception as e:
        logger.warning("AdvancedSQLiteSession unavailable; skipping validator session fork: %s", e)
        return None
    import uuid as _uuid

    ephemeral_id = f"validator-{skill_name}-{_uuid.uuid4().hex[:8]}"
    try:
        session = AdvancedSQLiteSession(session_id=ephemeral_id, db_path=":memory:", create_tables=True)
        await session.add_items(filtered)
    except Exception as e:
        logger.warning("Failed to populate validator session for %s: %s", skill_name, e)
        return None
    logger.debug(
        "validator %s session forked with %d tool events (dropped %d non-tool items)",
        skill_name,
        len(filtered),
        len(items) - len(filtered),
    )
    return session


def _filter_tool_events(items: List[Any]) -> List[Any]:
    """Keep only tool-call assistant items and tool-result items.

    - Drops ``role: user`` (prompt-injection risk via user text)
    - Drops ``role: assistant`` messages without ``tool_calls`` (reasoning
      text — contamination risk)
    - Drops ``role: system`` (validator has its own ``instructions``)
    - Keeps ``role: tool`` (tool-result payloads — these are facts)
    - Keeps ``role: assistant`` WITH ``tool_calls`` but strips text ``content``
      so only the tool-call signature remains
    """
    kept: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role == "tool":
            kept.append(item)
        elif role == "assistant" and item.get("tool_calls"):
            # Strip text reasoning; keep only the tool_calls signature
            clean = {k: v for k, v in item.items() if k != "content"}
            kept.append(clean)
    return kept


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_JSON_RE = re.compile(r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})", re.DOTALL)


def _parse_json_block(raw: str) -> Optional[dict]:
    """Extract the first JSON object from the sub-agent output.

    Prefers fenced ```json blocks; falls back to first bare ``{…}``.
    """
    if not raw:
        return None
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback — look for a JSON object anywhere in the text. Accept either
    # the canonical ``checks`` form OR a bare ``blocking_issues`` payload:
    # ``_parse_validator_checks`` handles both, and downgrading a
    # ``{"blocking_issues": [...]}`` emission to a malformed-output warning
    # would silently let a would-be-blocking run through.
    for match in _BARE_JSON_RE.findall(raw):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and ("checks" in obj or "blocking_issues" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None
