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
    "list_databases",
    "list_schemas",
    "list_tables",
    "describe_table",
    "get_table_ddl",
    "search_table",
    "read_query",
}


VALIDATOR_MAX_TURNS = 10


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
      "severity": "blocking",
      "observed": {"key": "value"},
      "expected": {"key": "value"}
    }
  ],
  "blocking_issues": ["short summary string", "..."]
}
```

- ``severity`` must be ``"blocking"`` or ``"advisory"``.
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
    tools = _select_readonly_tools(db_func_tool)

    # ── prompt ────────────────────────────────────────────────────────
    prompt = _build_prompt(target, precheck_context)

    # ── execute ───────────────────────────────────────────────────────
    raw_output = ""
    try:
        async for action in model.generate_with_tools_stream(
            prompt=prompt,
            tools=tools,
            mcp_servers={},
            instruction=instructions,
            max_turns=VALIDATOR_MAX_TURNS,
            session=None,
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

    for raw_check in parsed.get("checks", []) or []:
        if not isinstance(raw_check, dict):
            continue
        passed = bool(raw_check.get("passed", False))
        severity_raw = raw_check.get("severity", "blocking")
        severity = severity_raw if severity_raw in ("blocking", "advisory") else "blocking"
        report.checks.append(
            CheckResult(
                name=str(raw_check.get("name", "unnamed")),
                passed=passed,
                severity=severity,
                source=f"skill:{skill.name}",
                observed=raw_check.get("observed") if isinstance(raw_check.get("observed"), dict) else None,
                expected=raw_check.get("expected") if isinstance(raw_check.get("expected"), dict) else None,
            )
        )
    return report


def _select_readonly_tools(db_func_tool: Optional["DBFuncTool"]) -> List[Any]:
    """Return the subset of ``db_func_tool.available_tools()`` in the read-only whitelist.

    ``DBFuncTool.available_tools()`` already excludes mutation tools
    (``execute_ddl`` / ``execute_write`` / ``transfer_query_result`` are
    registered separately by nodes that need them), but we still filter by
    :data:`VALIDATOR_READONLY_TOOL_NAMES` to be safe against future additions.
    """
    if db_func_tool is None:
        return []
    allowed: List[Any] = []
    for tool in db_func_tool.available_tools():
        name = getattr(tool, "name", "")
        if name in VALIDATOR_READONLY_TOOL_NAMES:
            allowed.append(tool)
    return allowed


def _build_prompt(target: DeliverableTarget, precheck: Optional[ValidationReport]) -> str:
    """Construct the validator sub-agent input — compact target + precheck dump."""
    lines: List[str] = []
    if isinstance(target, TableTarget):
        lines.append("Validate the table written by the most recent tool call.")
        lines.append(f"Database: {target.database}")
        if target.db_schema:
            lines.append(f"Schema: {target.db_schema}")
        lines.append(f"Table: {target.table}")
        if target.rows_affected is not None:
            lines.append(f"Rows affected (tool-reported): {target.rows_affected}")
    elif isinstance(target, TransferTarget):
        lines.append("Validate the cross-database transfer that just completed.")
        lines.append(f"Source database: {target.source.name}")
        lines.append(f"Target: {target.target.database}.{target.target.fqn}")
        if target.source_row_count is not None:
            lines.append(f"Source row count (tool-reported): {target.source_row_count}")
        if target.transferred_row_count is not None:
            lines.append(f"Transferred row count (tool-reported): {target.transferred_row_count}")
    elif isinstance(target, SessionTarget):
        lines.append(f"Validate the run's {len(target.targets)} deliverable(s):")
        for t in target.targets:
            if isinstance(t, TableTarget):
                lines.append(f"- table {t.database}.{t.fqn} (rows_affected={t.rows_affected})")
            elif isinstance(t, TransferTarget):
                lines.append(
                    f"- transfer {t.source.name} -> {t.target.database}.{t.target.fqn} "
                    f"(src={t.source_row_count}, tgt={t.transferred_row_count})"
                )

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
    # Fallback — look for a JSON object anywhere in the text
    for match in _BARE_JSON_RE.findall(raw):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "checks" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None
