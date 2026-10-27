# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Semantic Function Tools

Provides unified interface to semantic layer services through runtimes.
All public semantic tools require a successfully initialized semantic runtime.
"""

import csv
import hashlib
import io
import itertools
import json
import re
import secrets
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple

from agents import Tool
from pydantic import BaseModel

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG, normalize_metric_name
from datus.tools.func_tool.base import (
    FuncToolResult,
    normalize_null,
    tool_schema,
    trans_to_function_tool,
)
from datus.tools.func_tool.generation_evidence import GenerationEvidence
from datus.tools.semantic_tools.dosi import DosiConfig, DosiRuntime
from datus.tools.semantic_tools.dosi.engine import has_model_files
from datus.tools.semantic_tools.models import (
    AttributionRequest,
    AttributionResult,
    AttributionWindow,
)
from datus.tools.semantic_tools.paging import (
    METRIC_CATALOG_MAX_PAGES,
    METRIC_CATALOG_PAGE_SIZE,
    metric_catalog_paging,
)
from datus.utils.compress_utils import DataCompressor
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.mcp_decorators import mcp_tool, mcp_tool_class
from datus.utils.query_result_text import omitted_rows_line

logger = get_logger(__name__)


_IN_LIST = re.compile(r"(?<![\w.])([A-Za-z_]\w*\.[A-Za-z_]\w*)\s+IN\s*\(([^()]*)\)", re.IGNORECASE)
_IN_ITEM = re.compile(r"\s*(?:'((?:[^']|'')*)'|(-?\d+(?:\.\d+)?))\s*(?:,|$)")


def _in_list_conditions(sql: Optional[str]) -> List[Tuple[str, List[str]]]:
    """``dataset.field IN (literal, ...)`` conditions of a condition string.

    ``NOT IN`` and lists holding anything but string or number literals are
    left out: only a plain list of values can be checked value by value.
    """
    found: List[Tuple[str, List[str]]] = []
    for match in _IN_LIST.finditer(sql or ""):
        if sql[: match.start()].rstrip().upper().endswith("NOT"):
            continue
        body = match.group(2)
        values: List[str] = []
        position = 0
        while position < len(body):
            item = _IN_ITEM.match(body, position)
            if not item or item.end() == position:
                values = []
                break
            values.append(item.group(1).replace("''", "'") if item.group(1) is not None else item.group(2))
            position = item.end()
        if values:
            found.append((match.group(1), values))
    return found


def _value_key(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _result_column_for(field: str, columns: List[str]) -> Optional[str]:
    """The result column of ``field``: its qualified name, else its bare name when only one column has it."""
    if field in columns:
        return field
    short = field.split(".")[-1]
    matches = [column for column in columns if column == short]
    return matches[0] if len(matches) == 1 else None


def _filters_on_metric(where: Optional[str], metrics: List[str]) -> bool:
    """Whether ``where`` names a selected metric, which filters the result rows after aggregation."""
    return any(
        re.search(rf"(?<![\w.]){re.escape(metric)}(?![\w.])", where or "")
        for metric in metrics
        if isinstance(metric, str)
    )


NO_METRICS_PRESENT_MESSAGE = "No metrics present in the model."


def _normalize_metric_metadata(raw) -> dict:
    """Keep runtime-provided metric metadata only when it is tool-safe."""
    if not isinstance(raw, dict):
        return {}

    safe_metadata = {}
    for key, value in raw.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        safe_metadata[str(key)] = value
    return safe_metadata


def _metric_list_row(metric: Any, subject_path: Optional[List[str]] = None) -> dict:
    """One ``list_metrics`` row: what a caller needs to pick the metric.

    How the metric is computed (kind, derivation, time axis, datasets) is left
    to ``get_metric``, which the caller reads for every metric it queries: on a
    real model those keys were a third of a listing that is mostly never
    followed up.

    ``subject_path`` is the knowledge-base navigation path, which the caller owns
    because the KB — not the runtime — is the source of truth for it. An runtime
    may publish a ``path`` of its own; it is deliberately ignored so that the
    navigation path a caller reads here is the one the subject tree shows.
    ``unit``/``format`` stay when set: nothing else shows them.
    """
    row = {"name": getattr(metric, "name", None), "description": getattr(metric, "description", None)}
    for key in ("unit", "format"):
        value = getattr(metric, key, None)
        if value is not None:
            row[key] = value
    if subject_path:
        row["path"] = subject_path
    return row


def render_metric_list(
    rows: List[Mapping[str, Any]], *, total: Optional[int], has_more: Optional[bool], next_offset: Optional[int]
) -> str:
    """A list_metrics page as text: metrics grouped under their subject path,
    ``- name -- description`` each, and how to fetch the rest when there is more."""
    if not rows:
        return "no metrics"
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(" / ".join(row.get("path") or []), []).append(row)
    lines: List[str] = []
    for path, members in groups.items():
        if path:
            lines.append(f"{path}:")
        for row in members:
            text = f"- {row.get('name')}"
            if row.get("description"):
                text += f" -- {_one_line(row['description'])}"
            marks = [f"{key} {row[key]}" for key in ("unit", "format") if row.get(key) is not None]
            if marks:
                text += f" [{'; '.join(marks)}]"
            lines.append(text)
    if has_more and next_offset is not None:
        shown = f"{len(rows)} of {total}" if total is not None else f"{len(rows)}"
        lines.append(f"more: {shown} metrics shown; call again with offset={next_offset}")
    return "\n".join(lines)


def _normalize_dataset_names(raw: Any) -> List[str]:
    """Dataset names a runtime reports for a metric, as a list of non-empty strings.

    A lone string names one dataset; iterating it would yield characters. Only
    string entries are kept — coercing anything else would invent names like
    "None" that a policy could match on.
    """
    if isinstance(raw, str):
        name = raw.strip()
        return [name] if name else []
    if isinstance(raw, (list, tuple, set)):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return []


# An expression longer than this is not written out in get_dataset: the field
# shows the columns it reads instead (``[from ...]``). The longest computed
# fields of a real model run to thousands of characters; the median is 45.
_DATASET_EXPR_FOLD = 120


def _one_line(text: Any) -> str:
    """``text`` on one line: an authored expression keeps its tokens, not its
    line breaks and indentation."""
    return " ".join(str(text).split())


def _render_param(param: Mapping[str, Any]) -> str:
    """``name type = default [domain] -- description [declared_on metric]``."""
    line = f"{param.get('name')} {param.get('type')} = {param.get('default')}"
    if param.get("allowed"):
        line += f" [one of: {', '.join(str(v) for v in param['allowed'])}]"
    elif param.get("min") is not None or param.get("max") is not None:
        lo = param.get("min") if param.get("min") is not None else ""
        hi = param.get("max") if param.get("max") is not None else ""
        line += f" [{lo}..{hi}]"
    if param.get("description"):
        line += f" -- {_one_line(param['description'])}"
    if param.get("declared_on"):
        line += f" [declared_on {param['declared_on']}]"
    return line


def _render_window(window: Mapping[str, Any]) -> str:
    """The engine's window description on one line: family and function, then
    the keys that say what it compares."""
    parts = [str(window.get("family") or ""), str(window.get("function") or "")]
    for key in ("count", "grain", "direction", "n", "buckets", "preceding", "reset", "scope"):
        value = window.get(key)
        if value is not None and value is not False:
            parts.append(f"{key} {value}")
    order = window.get("order")
    if isinstance(order, Mapping) and order.get("by"):
        parts.append(f"order by {order.get('by')} {order.get('direction') or ''}".rstrip())
    partition = window.get("partition")
    if isinstance(partition, Mapping) and partition.get("mode"):
        text = f"partition {partition['mode']}"
        if partition.get("exclude"):
            text += f" excluding {', '.join(partition['exclude'])}"
        parts.append(text)
    return ", ".join(part for part in parts if part)


def render_metric_detail(detail: Mapping[str, Any], subject_path: Optional[List[str]] = None) -> str:
    """One metric as get_metric shows it: one ``key: value`` line per fact, a
    fact the metric does not have left out. ``expr`` names other metrics and
    fields; get_metric and get_dataset describe those.
    """
    lines: List[str] = []

    def line(key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")

    line("description", _one_line(detail["description"]) if detail.get("description") else None)
    line("path", " / ".join(subject_path) if subject_path else None)
    if detail.get("kind") and detail.get("kind") != "aggregate":
        line("kind", detail["kind"])
    line("expr", _one_line(detail["expr"]) if detail.get("expr") else None)
    line("datasets", ", ".join(detail.get("datasets") or []))
    dimension_datasets = []
    for dataset in detail.get("dimension_datasets") or []:
        fields = dataset.get("fields")
        dimension_datasets.append(f"{dataset['name']}({', '.join(fields)})" if fields else str(dataset["name"]))
    line("dimension_datasets", ", ".join(dimension_datasets))
    if isinstance(detail.get("window"), Mapping):
        line("window", _render_window(detail["window"]))
    across = detail.get("compares_across")
    line("compares_across", ", ".join(across) if isinstance(across, list) else across)
    params = [p for p in detail.get("params") or [] if isinstance(p, Mapping)]
    if params:
        lines.append("params:")
        lines.extend(f"- {_render_param(p)}" for p in params)
    line("required", ", ".join(detail.get("required_dimensions") or []))
    if detail.get("requires_time_axis"):
        line("requires_time_axis", "true")
    grains = detail.get("time_granularities") or []
    line("time", f"metric_time ({', '.join(grains)})" if grains else "none")
    return "\n".join(lines)


def _render_query_warning(warning: Any) -> str:
    """One engine warning on one line: ``<level> <code>: <message>``, then the
    part and condition when the message does not name them, then the hint."""
    if not isinstance(warning, Mapping):
        return f"warning: {_one_line(warning)}"
    message = _one_line(warning.get("message") or "")
    label = " ".join(str(value) for value in (warning.get("level") or "warning", warning.get("code")) if value)
    text = f"{label}: {message}"
    for key in ("part", "condition"):
        value = warning.get(key)
        if value and str(value) not in message:
            text += f" ({key}: {_one_line(value)})"
    if warning.get("hint"):
        text += f"; hint: {_one_line(warning['hint'])}"
    return text


def _render_query_output(output: Mapping[str, Any]) -> str:
    """A result column named after a parameter binding: which metric, bound how."""
    bound = ", ".join(
        f"{name}={value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
        for name, value in (output.get("params") or {}).items()
    )
    text = f"column {output.get('column')}: {output.get('metric')}"
    return f"{text} with {bound}" if bound else text


def render_dataset_detail(detail: Mapping[str, Any]) -> str:
    """One dataset as get_dataset shows it: a header line, one line per field —
    ``- name`` for a column, ``- expr AS name`` for a computed or fixed field —
    and the relationships seen from this dataset.
    """
    lines: List[str] = []
    header = []
    if detail.get("description"):
        header.append(_one_line(detail["description"]))
    if detail.get("source") and detail.get("source") != detail.get("name"):
        header.append(f"source: {detail['source']}")
    if detail.get("primary_time_dimension"):
        header.append(f"time: {detail['primary_time_dimension']}")
    if detail.get("primary_key"):
        header.append(f"pk: {', '.join(detail['primary_key'])}")
    if header:
        lines.append(" | ".join(header))
    for field in detail.get("fields") or []:
        name = field.get("name")
        expr = _one_line(field["expr"]) if field.get("expr") else None
        fixed = bool(expr) and expr.startswith("fixed(")
        marks = []
        if expr and not fixed and len(expr) > _DATASET_EXPR_FOLD:
            text = f"- {name}"
            marks.append(f"from {', '.join(field.get('reads') or [])}")
        elif expr:
            text = f"- {expr} AS {name}"
        else:
            text = f"- {name}"
        if field.get("is_time"):
            marks.append(f"time {field['time_format']}" if field.get("time_format") else "time")
        elif field.get("time_format"):
            marks.append(f"format {field['time_format']}")
        if field.get("is_dimension") is False:
            marks.append("not a dimension")
        if field.get("datasets"):
            marks.append(f"reads {', '.join(field['datasets'])}")
        if field.get("time_axis"):
            marks.append(f"time_axis {field['time_axis']}")
        if marks:
            text += f" [{'; '.join(marks)}]"
        if field.get("description"):
            text += f" -- {_one_line(field['description'])}"
        lines.append(text)
    relationships = detail.get("relationships") or []
    if relationships:
        lines.append("relationships:")
        for rel in relationships:
            on = ", ".join(f"{mine}={theirs}" for mine, theirs in rel.get("columns") or [])
            prefix = f"{rel['name']}: " if rel.get("name") else ""
            lines.append(f"- {prefix}{rel.get('cardinality')} {rel.get('to')} on {on}")
    return "\n".join(lines)


def _normalize_pagination_bound(value: Any, *, default: int, minimum: int) -> int:
    """Coerce an LLM-supplied paging bound to an int.

    A tool schema declaring ``int`` does not stop a model from sending ``"200"``,
    and runtimes use these values in arithmetic and slicing — a string reaches
    them as ``TypeError: slice indices must be integers``, which surfaces as a
    failed call the caller cannot act on. Anything unusable falls back to the
    default rather than failing the listing.
    """
    value = normalize_null(value)
    if isinstance(value, bool) or value is None:
        return default
    try:
        bound = int(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric paging bound %r; using %d.", value, default)
        return default
    return max(minimum, bound)


def _normalize_name_list(value) -> List[str]:
    """Normalize LLM-provided string/list arguments into a clean list of names."""
    value = normalize_null(value)
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]

    names = []
    for candidate in candidates:
        candidate = normalize_null(candidate)
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            names.append(text)
    return names


def _normalize_optional_path(value) -> Optional[List[str]]:
    """Normalize optional subject paths and drop null placeholders."""
    names = _normalize_name_list(value)
    return names or None


# Defaults for `metric_datasets` paging. The page is large because Dosi may rebuild
# its whole catalog per request, making each extra page cost a full pass.
_METRIC_DATASETS_PAGE_SIZE = METRIC_CATALOG_PAGE_SIZE
_METRIC_DATASETS_MAX_PAGES = METRIC_CATALOG_MAX_PAGES

_TIME_GRANULARITY_ORDER = ("day", "week", "month", "quarter", "year")
_TIME_GRANULARITIES = set(_TIME_GRANULARITY_ORDER)

# Token budget for a ``query_metrics`` payload, well above the ``DataCompressor``
# default of 1024. A metric query returns aggregated rows whose width the caller
# chose explicitly through ``metrics``/``dimensions``: 16 metrics over 32 grouped
# rows is ~2.4k tokens, which the old budget cut down by dropping metric columns —
# losing the requested metric to save a few hundred tokens. Row compression still
# applies above this budget, where result size is driven by dimension cardinality
# rather than by the caller's column choice.
METRIC_RESULT_TOKEN_BUDGET = 8000

# Default page for ``list_metrics``. One line per metric makes a whole catalog
# cheap — 338 metrics with Chinese descriptions is ~12.6k tokens — so the default
# page covers most models in one call.
_LIST_METRICS_DEFAULT_LIMIT = 200


def extract_time_query_capabilities(raw_dimensions) -> Dict[str, Any]:
    """Extract the metric-level time contract carried by the runtime's dimensions."""
    candidates = []
    for dimension in raw_dimensions or []:
        if isinstance(dimension, dict):
            name = dimension.get("name")
            is_primary_time = dimension.get("is_primary_time")
            raw_granularities = dimension.get("time_granularities")
        else:
            name = getattr(dimension, "name", None)
            is_primary_time = getattr(dimension, "is_primary_time", None)
            raw_granularities = getattr(dimension, "time_granularities", None)

        granularities = {
            str(granularity).strip().lower()
            for granularity in _normalize_name_list(raw_granularities)
            if str(granularity).strip().lower() in _TIME_GRANULARITIES
        }
        ordered_granularities = [granularity for granularity in _TIME_GRANULARITY_ORDER if granularity in granularities]
        if name and (is_primary_time or ordered_granularities):
            candidates.append(
                (
                    bool(is_primary_time),
                    str(name),
                    ordered_granularities,
                )
            )

    if not candidates:
        return {"time_dimension": None, "time_granularities": []}

    _, time_dimension, time_granularities = next(
        (candidate for candidate in candidates if candidate[0]),
        candidates[0],
    )
    return {
        "time_dimension": time_dimension,
        "time_granularities": time_granularities,
    }


def _serialize_validation_issue(issue) -> dict:
    if hasattr(issue, "model_dump"):
        issue_data = issue.model_dump(mode="json")
    else:
        issue_data = {"severity": "error", "message": str(issue)}

    severity = issue_data.get("severity")
    if severity is not None:
        issue_data["severity"] = str(severity).lower()
    return issue_data


def _is_no_metrics_present_issue(issue: dict) -> bool:
    message = str(issue.get("message") or "")
    return NO_METRICS_PRESENT_MESSAGE in message


def _validation_has_errors(issues: List[dict]) -> bool:
    return any(str(issue.get("severity") or "").lower() == "error" for issue in issues)


class _CompactValidationIssue(BaseModel):
    """Bounded validation issue returned by the semantic tool."""

    severity: str
    message: str
    location: Any | None = None


def _compact_validation_issues(
    issues: List[dict],
    *,
    limit: int = 8,
    message_limit: int = 600,
) -> List[_CompactValidationIssue]:
    """Keep validation tool output bounded while full details remain in logs."""
    compact: List[_CompactValidationIssue] = []
    for issue in issues[:limit]:
        message = str(issue.get("message") or "").strip()
        if len(message) > message_limit:
            message = f"{message[:message_limit]}... [truncated]"
        compact.append(
            _CompactValidationIssue(
                severity=str(issue.get("severity") or "error").lower(),
                message=message,
                location=issue.get("location"),
            )
        )
    if len(issues) > limit:
        compact.append(
            _CompactValidationIssue(
                severity="warning",
                message=f"{len(issues) - limit} additional validation issue(s) omitted; see logs for details.",
            )
        )
    return compact


def _format_validation_error(issues: List[dict]) -> str:
    count = len(issues)
    if count == 0:
        return "0 validation errors"

    messages = []
    for issue in issues[:3]:
        message = str(issue.get("message") or "").strip()
        if message:
            messages.append(message)

    if not messages:
        return f"{count} validation errors"

    suffix = f"; ... {count - len(messages)} more" if count > len(messages) else ""
    return f"{count} validation errors: {'; '.join(messages)}{suffix}"


def _run_async(coro):
    """
    Run async coroutine safely, handling both sync and async contexts.

    Delegates to the centralized run_async utility which handles:
    - Deadlock prevention for nested calls
    - Proper event loop management
    - Timeout support
    - Improved error handling

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    from datus.utils.async_utils import run_async

    return run_async(coro)


@mcp_tool_class(
    name="semantic_tool",
    availability_property="has_semantic_tools",
)
class SemanticTools:
    """Function tool wrapper for semantic layer operations."""

    permission_category: str = "semantic_tools"

    MAX_QUERY_METRICS_RESULT_CACHE_SIZE = 100

    @classmethod
    def all_tools_name(cls) -> List[str]:
        """Return list of all tool method names for wizard display."""
        return [
            "list_metrics",
            "get_metric",
            "get_dataset",
            "query_metrics",
            "validate_semantic",
            "attribution_analyze",
            *cls.ONTOLOGY_TOOL_NAMES,
        ]

    @classmethod
    def create_dynamic(cls, agent_config: AgentConfig, sub_agent_name: Optional[str] = None) -> "SemanticTools":
        """Create a Dosi-backed SemanticTools instance for dynamic MCP mode."""
        return cls(agent_config, sub_agent_name=sub_agent_name)

    @classmethod
    def create_static(
        cls,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> "SemanticTools":
        """Create a SemanticTools instance for static MCP mode (``database_name`` is unused)."""
        return cls.create_dynamic(agent_config, sub_agent_name=sub_agent_name)

    def __init__(
        self,
        agent_config: AgentConfig,
        sub_agent_name: Optional[str] = None,
        generation_evidence: Optional[GenerationEvidence] = None,
        runtime_db_context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        warehouse_dry_run_provider: Optional[Callable[[str], Mapping[str, Any]]] = None,
        semantic_model_path_provider: Optional[Callable[[], Optional[str]]] = None,
        semantic_metric_names_provider: Optional[Callable[[], Optional[List[str]]]] = None,
        ontology_enabled: bool = False,
    ):
        """
        Initialize semantic function tool.

        Args:
            agent_config: Agent configuration
            sub_agent_name: Optional sub-agent name for scoped storage
            generation_evidence: Optional shared tracker for validate_semantic and query_metrics(dry_run=True)
                publish-gate evidence.
            runtime_db_context_provider: Optional callback that returns the per-turn datasource/catalog/database/schema
                context used to initialize the semantic runtime.
            warehouse_dry_run_provider: Optional host callback that validates
                runtime-compiled SQL against the active warehouse.
            semantic_model_path_provider: Optional callback that returns the exact
                request-local semantic model selected for authoring or validation.
            semantic_metric_names_provider: Optional callback returning the
                present touched metrics that require compiled publish evidence.
            ontology_enabled: Load the project's Apache Ossie ontologies
                (``subject/ontology/``) through the mapping documents that
                bind them to this datasource's models and expose the ontology tools. Off for every agent that
                does not ask for it. Mapping documents there say which model
                each ontology binds.
        """
        self.agent_config = agent_config
        self.sub_agent_name = sub_agent_name
        self.generation_evidence = generation_evidence
        self._runtime_db_context_provider = runtime_db_context_provider
        self._warehouse_dry_run_provider = warehouse_dry_run_provider
        self._semantic_model_path_provider = semantic_model_path_provider
        self._semantic_metric_names_provider = semantic_metric_names_provider
        self.ontology_enabled = bool(ontology_enabled)
        self._runtime_db_context_static: Dict[str, str] = {}
        self._runtime_db_context_static_set = False

        # The semantic runtime remains the source of executable metric definitions.
        # list_metrics reads the KB only for navigation paths shared with
        # ContextSearchTools.list_subject_tree; ContextSearchTools owns RAG discovery.
        self.metric_rag = MetricRAG(agent_config, sub_agent_name)
        # The token budget of an executed result's preview, and of a dry run's data.
        self.compressor = DataCompressor(
            model_name=agent_config.active_model().model,
            token_threshold=METRIC_RESULT_TOKEN_BUDGET,
        )
        self._query_metrics_result_cache: OrderedDict[str, dict] = OrderedDict()

        # Lazy load runtime and attribution tool
        self._runtime: Optional[DosiRuntime] = None
        self._runtime_load_error: Optional[str] = None
        self._runtime_context_key: Optional[Tuple[str, ...]] = None

    @staticmethod
    def _query_data_row_count(data: Any) -> int:
        if data is None:
            return 0
        if hasattr(data, "num_rows"):
            try:
                return int(data.num_rows)
            except (TypeError, ValueError):
                return 0
        if hasattr(data, "shape"):
            try:
                return int(data.shape[0])
            except (TypeError, ValueError, IndexError):
                return 0
        try:
            return len(data)
        except TypeError:
            return 0

    @staticmethod
    def _query_data_to_csv(columns: List[str], data: Any) -> str:
        if data is None:
            return ""

        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        if hasattr(data, "to_csv"):
            return data.to_csv(index=False)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)
        for row in data or []:
            if isinstance(row, dict):
                writer.writerow([row.get(column, "") for column in columns])
            elif isinstance(row, (list, tuple)):
                writer.writerow(row)
            else:
                writer.writerow([row])
        return buf.getvalue()

    def _cache_query_metrics_result(
        self, columns: List[str], data: Any, metadata: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        if data is None:
            return None
        full_csv = self._query_data_to_csv(columns, data)
        if not full_csv:
            return None

        # Random, not a counter: ``get_query_metrics_result`` hands a cached result
        # to whoever names its key, and on the agent's dynamic MCP server one
        # instance serves every client of a datasource.
        cache_key = f"query_metrics:{secrets.token_urlsafe(12)}"
        self._query_metrics_result_cache[cache_key] = {
            "columns": list(columns),
            "csv": full_csv,
            "row_count": self._query_data_row_count(data),
            # The rows themselves, for ``get_query_metrics_result`` to slice a page
            # out of: re-parsing ``csv`` would cost every row before the offset and
            # trip ``csv.field_size_limit`` on a large cell. The compressor copies
            # before it changes anything, so this is still the full result.
            "data": data,
            # What the result text summarizes, kept whole for code that checks the
            # result: parameter bindings (``outputs``), time columns, warnings.
            "metadata": dict(metadata or {}),
        }
        while len(self._query_metrics_result_cache) > self.MAX_QUERY_METRICS_RESULT_CACHE_SIZE:
            self._query_metrics_result_cache.popitem(last=False)
        return cache_key

    def get_cached_query_metrics_result(self, cache_key: str) -> Optional[dict]:
        return self._query_metrics_result_cache.get(cache_key)

    # One page has to fit an MCP response; the cached result is bounded only by the query.
    MAX_QUERY_METRICS_RESULT_PAGE = 1000
    # ...and a row cap alone does not bound it: a few wide cells can make 1000 rows
    # any size. A page over this many characters is cut down to fewer rows.
    MAX_QUERY_METRICS_RESULT_PAGE_CHARS = 1_000_000

    # MCP only, deliberately not in ``available_tools``: a node's model is told the
    # full result is used for its final output, and paging it into the context
    # would be the opposite of the compression.
    @mcp_tool()
    def get_query_metrics_result(self, result_id: str, offset: int = 0, limit: int = 500) -> FuncToolResult:
        """
        Read the full rows behind a ``query_metrics`` or ``select_records`` result, one page at a time.

        An executed (not dry-run) query returns a preview of its rows and a
        ``result_id`` line naming the complete result it kept. Use this when the
        preview is not enough. It runs no query, and only knows results returned
        to this caller.

        Args:
            result_id: The ``result_id`` an earlier ``query_metrics`` or ``select_records`` call returned.
            offset: Index of the first row to return, 0-based.
            limit: Maximum number of rows to return, at most 1000. A page is also
                cut to fewer rows when its CSV would pass about 1,000,000
                characters; ``returned`` says how many came back, so continue
                from ``offset + returned``.

        Returns:
            result = {"result_id", "columns", "csv" (header plus this page's rows),
            "row_count" (rows in the whole result), "offset", "returned", "has_more"}
        """
        result_id = str(result_id or "").strip()
        cached = self.get_cached_query_metrics_result(result_id)
        if cached is None:
            return FuncToolResult(success=0, error=f"Unknown or expired query_metrics result_id: {result_id}")
        offset = max(int(offset or 0), 0)
        limit = min(max(int(limit or 0), 1), self.MAX_QUERY_METRICS_RESULT_PAGE)

        # Serialized with the same writer as the full ``csv``, so a page reads
        # exactly like the matching rows of it. Halved until it fits the character
        # budget, but never below one row: paging has to advance past a row that
        # is on its own wider than the budget.
        columns = list(cached["columns"])
        page_rows = limit
        while True:
            page = self._slice_query_data(cached["data"], offset, page_rows)
            page_csv = self._query_data_to_csv(columns, page)
            if len(page_csv) <= self.MAX_QUERY_METRICS_RESULT_PAGE_CHARS or page_rows == 1:
                break
            page_rows = max(1, page_rows // 2)
        returned = self._query_data_row_count(page)
        return FuncToolResult(
            result={
                "result_id": result_id,
                "columns": columns,
                "csv": page_csv,
                "row_count": cached["row_count"],
                "offset": offset,
                "returned": returned,
                "has_more": offset + returned < cached["row_count"],
            }
        )

    @staticmethod
    def _slice_query_data(data: Any, offset: int, limit: int) -> Any:
        """Rows ``[offset, offset + limit)`` of a query result, in its own shape."""
        if hasattr(data, "num_rows") and callable(getattr(data, "slice", None)):  # pyarrow.Table
            return data.slice(offset, limit)
        if hasattr(data, "iloc"):  # pandas.DataFrame
            return data.iloc[offset : offset + limit]
        if isinstance(data, (list, tuple)):
            return data[offset : offset + limit]
        return list(itertools.islice(data, offset, offset + limit))

    #: Rows kept at each end of a long result, most first: a preview over the
    #: budget falls back to the next.
    _PREVIEW_EDGE_ROWS = (10, 5, 2, 1)
    #: Longest text value tried in turn once one row at each end is still over.
    _PREVIEW_CELL_CHARS = (200, 50)

    def _render_query_result(
        self,
        result_id: Optional[str],
        columns: List[str],
        data: Any,
        *,
        warnings: Optional[List[Any]] = None,
        notes: Optional[List[str]] = None,
        outputs: Optional[List[Any]] = None,
        row_cap: Optional[int] = None,
    ) -> str:
        """An executed result as the agent reads it (``datus.utils.query_result_text``
        has the layout). Only what there is to report gets a line: the compiled
        SQL runs to tens of thousands of characters beside rows that are already
        here, and a column named after its metric needs no ``column`` line."""
        total = self._query_data_row_count(data)
        preview, cut = self._preview_csv(columns, data, total)
        rows = f"rows: {total}"
        if row_cap is not None and total >= row_cap:
            rows += f" (reached the {row_cap}-row cap; there may be more)"
        if cut:
            rows += f"; text values longer than {cut} characters are cut and end in …"
        lines = [f"result_id: {result_id}"] if result_id else []
        lines.append(rows)
        lines.extend(_render_query_warning(warning) for warning in warnings or [])
        lines.extend(f"note: {_one_line(note)}" for note in notes or [])
        lines.extend(
            _render_query_output(output)
            for output in outputs or []
            if isinstance(output, Mapping) and (output.get("column") != output.get("metric") or output.get("params"))
        )
        return "\n".join(lines) + "\n\n" + preview

    def _preview_csv(self, columns: List[str], data: Any, total: int) -> Tuple[str, Optional[int]]:
        """The rows under a CSV header, fitted to the token budget: fewer rows
        at each end of a long result, then long text values cut; never fewer
        columns, as each was asked for by name. Returns the CSV and the length
        values were cut to, None when none was. Past the last step the preview
        goes out over the budget."""
        budget = self.compressor.max_tolerable_tokens
        shown = None
        for edge in self._PREVIEW_EDGE_ROWS:
            if shown is not None and total <= 2 * edge:
                continue  # every row again: the same text
            shown = edge
            text = self._edge_rows_csv(columns, data, total, edge)
            if self.compressor.count_tokens(text) <= budget:
                return text, None
        for max_chars in self._PREVIEW_CELL_CHARS:
            text = self._edge_rows_csv(columns, data, total, shown, max_chars)
            if self.compressor.count_tokens(text) <= budget:
                break
        return text, max_chars

    def _edge_rows_csv(
        self, columns: List[str], data: Any, total: int, edge: int, max_chars: Optional[int] = None
    ) -> str:
        """Every row when there are at most ``2 * edge``; else the first and last
        ``edge`` around the omitted-rows line."""
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(columns)
        if total <= 2 * edge:
            writer.writerows(self._row_values(columns, data, max_chars))
        else:
            writer.writerows(self._row_values(columns, self._slice_query_data(data, 0, edge), max_chars))
            buf.write(omitted_rows_line(total - 2 * edge) + "\n")
            writer.writerows(self._row_values(columns, self._slice_query_data(data, total - edge, edge), max_chars))
        return buf.getvalue().rstrip("\n")

    @staticmethod
    def _row_values(columns: List[str], data: Any, max_chars: Optional[int] = None) -> List[List[Any]]:
        """Rows of a query result as value lists in ``columns`` order, each text
        value longer than ``max_chars`` cut to it."""
        if data is None:
            return []
        if hasattr(data, "to_pylist"):  # pyarrow.Table
            data = data.to_pylist()
        elif hasattr(data, "to_dict"):  # pandas.DataFrame
            data = data.to_dict(orient="records")
        rows = []
        for row in data:
            if isinstance(row, Mapping):
                values = [row.get(column) for column in columns]
            elif isinstance(row, (list, tuple)):
                values = list(row)
            else:
                values = [row]
            if max_chars:
                values = [
                    value[:max_chars] + "…" if isinstance(value, str) and len(value) > max_chars else value
                    for value in values
                ]
            rows.append(values)
        return rows

    def metric_datasets(self) -> Optional[Dict[str, List[str]]]:
        """Metric name -> datasets it reads, for the tool-transformer context.

        ``None`` when no runtime is configured; ``{}`` when the catalog is empty.
        Consumers treat a name missing from the map as "no such metric", so every
        page is read. Result is fresh on each call, making a metric added after a
        model reload visible.
        """
        runtime = self.runtime
        if runtime is None:
            return None

        lightweight = getattr(runtime, "metric_datasets", None)
        if callable(lightweight):
            reported = _run_async(lightweight())
            if not isinstance(reported, Mapping):
                logger.warning(
                    "Runtime metric_datasets returned %s; reporting the mapping as unavailable.",
                    type(reported).__name__,
                )
                return None
            mapping = {}
            for name, datasets in reported.items():
                metric_name = str(name or "").strip()
                if metric_name:
                    mapping[metric_name] = _normalize_dataset_names(datasets)
            return mapping

        page_size, max_pages = self._metric_catalog_paging()
        mapping: Dict[str, List[str]] = {}
        offset = 0
        for _ in range(max_pages):
            page = list(_run_async(runtime.list_metrics(limit=page_size, offset=offset)))
            if not page:
                return mapping
            for metric in page:
                name = str(getattr(metric, "name", "") or "")
                if not name:
                    continue
                metadata = _normalize_metric_metadata(getattr(metric, "metadata", None))
                mapping[name] = _normalize_dataset_names(metadata.get("datasets"))
            offset += len(page)

        # A catalog that fills the bound exactly is complete; one more request tells them apart.
        if not list(_run_async(runtime.list_metrics(limit=page_size, offset=offset))):
            return mapping

        logger.warning(
            "Metric catalog still returning rows after %d pages; reporting the mapping as unavailable.",
            max_pages,
        )
        return None

    def _metric_catalog_paging(self) -> Tuple[int, int]:
        """Page size and page cap for Dosi catalog scans."""
        return metric_catalog_paging()

    def _semantic_model_artifact_evidence(self, semantic_model_name: str) -> Dict[str, str]:
        """Return exact Ossie artifact identity for target-bound validation evidence."""
        if not semantic_model_name:
            return {}
        try:
            from datus.agent.node.semantic_authoring import discover_osi_semantic_models

            matches = [
                model
                for model in discover_osi_semantic_models(self.agent_config)
                if str(model.get("semantic_model_name") or "") == semantic_model_name
            ]
            if len(matches) != 1:
                return {}
            path = Path(str(matches[0]["absolute_path"])).expanduser().resolve(strict=True)
            return {
                "semantic_model_name": semantic_model_name,
                "semantic_model_file": str(path),
                "semantic_model_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        except (KeyError, OSError, RuntimeError):
            return {}

    @staticmethod
    def _normalize_runtime_db_context(runtime_db_context: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        if not runtime_db_context:
            return {}

        normalized: Dict[str, str] = {}
        for key in (
            "datasource",
            "catalog",
            "catalog_name",
            "database",
            "database_name",
            "schema",
            "db_schema",
            "schema_name",
        ):
            value = runtime_db_context.get(key)
            if value is None:
                continue
            text = value.strip() if isinstance(value, str) else str(value).strip()
            if text:
                normalized[key] = text

        if "catalog" not in normalized and "catalog_name" in normalized:
            normalized["catalog"] = normalized["catalog_name"]
        if "database" not in normalized and "database_name" in normalized:
            normalized["database"] = normalized["database_name"]
        if "schema" not in normalized:
            if "db_schema" in normalized:
                normalized["schema"] = normalized["db_schema"]
            elif "schema_name" in normalized:
                normalized["schema"] = normalized["schema_name"]
        return normalized

    def set_runtime_db_context(self, runtime_db_context: Optional[Mapping[str, Any]]) -> None:
        """Set a static runtime DB context and invalidate any runtime built for the old context."""
        normalized = self._normalize_runtime_db_context(runtime_db_context)
        if normalized == self._runtime_db_context_static and self._runtime_db_context_static_set:
            return
        self._runtime_db_context_static = normalized
        self._runtime_db_context_static_set = True
        self._runtime = None
        self._runtime_context_key = None

    def _runtime_db_context(self) -> Dict[str, str]:
        if callable(self._runtime_db_context_provider):
            try:
                return self._normalize_runtime_db_context(self._runtime_db_context_provider())
            except Exception as e:
                logger.debug("Failed to resolve runtime DB context for semantic runtime: %s", e)
                return {}
        if self._runtime_db_context_static_set:
            return dict(self._runtime_db_context_static)
        runtime_context_getter = getattr(self.agent_config, "runtime_db_context", None)
        if callable(runtime_context_getter):
            try:
                return self._normalize_runtime_db_context(runtime_context_getter())
            except Exception as e:
                logger.debug("Failed to resolve AgentConfig runtime DB context for semantic runtime: %s", e)
        return {}

    def _selected_semantic_model_path(self) -> str:
        if not callable(self._semantic_model_path_provider):
            return ""
        try:
            return str(self._semantic_model_path_provider() or "").strip()
        except Exception as e:
            logger.debug("Failed to resolve the selected semantic model path: %s", e)
            return ""

    def _extract_db_config(self, datasource: str) -> Optional[dict]:
        """Extract db_config dict from the selected database config."""
        try:
            db_config_obj = self.agent_config.current_db_config(datasource)
        except Exception:
            return None
        if db_config_obj is None:
            return None
        raw = db_config_obj.to_dict()
        # HS2 session configuration is a mapping, not a Python repr string.
        structured = {"configuration"} if str(raw.get("type", "")).lower() == "hive" else set()
        extra = raw.get("extra")
        db_config = {
            k: v if k in structured else str(v)
            for k, v in raw.items()
            if v is not None and v != "" and k not in ("extra", "path_pattern", "default")
        }
        # Preserve connector-specific `extra` fields without overwriting explicit top-level keys
        if isinstance(extra, dict):
            for k, v in extra.items():
                if v is None or v == "":
                    continue
                db_config.setdefault(k, v if k in structured else str(v))
        return db_config

    @property
    def runtime(self) -> Optional[DosiRuntime]:
        """Load the embedded Dosi runtime for the current datasource/model."""
        try:
            runtime_db_context = self._runtime_db_context()
            datasource = runtime_db_context.get("datasource") or self.agent_config.current_datasource
            semantic_model_path = self._selected_semantic_model_path()
            context_key = (
                datasource or "",
                runtime_db_context.get("catalog", ""),
                runtime_db_context.get("database", ""),
                runtime_db_context.get("schema", ""),
                semantic_model_path,
            )
            if self._runtime is not None:
                if self._runtime_context_key is None or self._runtime_context_key == context_key:
                    return self._runtime
                self._runtime = None
                self._runtime_context_key = None

            from datus.configuration.agent_config import _apply_runtime_db_context_to_dosi_profile

            db_config = _apply_runtime_db_context_to_dosi_profile(
                self._extract_db_config(datasource), runtime_db_context
            )
            runtime_config = DosiConfig(
                datasource=datasource,
                db_config=db_config,
                timeout_seconds=float((db_config or {}).get("timeout_seconds", 30)),
                semantic_models_path=str(self.agent_config.path_manager.semantic_model_path(datasource)),
                semantic_model_path=semantic_model_path or None,
            )
            # Ontology-aware agents always load the ontology directory. Every
            # other agent stays on the plain model, unless the datasource has
            # none: then its only models are inline in mapping documents there,
            # and skipping the directory would leave it without a model.
            if self.ontology_enabled or not has_model_files(runtime_config):
                runtime_config = runtime_config.model_copy(
                    update={"ontology_dir": str(self.agent_config.path_manager.ontology_dir)}
                )
            self._runtime = DosiRuntime(runtime_config)
            self._runtime_context_key = context_key
            self._runtime_load_error = None
            logger.info("Loaded Dosi runtime for datasource %s", datasource)
        except Exception as e:
            logger.warning("Failed to load Dosi runtime: %s", e)
            self._runtime_load_error = str(e)
            self._runtime = None
            self._runtime_context_key = None
        return self._runtime

    async def _attribute(self, runtime: DosiRuntime, request: AttributionRequest) -> AttributionResult:
        return await runtime.attribute(request)

    def _runtime_unavailable_message(self) -> str:
        """Return a consistent message for Dosi runtime failures."""
        if self._runtime_load_error:
            return f"Dosi runtime unavailable: {self._runtime_load_error}"
        return "Dosi runtime unavailable."

    def _require_runtime(self, tool_name: str) -> tuple[Optional[DosiRuntime], Optional[FuncToolResult]]:
        """Load the semantic runtime or return a tool failure result."""
        runtime = self.runtime
        if runtime is not None:
            return runtime, None
        return None, FuncToolResult(
            success=0,
            error=f"{tool_name} requires a successfully initialized Dosi runtime. "
            f"{self._runtime_unavailable_message()}",
        )

    def _reload_runtime(self) -> bool:
        """
        Reload the semantic runtime to pick up new configuration changes.

        This is useful after writing new metric/semantic model YAML files so
        subsequent calls discover the updated Dosi model catalog.

        Returns:
            True if reload succeeded, False otherwise
        """
        try:
            # Clear cached runtime and attribution tool
            self._runtime = None
            self._runtime_context_key = None

            # Force reload by accessing the property
            if self.runtime is not None:
                logger.info("Successfully reloaded Dosi runtime")
                return True
            else:
                logger.error("Failed to reload semantic runtime")
                return False

        except Exception as e:
            logger.error(f"Error reloading semantic runtime: {e}", exc_info=True)
            return False

    def available_tools(self) -> List[Tool]:
        """
        Get list of available tools.

        Returns:
            List of Tool objects for LLM function calling
        """
        tools = [
            trans_to_function_tool(self.list_metrics),
            trans_to_function_tool(self.get_metric),
            trans_to_function_tool(self.get_dataset),
            trans_to_function_tool(self.query_metrics),
            trans_to_function_tool(self.validate_semantic),
            trans_to_function_tool(self.attribution_analyze),
        ]
        if self.ontology_enabled:
            tools.extend(trans_to_function_tool(getattr(self, name)) for name in self.ONTOLOGY_TOOL_NAMES)
        return tools

    # ==================== Ontology tools ====================

    #: Exposed only when ``ontology_enabled``; without a loaded ontology they
    #: could only fail.
    ONTOLOGY_TOOL_NAMES = ("describe_concept", "find_concept_paths", "select_records")

    @staticmethod
    def _semantic_failure(tool_name: str, exc: Exception) -> FuncToolResult:
        """A failed tool call that keeps the engine's structured retry payload."""
        payload = getattr(exc, "payload", None)
        if payload is not None and getattr(payload, "error_type", None) == "semantic_validation_error":
            data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
            return FuncToolResult(
                success=0,
                error=getattr(payload, "message", "") or f"{tool_name} validation failed",
                result=data,
            )
        logger.warning(f"{tool_name} failed: {exc}")
        return FuncToolResult(success=0, error=f"{tool_name} failed: {exc}")

    def describe_concept(self, name: Optional[str] = None) -> FuncToolResult:
        """
        Describe the ontology: the concept catalog, or one concept in full.

        Omit ``name`` for the catalog: every concept with its identity, links
        out/in, property count and whether it can root a query. Pass a concept
        name for everything needed to query it: properties as concept paths with
        their type, CLOSED value domain (a filter literal must be one of these
        values) and whether they are time axes; links out with their role and
        target; which concepts link in; the metrics it can group, with the exact
        group-by path to use; and the time dimensions reachable from it. Every
        path it returns can be passed as a field, dimension or filter column to
        query_metrics and select_records.

        Args:
            name: Concept name, e.g. ``FlightAlert``. Omit for the catalog.

        Returns:
            FuncToolResult whose result is the engine's describe payload.
        """
        runtime, error = self._require_runtime("describe_concept")
        if error:
            return error
        try:
            return FuncToolResult(result=_run_async(runtime.describe(normalize_null(name))))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed tool call
            return self._semantic_failure("describe_concept", exc)

    def find_concept_paths(self, source: str, target: str) -> FuncToolResult:
        """
        List every named path from one concept to another (up to 3 hops).

        Call it when a concept is reachable more than one way, e.g. a flight
        reaches an airport as ``arrives_at`` (role destination) and as
        ``departs_from`` (role origin). Each path carries the roles it walks;
        ``equivalences`` names paths the ontology declares to mean the same
        thing, so several paths may be only one or two distinct meanings. Pick
        the path whose role matches the question and use it as a concept path.

        Args:
            source: Concept to start from, e.g. ``Flight``.
            target: Concept to reach, e.g. ``Airport``.

        Returns:
            FuncToolResult with ``paths`` (path, roles, hops, multiplicity),
            ``equivalences`` and ``max_hops``.
        """
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            return FuncToolResult(success=0, error="find_concept_paths requires both source and target concepts")
        runtime, error = self._require_runtime("find_concept_paths")
        if error:
            return error
        try:
            return FuncToolResult(result=_run_async(runtime.concept_paths(source, target)))
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed tool call
            return self._semantic_failure("find_concept_paths", exc)

    #: Rows returned when the caller sets no limit; the full result is cached.
    SELECT_RECORDS_DEFAULT_LIMIT = 200

    @tool_schema(strict_mode=False)
    def select_records(
        self,
        from_concept: str,
        fields: List[str],
        where: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> FuncToolResult:
        """
        Read individual records (not aggregates) of one concept.

        Use it to list ids, names, statuses or other properties of matching
        records: "which alerts are open on flights into JFK". For counts, sums,
        averages or rates use query_metrics instead; never aggregate records by
        hand.

        Args:
            from_concept: The concept each returned row is one record of, e.g.
                ``FlightAlert``. Must be a concept that can root a query.
            fields: Concept paths to return. Follow links by name to reach related
                concepts, e.g. ``FlightAlert.flight.number`` or
                ``FlightAlert.flight.arrives_at.name``.
            where: Optional SQL boolean expression over concept paths, e.g.
                ``FlightAlert.status = 'OPEN' AND FlightAlert.flight.arrives_at.code = 'JFK'``.
                A literal compared to a property with a closed domain must be one
                of the domain's values.
            order_by: Optional sort keys: a requested field path (``FlightAlert.id``) or a
                result column name; prefix with '-' for descending.
            limit: Optional maximum number of rows (default 200).

        Returns:
            FuncToolResult with text laid out like query_metrics': a ``result_id``
            line, a ``rows`` line, then the rows as CSV with columns named by the
            requested field paths (the full result is cached and used for the
            final output).
        """
        from_concept = str(from_concept or "").strip()
        fields = _normalize_name_list(fields)
        order_by = _normalize_name_list(order_by)
        where = normalize_null(where)
        limit = normalize_null(limit)
        if not from_concept or not fields:
            return FuncToolResult(success=0, error="select_records requires from_concept and at least one field")
        runtime, error = self._require_runtime("select_records")
        if error:
            return error
        try:
            result = _run_async(
                runtime.select(
                    from_concept=from_concept,
                    fields=fields,
                    where=where,
                    order_by=order_by or None,
                    limit=int(limit) if limit is not None else self.SELECT_RECORDS_DEFAULT_LIMIT,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed tool call
            return self._semantic_failure("select_records", exc)

        return FuncToolResult(
            result=self._render_query_result(
                self._cache_query_metrics_result(result.columns, result.data),
                list(result.columns),
                result.data,
                # Only the default page, not a limit the caller asked for.
                row_cap=self.SELECT_RECORDS_DEFAULT_LIMIT if limit is None else None,
            )
        )

    @mcp_tool()
    def list_metrics(
        self,
        path: Optional[List[str]] = None,
        limit: int = _LIST_METRICS_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> FuncToolResult:
        """
        List executable metrics, using the knowledge base for subject paths.

        The reply is text: metrics grouped under their subject path, one line
        each, ``- name -- description``, enough to choose metrics. How a metric
        is computed, which tables it reads and its time axis are in get_metric;
        read it for every metric you query. When the page is not the whole
        list, a last ``more:`` line gives the offset to call again with.

        Args:
            path: Optional subject tree path filter (e.g., ["Finance", "Revenue"])
            limit: Maximum number of metrics to return
            offset: Number of metrics to skip
        """
        # Normalize null values and stringified numbers from LLM
        path = _normalize_optional_path(path)
        limit = _normalize_pagination_bound(limit, default=_LIST_METRICS_DEFAULT_LIMIT, minimum=1)
        offset = _normalize_pagination_bound(offset, default=0, minimum=0)
        logger.debug(f"list_metrics called: path={path}, limit={limit}, offset={offset}")
        runtime, error = self._require_runtime("list_metrics")
        if error:
            return error

        try:
            try:
                kb_paths = self._metric_subject_paths(path)
            except Exception as exc:
                if path:
                    raise
                logger.warning(
                    "Could not load KB metric subject paths; returning the runtime catalog without paths: %s",
                    exc,
                )
                kb_paths = {}
            if path:
                if not kb_paths:
                    return self._metric_list_result([], total=0, offset=offset, limit=limit)

                async_result = self._runtime_metrics_for_names(runtime, set(kb_paths))
                runtime_metrics = [
                    _metric_list_row(metric, kb_paths[normalize_metric_name(metric.name)])
                    for metric in async_result
                    if normalize_metric_name(metric.name) in kb_paths
                ]
                total = len(runtime_metrics)
                paginated_metrics = runtime_metrics[offset : offset + limit]
                return self._metric_list_result(
                    paginated_metrics,
                    total=total,
                    offset=offset,
                    limit=limit,
                )

            async_result = _run_async(runtime.list_metrics(path=None, limit=limit, offset=offset))
            runtime_metrics = [
                _metric_list_row(metric, kb_paths.get(normalize_metric_name(metric.name))) for metric in async_result
            ]
            # Runtime path has no guaranteed upstream total — leave it None; has_more
            # falls back to a full page.
            return self._metric_list_result(runtime_metrics, total=None, offset=offset, limit=limit)

        except Exception as e:
            logger.error(f"Error listing metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to list metrics: {str(e)}",
            )

    def _metric_subject_paths(self, path: Optional[List[str]]) -> Dict[str, List[str]]:
        """Return datasource-scoped KB subject paths keyed by normalized metric name."""
        rows = self.metric_rag.search_all_metrics(select_fields=["name"])
        paths: Dict[str, List[str]] = {}
        for row in rows:
            name = normalize_metric_name(row.get("name"))
            raw_path = row.get("subject_path")
            if not name or not isinstance(raw_path, list):
                continue
            subject_path = _normalize_name_list(raw_path)
            if subject_path and (not path or subject_path[: len(path)] == path):
                paths[name] = subject_path
        return paths

    def _runtime_metrics_for_names(self, runtime: DosiRuntime, names: set[str]) -> List[Any]:
        """Read unfiltered runtime pages until every KB candidate is found or the catalog ends."""
        page_size, max_pages = self._metric_catalog_paging()
        matched: List[Any] = []
        found: set[str] = set()
        offset = 0

        for _ in range(max_pages):
            page = list(_run_async(runtime.list_metrics(path=None, limit=page_size, offset=offset)))
            if not page:
                return matched

            for metric in page:
                name = normalize_metric_name(getattr(metric, "name", None))
                if name in names and name not in found:
                    matched.append(metric)
                    found.add(name)

            if found == names:
                return matched
            offset += len(page)

        if not list(_run_async(runtime.list_metrics(path=None, limit=page_size, offset=offset))):
            return matched

        raise DatusException(
            ErrorCode.TOOL_EXECUTION_FAILED,
            message=(
                f"Metric catalog still returning rows after {max_pages} pages; "
                "cannot apply the knowledge-base subject path safely."
            ),
        )

    def _runtime_metric_by_name(self, runtime: DosiRuntime, name: str) -> Optional[Any]:
        """The catalog entry for one metric, over the whole catalog.

        Paged the same way ``list_metrics`` pages it rather than read once with a
        bound, so a name ``list_metrics`` handed out from a later page is still
        describable. Running out of pages means the name is not in the catalog —
        the caller reports an unknown metric, which is a different answer from
        the integrity failure ``_runtime_metrics_for_names`` raises when a name
        the knowledge base knows cannot be located.
        """
        page_size, max_pages = self._metric_catalog_paging()
        wanted = normalize_metric_name(name)
        offset = 0

        for _ in range(max_pages):
            page = list(_run_async(runtime.list_metrics(path=None, limit=page_size, offset=offset)))
            if not page:
                return None
            for metric in page:
                if normalize_metric_name(getattr(metric, "name", None)) == wanted:
                    return metric
            offset += len(page)

        logger.warning(
            "Metric catalog still returning rows after %d pages; stopped looking for %s.",
            max_pages,
            name,
        )
        return None

    @mcp_tool()
    def get_metric(
        self,
        name: str,
        path: Optional[List[str]] = None,
    ) -> FuncToolResult:
        """
        Get one metric's definition and what a query of it can use. One metric
        per call; issue several calls in parallel to describe several.

        The reply is text, one ``key: value`` per line, and leaves out what the
        metric does not have:
        - ``expr``: how the metric is computed from other metrics or fields —
          an aggregate (``COUNT(DISTINCT t.customer_id)``), a filter
          (``base FILTER (WHERE condition)``: ``base`` counted only on rows
          meeting the condition), a compose's arithmetic, a level-of-detail
          declaration, or a window's base. Names in it are other metrics
          (get_metric) or ``dataset.field`` (get_dataset shows the field's
          definition: one written ``fixed(...) AS name`` is a per-key value
          computed over rows of its own).
        - ``datasets``: what it reads; read each with get_dataset before the
          first query — the dataset and field descriptions hold conventions
          every query on it follows (a snapshot's version day, what a code
          value means). ``dimension_datasets``: the datasets whose fields
          it can be grouped and filtered by (``name(f1, f2)`` when only
          those fields).
        - ``window`` / ``compares_across``: a window metric's rule, and the
          dimensions a ranking or share compares across.
        - ``params``: ``name type = default [domain] -- meaning
          [declared_on metric]``; bind them in query_metrics ``params``. One
          ``declared_on`` a fixed field's base is that value's own window.
        - ``required``: dimensions every query of it must group by.
          ``requires_time_axis``: the query must also group by ``metric_time``
          at a grain.
        - ``time``: ``metric_time`` and its grains, finest first (the first is
          the default); ``none`` for a metric
          whose table has no time field (a snapshot): query it without
          time_start/time_end and pick the day it reads in ``where``.

        Args:
            name: Exact metric name from the selected model or metric catalog.
            path: Optional subject tree path (e.g., ["Finance", "Revenue"])
        """
        name = str(normalize_null(name) or "").strip()
        path = _normalize_optional_path(path)
        logger.debug(f"get_metric called: name={name}, path={path}")
        runtime, error = self._require_runtime("get_metric")
        if error:
            return error

        if not name:
            return FuncToolResult(
                success=0,
                error="get_metric requires a metric name. Use an exact name from the selected model or metric catalog.",
            )

        try:
            detail = _run_async(runtime.describe_metric(name))
        except Exception as e:
            payload = getattr(e, "payload", None)
            if getattr(payload, "code", None) == "unknown_metric":
                return FuncToolResult(
                    success=0,
                    error=f"Unknown metric {name!r}. Check the exact names in the selected model or metric catalog.",
                )
            logger.error(f"Error describing metric {name}: {e}")
            return FuncToolResult(success=0, error=f"Failed to get metric detail: {str(e)}")

        try:
            # The KB, not the runtime, owns the navigation path — same rule
            # list_metrics follows, so both tools name a metric the same way.
            subject_path = self._metric_subject_paths(None).get(normalize_metric_name(name))
        except Exception as exc:
            logger.warning("get_metric could not read the KB subject path for %s: %s", name, exc)
            subject_path = None

        return FuncToolResult(success=1, result=render_metric_detail(detail, subject_path))

    def metric_record(self, name: str, path: Optional[List[str]] = None) -> FuncToolResult:
        """One metric as a structured record, for callers that store or check it
        (metric artifacts, insight snapshots) rather than an agent reading it:
        the engine's description, the knowledge-base ``path`` when there is
        one, and the ``dimensions`` a query may group and filter it by.
        Not an agent tool — ``get_metric`` is.
        """
        name = str(normalize_null(name) or "").strip()
        runtime, error = self._require_runtime("get_metric")
        if error:
            return error
        try:
            detail = _run_async(runtime.describe_metric(name))
            dimensions = _run_async(runtime.get_dimensions(metric_name=name, path=path))
        except Exception as e:
            return FuncToolResult(success=0, error=f"Failed to get metric detail: {str(e)}")
        record = {key: value for key, value in detail.items() if key not in ("measures", "explicit_measure")}
        try:
            subject_path = self._metric_subject_paths(None).get(normalize_metric_name(name))
        except Exception as exc:
            logger.warning("metric_record could not read the KB subject path for %s: %s", name, exc)
            subject_path = None
        record.pop("path", None)
        if subject_path:
            record["path"] = subject_path
        rows = []
        for dimension in dimensions or []:
            row = dimension.model_dump() if hasattr(dimension, "model_dump") else dict(dimension)
            rows.append({key: value for key, value in row.items() if value is not None})
        record["dimensions"] = rows
        return FuncToolResult(success=1, result=record)

    @mcp_tool()
    def get_dataset(self, name: str) -> FuncToolResult:
        """
        Get one dataset's fields and their definitions. Read every dataset a
        query reads — each in the metrics' ``datasets``, each you group or
        filter on — once, before the first query_metrics; issue the calls in
        parallel. The dataset's description and its fields' descriptions hold
        conventions every query on it must follow (a snapshot's version day,
        what a code value means), and nothing else states them.

        The reply is text. The first line, when there is one, has the
        dataset's description, ``source``, ``time`` (the field ``metric_time``
        and the period land on) and ``pk``. Then one line per field:
        - ``- name -- description``: a column.
        - ``- expression AS name``: a computed field. Write ``name`` in
          conditions and group-bys, never the expression.
        - ``- fixed(base, keys) AS name``: a value computed per key over rows
          of its own (``base`` is a metric, see get_metric). A ``where``
          condition does not reach it; ``context_filter`` does.
        - ``- name [from columns]``: a computed field whose expression is too
          long to show, with the columns it reads.
        Bracketed marks: ``time <format>`` a time field, ``format <format>``
        a stored date format, ``not a dimension``, ``reads <dataset>`` a fixed
        field computed over another dataset, ``time_axis <column>`` where the
        period lands for it when not this dataset's ``time``.
        ``relationships:`` lists ``cardinality dataset on column=column``;
        ``many_to_many`` pairs two facts on those columns only.

        Args:
            name: Dataset name, as a metric's ``datasets`` /
                ``dimension_datasets`` or a field's ``dataset.field`` names it
        """
        name = str(normalize_null(name) or "").strip()
        runtime, error = self._require_runtime("get_dataset")
        if error:
            return error
        if not name:
            return FuncToolResult(success=0, error="get_dataset requires a dataset name.")
        try:
            detail = _run_async(runtime.describe_dataset(name))
        except Exception as e:
            payload = getattr(e, "payload", None)
            message = getattr(payload, "message", None) or str(e)
            if getattr(payload, "code", None) != "unknown_dataset":
                logger.error(f"Error describing dataset {name}: {e}")
            return FuncToolResult(success=0, error=message)
        return FuncToolResult(success=1, result=render_dataset_detail(detail))

    @staticmethod
    def _metric_list_result(
        rows: List[dict],
        *,
        total: Optional[int],
        offset: int,
        limit: Optional[int] = None,
    ) -> FuncToolResult:
        """One list_metrics page, rendered.

        When ``total`` is known (storage path) ``has_more`` is exact. When
        ``total`` is None (runtime path) ``has_more`` falls back to
        ``len(rows) == limit`` — a heuristic, but good enough for the LLM
        to decide whether to fetch another page.
        """
        if total is not None:
            has_more: Optional[bool] = offset + len(rows) < total
        elif limit is not None:
            has_more = len(rows) == limit
        else:
            has_more = None
        next_offset = offset + len(rows) if has_more else None
        return FuncToolResult(
            success=1,
            result=render_metric_list(rows, total=total, has_more=has_more, next_offset=next_offset),
        )

    # Dosi binding names come from metric declarations and require an open schema.
    @tool_schema(strict_mode=False)
    @mcp_tool()
    def query_metrics(
        self,
        metrics: List[str],
        dimensions: Optional[List[str]] = None,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        time_granularity: Optional[str] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        context_filter: Optional[str] = None,
    ) -> FuncToolResult:
        """
        Query metrics data (requires runtime).

        Return complete metric results by default. Do not pass limit just to
        preview data, reduce output size, or be conservative; the full result
        is cached and used for the final output, so a cut preview is never a
        reason to query again. Use limit only when the user explicitly asks for
        Top N, first N, maximum N rows, a preview, or another row-count
        restriction. When using limit for Top N/Bottom N, also pass order_by so
        the truncation has stable business meaning.

        The result is text: a `result_id` line, a `rows` line with the row
        count, then one line for each thing to report — a `warning` or `info`
        from the engine, a `note` on values of an `IN (...)` list that matched
        no row, a `column` line mapping a column named after a parameter
        binding to its metric and binding — then a blank line and the rows as
        CSV. A long result shows its first and last rows around a
        `... N rows omitted` line.

        Where conditions go. One call is one query and returns one table. Write
        columns as `dataset.field` with the field name get_dataset shows after
        `AS`, never the expression. Dates in time_start/time_end are YYYY-MM-DD
        with time_end exclusive: one day is `2026-06-30` to `2026-07-01`, never
        the same start and end. Inside a condition, write a date in the column's
        stored format (the `time`/`format` mark in get_dataset).
        1. Read the metrics' `expr` first. A field get_dataset defines as
           `fixed(...)` — named in `expr`, or read by a field `expr` names — is
           a per-key value computed over rows of its own. A scope the whole
           question obeys (one segment or category the question is limited
           to, a condition on a table only a fixed value reads) goes in
           context_filter: every part of the query applies it where it can
           read its columns.
        2. The period goes in time_start/time_end, never repeated in `where`:
           every metric and every fixed value lands on its own table's time
           field. A `params` entry
           `declared_on` a fixed value's base is that value's own window: bind
           it, and either let the period cover it or leave the period out and
           put the main metric's own dates in `where`. A fixed value that must
           not follow the period (a first-ever date, a lifetime total — read
           its description) likewise: dates in `where`, on the
           date field of the metric's own table. Previous-period and running
           windows take their dates only from time_start/time_end.
        3. `where` picks the rows the metrics count, like SQL WHERE, and never
           changes a fixed value. A condition on a fixed value itself or on a
           selected metric is applied after it is computed. A snapshot or
           mapping table's version day (get_metric `time: none`) goes here; for
           each day's state over a range, select every version day and group by
           that field.
        4. On a dimension get_metric lists under `compares_across`: "rank or
           share among all values" → where; "only among these" → context_filter.
        5. Act on every `warning` line of a result as its `hint` says; an
           `info` line only records where a condition landed. A
           `where_needs_context` error names a where condition only some parts
           can read: move it to context_filter. When the engine refuses a
           combination, follow its `suggested_retry`; if that means two
           queries, run both and say so.

        Args:
            metrics: List of metric names to query
            dimensions: Optional list of dimensions to group by, as `dataset.field`,
                        from the datasets get_metric lists under `dimension_datasets`
                        (get_dataset shows their fields). To group by the metric's time axis, use
                        `metric_time`, with `time_granularity` for its grain.
            time_start: Optional inclusive start of a half-open time range (ISO date YYYY-MM-DD)
            time_end: Optional exclusive end of a half-open time range (ISO date YYYY-MM-DD)
            time_granularity: Optional time granularity for aggregation ('day', 'week', 'month', 'quarter', 'year')
            where: Optional SQL boolean expression (without the WHERE keyword). Each
                   top-level AND condition runs where its columns exist: a condition on a
                   selected metric (`revenue > 1000`) filters the result rows after
                   aggregation, like HAVING; in a query with a window metric, a condition
                   on group-by fields only also filters the result, so the kept rows keep
                   the ranks and partition counts of the whole population; every other
                   condition filters rows before aggregation.
            context_filter: Optional SQL boolean expression every part of the query
                   applies to its input rows before aggregation, where it can read
                   the columns — the
                   metrics, the rows a fixed value is computed over, the population
                   a window ranks. `where="market = 'North'"` shows North's rank
                   among all markets; `context_filter="market = 'North'"` ranks
                   North's rows against each other.
            limit: Optional maximum number of rows
            order_by: Optional list of result columns to sort by. Use column name for ascending,
                      prefix with '-' for descending. A Dosi input dimension `metric_time` may
                      produce a result/order key such as `metric_time__day`. Examples:
                      ['metric_time__day'] for ascending, ['-message_count'] for descending.
                      Do NOT use 'asc'/'desc' keywords.
            params: Optional bindings for the parameters get_metric lists under
                `params` — the metric's own, its members', and those of the fixed
                values it reads (`declared_on`) — by name. A scalar binds one value; a
                non-default one names the column `<metric>__k_7`. A list returns one
                column per value in the same query: `{"k": [1, 7]}` returns
                `<metric>__k_1` and `<metric>__k_7`, so several values of one
                parameter side by side never need separate queries. Omitted
                parameters take their default, and the column keeps the plain
                metric name. A value that does not spell a readable suffix
                (non-ASCII text, punctuation) gets an opaque token instead, so map
                such columns to bindings through the result's `column` lines,
                never by parsing column names.
            dry_run: If True, return the compiled SQL without executing the query,
                with the warnings an execution would carry. A configured warehouse
                dry-run provider also validates that SQL. Use it only when you need
                the SQL itself: an executed result carries the same warnings, just
                not its SQL, so a dry run before executing only repeats the query.

        Returns:
            FuncToolResult with the result text, or the compiled SQL of a dry run
        """
        metrics = _normalize_name_list(metrics)
        dimensions = _normalize_name_list(dimensions)
        order_by = _normalize_name_list(order_by)
        params = normalize_null(params)

        runtime, error = self._require_runtime("query_metrics")
        if error:
            return error

        if not metrics:
            return FuncToolResult(
                success=0,
                error=(
                    "query_metrics requires at least one metric name. "
                    "Pass exact names from the selected model or metric catalog."
                ),
            )
        if params is not None and not isinstance(params, dict):
            return FuncToolResult(
                success=0,
                error="query_metrics params must be an object of parameter bindings",
            )
        # Sanitize optional parameters: LLMs may pass string null placeholders
        # instead of omitting them. In particular, an empty-string limit must not
        # reach runtimes that convert a present limit with ``int(limit)``.
        time_start = normalize_null(time_start)
        time_end = normalize_null(time_end)
        time_granularity = normalize_null(time_granularity)
        where = normalize_null(where)
        context_filter = normalize_null(context_filter)
        limit = normalize_null(limit)
        logger.debug(
            f"query_metrics called: metrics={metrics}, dimensions={dimensions}, "
            f"time=[{time_start},{time_end}], granularity={time_granularity}, where={where}, "
            f"context_filter={context_filter}, limit={limit}, dry_run={dry_run}"
        )

        try:
            # Execute query via runtime
            runtime_query_kwargs = {
                "metrics": metrics,
                "dimensions": dimensions,
                "time_start": time_start,
                "time_end": time_end,
                "time_granularity": time_granularity,
                "where": where,
                "limit": limit,
                "order_by": order_by or None,
                "dry_run": dry_run,
            }
            if params:
                runtime_query_kwargs["params"] = params
            if context_filter:
                runtime_query_kwargs["context_filter"] = context_filter
            result = _run_async(runtime.query_metrics(**runtime_query_kwargs))
            if not dry_run:
                metadata = result.metadata or {}
                return FuncToolResult(
                    result=self._render_query_result(
                        self._cache_query_metrics_result(result.columns, result.data, metadata),
                        list(result.columns),
                        result.data,
                        warnings=metadata.get("warnings"),
                        notes=self._query_hints(runtime, runtime_query_kwargs, result),
                        outputs=metadata.get("outputs"),
                    )
                )

            # A dry run keeps its structure: the compiled SQL and the warehouse
            # check are read by code (publish evidence, the Hub preview) as well.
            # Drop non-JSON-serializable metadata entries. ``str(v)`` on
            # objects yields ``<... object at 0x...>``, which is useless to
            # both LLM callers and humans.
            safe_metadata = {}
            for k, v in (result.metadata or {}).items():
                try:
                    json.dumps(v)
                    safe_metadata[k] = v
                except (TypeError, ValueError):
                    continue
            warehouse_error = None
            if callable(self._warehouse_dry_run_provider) and not (
                isinstance(safe_metadata.get("warehouse_dry_run"), dict)
                and safe_metadata["warehouse_dry_run"].get("status") == "success"
            ):
                sql = str(safe_metadata.get("sql") or "").strip()
                if not sql:
                    warehouse_evidence: Mapping[str, Any] = {
                        "status": "failed",
                        "error": "Semantic runtime dry-run did not return compiled SQL.",
                    }
                else:
                    try:
                        warehouse_evidence = self._warehouse_dry_run_provider(sql)
                    except Exception as exc:
                        warehouse_evidence = {"status": "failed", "error": str(exc)}
                safe_metadata["warehouse_dry_run"] = dict(warehouse_evidence)
                if warehouse_evidence.get("status") != "success":
                    warehouse_error = str(warehouse_evidence.get("error") or "Warehouse EXPLAIN failed.")
            result_dict = {
                "columns": list(result.columns),
                "data": self.compressor.compress(result.data),
            }
            if safe_metadata:
                result_dict["metadata"] = safe_metadata

            tool_result = FuncToolResult(
                success=0 if warehouse_error else 1,
                error=f"Warehouse dry-run failed: {warehouse_error}" if warehouse_error else None,
                result=result_dict,
            )
            if self.generation_evidence:
                self.generation_evidence.record_metric_dry_run(
                    metrics,
                    tool_result,
                )
            return tool_result

        except Exception as e:
            # Surface backend validation rejections as structured planner guidance.
            # Duck-typed so the tool layer stays decoupled from any specific runtime.
            payload = getattr(e, "payload", None)
            if payload is not None and getattr(payload, "error_type", None) == "semantic_validation_error":
                data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
                # Only where_needs_context says which condition and where it can go.
                for key in ("condition", "parts"):
                    if not data.get(key):
                        data.pop(key, None)
                logger.debug(f"query_metrics validation rejection: code={data.get('code')}")
                message = getattr(payload, "message", "") or "query_metrics validation failed"
                if data.get("code") == "no_primary_time_dimension":
                    message = self._no_time_axis_message(runtime, metrics, data.get("metrics") or [])
                    data["message"] = message
                    # Every time field of the model, unrelated to the metric: the
                    # agent read them as places to move the period to.
                    data["required_dimensions"] = []
                return FuncToolResult(success=0, error=message, result=data)
            logger.error(f"Error querying metrics: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to query metrics: {str(e)}",
                result={"execution_info": e.execution_info} if getattr(e, "execution_info", None) else None,
            )

    def _query_hints(
        self,
        runtime: DosiRuntime,
        query_kwargs: Dict[str, Any],
        result: Any,
    ) -> List[str]:
        """Values of an ``IN (...)`` list that matched no row.

        A list of names checked against a name table that lacks some of them
        drops those names silently, and the agent read the shorter result as
        "no data for them". The result answers that itself only when it groups
        by the field and holds every row (no ``limit``); otherwise the same
        conditions are run grouped by that field alone, which returns at most
        one row per listed value. A ``where`` naming a selected metric filters
        rows after aggregation, so its outcome depends on the grouping and no
        check is made.
        """
        if _filters_on_metric(query_kwargs.get("where"), list(query_kwargs.get("metrics") or [])):
            return []
        columns = list(getattr(result, "columns", None) or [])
        rows = [row for row in (getattr(result, "data", None) or []) if isinstance(row, Mapping)]
        hints: List[str] = []
        checked = set()
        conditions = _in_list_conditions(query_kwargs.get("where")) + _in_list_conditions(
            query_kwargs.get("context_filter")
        )
        for field, values in conditions:
            if field in checked:
                continue
            checked.add(field)
            present = self._values_present(runtime, query_kwargs, field, columns, rows)
            if present is None:
                continue
            missing = [value for value in values if _value_key(value) not in present]
            if missing:
                hints.append(
                    f"{field} IN (...): {len(missing)} of {len(values)} values matched no row in this query: "
                    f"{', '.join(missing)}. Check whether {field} holds these values at all before reading "
                    f"their absence as no data."
                )
        return hints

    def _values_present(
        self,
        runtime: DosiRuntime,
        query_kwargs: Dict[str, Any],
        field: str,
        columns: List[str],
        rows: List[Mapping[str, Any]],
    ) -> Optional[set]:
        column = _result_column_for(field, columns)
        if column is not None and query_kwargs.get("limit") is None:
            return {_value_key(row.get(column)) for row in rows if row.get(column) is not None}
        probe_kwargs = dict(query_kwargs)
        probe_kwargs["dimensions"] = [field]
        probe_kwargs["time_granularity"] = None
        probe_kwargs["limit"] = None
        probe_kwargs["order_by"] = None
        try:
            probe = _run_async(runtime.query_metrics(**probe_kwargs))
        except Exception as exc:  # the result stands without the check
            logger.debug("IN-list check on %s skipped: %s", field, exc)
            return None
        probe_columns = list(getattr(probe, "columns", None) or [])
        column = _result_column_for(field, probe_columns)
        if column is None:
            return None
        return {
            _value_key(row.get(column))
            for row in (getattr(probe, "data", None) or [])
            if isinstance(row, Mapping) and row.get(column) is not None
        }

    def _no_time_axis_message(
        self,
        runtime: DosiRuntime,
        metrics: List[str],
        rejected: List[str],
    ) -> str:
        """How to query a metric the period cannot land on.

        The engine names the metric (a parameter binding's column, such as
        ``m__k_7``, for a parameterized one) and lists every time field of the
        model, which reads as a list of places to move the period to. What the
        agent needs instead: no period for this metric, its day in ``where``.
        """
        names: List[str] = []
        for name in rejected or metrics:
            requested = next((m for m in metrics if name == m or name.startswith(f"{m}__")), name)
            if requested not in names:
                names.append(requested)
        datasets: List[str] = []
        for name in names:
            try:
                metric = self._runtime_metric_by_name(runtime, name)
            except Exception as exc:  # the advice stands without the dataset names
                logger.debug("no dataset names for %s: %s", name, exc)
                continue
            metadata = _normalize_metric_metadata(getattr(metric, "metadata", None)) if metric is not None else {}
            for dataset in _normalize_dataset_names(metadata.get("datasets")):
                if dataset not in datasets:
                    datasets.append(dataset)
        one = len(names) == 1
        tables = f" ({', '.join(datasets)})" if datasets else ""
        return (
            f"{', '.join(names)} {'has' if one else 'have'} no time axis: the table{tables} "
            f"{'it reads has' if one else 'they read have'} no time field the period or `metric_time` could "
            "land on (a snapshot or mapping table, say). Query "
            f"{'it' if one else 'them'} without time_start/time_end, time_granularity and `metric_time`, and pick "
            "the day it reads with a condition on its table's date column in `where`."
        )

    @mcp_tool()
    def validate_semantic(
        self,
        scope: Literal["all", "semantic_model"] = "all",
        semantic_model_name: str = "",
    ) -> FuncToolResult:
        """
        Check model structure, references and metric compilation (requires runtime).

        This does not execute business queries or compare values against a baseline.
        Use query_metrics for execution, or verify_semantic_outputs when available
        for execution and numerical comparison against an explicit reference SELECT.

        After successful validation, the runtime is reloaded to pick up any new
        metrics or semantic model changes. This ensures that subsequent calls to
        query_metrics can find newly created metrics.

        Args:
            scope: Validation scope. Use "all" for full semantic-layer validation,
                including metrics. Use "semantic_model" when generating semantic
                models before metric definitions exist; this still fails on real
                semantic model errors but ignores the expected no-metrics issue.
            semantic_model_name: Optional OSI model to validate. Required with
                scope="semantic_model" when a datasource contains multiple models.

        Returns:
            FuncToolResult with validation status and issues
        """
        scope = normalize_null(scope) or "all"
        semantic_model_name = str(normalize_null(semantic_model_name) or "").strip()
        if scope not in ("all", "semantic_model"):
            return FuncToolResult(
                success=0,
                error="scope must be one of: all, semantic_model",
                result=None,
            )

        logger.debug(f"validate_semantic called scope={scope} semantic_model_name={semantic_model_name}")
        runtime, error = self._require_runtime("validate_semantic")
        if error:
            error.result = None
            return error

        try:
            validation_metric_names: Optional[List[str]] = None
            if callable(self._semantic_metric_names_provider):
                validation_metric_names = _normalize_name_list(self._semantic_metric_names_provider())
            validation_kwargs: Dict[str, Any] = {"scope": scope}
            if semantic_model_name:
                validation_kwargs["semantic_model_name"] = semantic_model_name
            if validation_metric_names is not None:
                validation_kwargs["metric_names"] = validation_metric_names
            validation_result = _run_async(runtime.validate_semantic(**validation_kwargs))
            validation_metadata = getattr(validation_result, "metadata", None)

            # Serialize ValidationIssue objects to dicts
            issues_data = [_serialize_validation_issue(issue) for issue in validation_result.issues]

            ignored_issues = []
            effective_issues = issues_data
            if scope == "semantic_model":
                ignored_issues = [issue for issue in issues_data if _is_no_metrics_present_issue(issue)]
                effective_issues = [issue for issue in issues_data if not _is_no_metrics_present_issue(issue)]

            effective_valid = validation_result.valid or (
                scope == "semantic_model" and not _validation_has_errors(effective_issues)
            )

            if issues_data:
                logger.warning(
                    "Semantic validation issues scope=%s valid=%s effective_valid=%s issues=%d ignored=%d",
                    scope,
                    validation_result.valid,
                    effective_valid,
                    len(effective_issues),
                    len(ignored_issues),
                )
                logger.debug(
                    "Full semantic validation issues=%s ignored=%s",
                    json.dumps(effective_issues, ensure_ascii=False),
                    json.dumps(ignored_issues, ensure_ascii=False),
                )

            # If validation succeeded, reload the runtime to pick up new metrics
            if effective_valid:
                logger.debug("Validation succeeded, reloading runtime to pick up new metrics...")
                self._reload_runtime()

            compact_issues = [
                issue.model_dump(exclude_none=True) for issue in _compact_validation_issues(effective_issues)
            ]
            compact_ignored_issues = [
                issue.model_dump(exclude_none=True) for issue in _compact_validation_issues(ignored_issues)
            ]
            result_payload = {
                "valid": effective_valid,
                "issues": compact_issues,
                "scope": scope,
                "ignored_issues": compact_ignored_issues,
                "issue_count": len(effective_issues),
                "ignored_issue_count": len(ignored_issues),
            }
            if effective_valid and isinstance(validation_metadata, dict):
                contract_digest = str(validation_metadata.get("contract_digest") or "").strip()
                metric_digests = validation_metadata.get("compiled_metric_digests")
                if contract_digest:
                    result_payload["contract_digest"] = contract_digest
                if isinstance(metric_digests, dict):
                    result_payload["compiled_metric_digests"] = dict(metric_digests)
                    result_payload["compiled_metric_count"] = len(metric_digests)
            if effective_valid and semantic_model_name:
                result_payload.update(self._semantic_model_artifact_evidence(semantic_model_name))

            tool_result = FuncToolResult(
                success=1 if effective_valid else 0,
                result=result_payload,
                error=None if effective_valid else _format_validation_error(compact_issues),
            )
            if self.generation_evidence:
                self.generation_evidence.record_validation_result(tool_result)
                if effective_valid:
                    self.generation_evidence.record_compiled_validation(
                        validation_metadata,
                        metric_names=validation_metric_names,
                    )
            return tool_result

        except Exception as e:
            logger.error(f"Error validating semantic config: {e}", exc_info=True)
            return FuncToolResult(
                success=0,
                error=f"Failed to validate semantic config: {str(e)}",
                result=None,
            )

    # Dosi parameter names come from metric declarations and require an open schema.
    @tool_schema(strict_mode=False)
    @mcp_tool()
    def attribution_analyze(
        self,
        metric_name: str,
        candidate_dimensions: List[str],
        baseline_start: str,
        baseline_end: str,
        current_start: str,
        current_end: str,
        max_selected_dimensions: int = 3,
        top_n_values: int = 10,
        where: Optional[str] = None,
        max_dimension_values: int = 500,
        time_dimension: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> FuncToolResult:
        """
        Descriptive dimension analysis for metric changes.

        Ranks candidate dimensions by change concentration and calculates delta
        contributions for selected dimensions. Results describe where a metric change
        is concentrated; they do not establish causation. Failed, truncated, and
        non-additive dimensions are excluded from rankings and summary contributions.

        Args:
            metric_name: Metric to analyze(from list_metrics/search_metrics)
            candidate_dimensions: List of dimensions to evaluate (from get_metric)
            baseline_start: Inclusive baseline start date in an OSI half-open range (e.g., "2026-01-01")
            baseline_end: Exclusive baseline end date (e.g., "2026-01-08" for Jan 1-7)
            current_start: Inclusive current start date in an OSI half-open range (e.g., "2026-01-08")
            current_end: Exclusive current end date (e.g., "2026-01-15" for Jan 8-14)
            max_selected_dimensions: Maximum dimensions to select (default 3)
            top_n_values: Number of top dimension values to return (default 10)
            where: Optional SQL boolean expression applied to every attribution query
            max_dimension_values: Maximum grouped values per dimension (hard-capped at 1000)
            time_dimension: Optional metric time dimension used for both windows
            params: Optional parameter bindings for a parameterized metric

        Returns:
            FuncToolResult with:
            - dimension_ranking: All dimensions ranked by importance score
            - selected_dimensions: Top dimensions selected for analysis
            - top_dimension_values: Delta contributions of dimension values
            - implementation: dosi or generic
            - strategy: term_wise, mix_shift, factor_shapley, or unsupported
            - total_change: Baseline, current, and delta values
            - per_dimension: Per-dimension contribution details
        """
        runtime, error = self._require_runtime("attribution_analyze")
        if error:
            return error
        assert runtime is not None

        try:
            request = AttributionRequest(
                metric=metric_name,
                dimensions=candidate_dimensions,
                baseline=AttributionWindow(start=baseline_start, end=baseline_end),
                current=AttributionWindow(start=current_start, end=current_end),
                where_sql=where,
                time_dimension=time_dimension,
                max_values_per_dimension=max_dimension_values,
                top_n_dimensions=max_selected_dimensions,
                top_n_values=top_n_values,
                params=params or {},
            )
            result = _run_async(self._attribute(runtime, request))

            return FuncToolResult(
                success=1,
                result=result.model_dump(exclude_none=True),
            )

        except Exception as e:
            payload = getattr(e, "payload", None)
            if payload is not None and getattr(payload, "error_type", None) == "semantic_validation_error":
                data = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
                logger.debug("attribution validation rejection: code=%s", data.get("code"))
                return FuncToolResult(
                    success=0,
                    error=getattr(payload, "message", "") or "attribution validation failed",
                    result=data,
                )
            logger.error(f"Error in attribution analysis: {e}")
            return FuncToolResult(
                success=0,
                error=f"Failed to analyze attribution: {str(e)}",
                result={"execution_info": e.execution_info} if getattr(e, "execution_info", None) else None,
            )
