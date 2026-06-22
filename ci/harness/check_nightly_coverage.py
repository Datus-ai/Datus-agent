#!/usr/bin/env python3
"""Verify coverage-map nightly claims against the nightly manifest and sync issues.

Two responsibilities (both non-blocking):

A. Reality verification — cross-check each flow's ``coverage.nightly`` claim
   against the nodeids the latest nightly run actually collected and executed
   (from ``nightly-manifest.json``). A flow that claims ``covered`` but whose
   nodeids were never collected/run is **drift**.
B. Per-flow issue sync — for every flow that is ``drift`` or a declared gap
   (``status in {gap, partial}``), maintain one GitHub tracking issue
   (create/update/reopen), and close it once the flow is healthy again.

Whitelist: ``status in {manual, external, not_applicable}`` are deliberate,
PR-reviewed non-coverage decisions and never produce an issue.

See docs/superpowers/specs/2026-06-20-nightly-coverage-gap-detection-design.md.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

# Statuses that mean "intentionally not auto-covered" — never flagged/issued.
WHITELIST_STATUSES = {"manual", "external", "not_applicable"}
# Statuses that are a declared backlog gap — always tracked with an issue.
GAP_STATUSES = {"gap", "partial"}

ISSUE_LABEL = "coverage-gap"
MARKER_RE = re.compile(r"<!--\s*coverage-flow:(?P<flow_id>[^\s>]+)\s*-->")


# ── data model ──────────────────────────────────────────────────────────────


@dataclass
class FlowResult:
    """Outcome of classifying one flow's nightly coverage."""

    flow_id: str
    group_id: str
    title: str
    priority: str
    status: str
    docs: list[str] = field(default_factory=list)
    nightly_nodeids: list[str] = field(default_factory=list)
    notes: str = ""
    gaps: list[str] = field(default_factory=list)
    # Classification outcome.
    kind: str = "ok"  # one of: ok, drift, declared_gap
    missing_nodeids: list[str] = field(default_factory=list)

    @property
    def needs_issue(self) -> bool:
        return self.kind in ("drift", "declared_gap")


# ── pure logic ──────────────────────────────────────────────────────────────


def ran_covers(map_nodeid: str, ran_set: Iterable[str]) -> bool:
    """Does any executed nodeid satisfy a coverage-map nodeid claim?

    Matching is granularity-aware: a file/class claim is satisfied by any ran
    nodeid *under* it; a function claim by an exact match or a parametrized
    variant. The ``::`` / ``[`` boundaries avoid prefix false positives
    (``test_x.py`` must not match ``test_x_extra.py``).
    """
    for ran in ran_set:
        if ran == map_nodeid:
            return True
        if ran.startswith(map_nodeid + "::"):
            return True
        if ran.startswith(map_nodeid + "["):
            return True
    return False


def build_ran_set(manifest: dict[str, Any]) -> set[str]:
    """Union of collected nodeids across nightly suites that were executed.

    A whole-suite ``skipped`` (e.g. filtered out by NIGHTLY_GROUP_FILTER)
    contributes nothing. Suite red/green is a separate signal — drift only asks
    "was it collected and run at all".
    """
    ran: set[str] = set()
    for suite in manifest.get("suites", []) or []:
        if not isinstance(suite, dict):
            continue
        # Only suites that were actually executed contribute nodeids; a
        # whole-suite skip (or any non-executed status) does not.
        if suite.get("status") not in ("passed", "failed"):
            continue
        collection = suite.get("collection")
        if not isinstance(collection, dict):
            continue
        for nodeid in collection.get("nodeids", []) or []:
            if isinstance(nodeid, str) and nodeid:
                ran.add(nodeid)
    return ran


def iter_flow_dicts(coverage_map: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(group_id, flow_dict)`` for every flow in the map."""
    groups = coverage_map.get("flow_groups")
    if not isinstance(groups, dict):
        return
    for group_id, group in groups.items():
        if not isinstance(group, dict):
            continue
        for flow in group.get("flows", []) or []:
            if isinstance(flow, dict):
                yield str(group_id), flow


def classify_flow(group_id: str, flow: dict[str, Any], ran_set: set[str]) -> FlowResult:
    """Classify one flow's nightly coverage as ok / drift / declared_gap."""
    coverage = flow.get("coverage") if isinstance(flow.get("coverage"), dict) else {}
    nightly = coverage.get("nightly") if isinstance(coverage.get("nightly"), dict) else {}
    status = str(nightly.get("status", "")) or "missing"
    nodeids = [n for n in (nightly.get("nodeids") or []) if isinstance(n, str)]

    result = FlowResult(
        flow_id=str(flow.get("id", "")),
        group_id=group_id,
        title=str(flow.get("title", "")),
        priority=str(flow.get("priority", "")),
        status=status,
        docs=[d for d in (flow.get("docs") or []) if isinstance(d, str)],
        nightly_nodeids=nodeids,
        notes=str(nightly.get("notes", "") or "").strip(),
        gaps=[g for g in (flow.get("gaps") or []) if isinstance(g, str)],
    )

    if status in WHITELIST_STATUSES:
        result.kind = "ok"
    elif status in GAP_STATUSES:
        result.kind = "declared_gap"
    elif status == "covered":
        missing = [nid for nid in nodeids if not ran_covers(nid, ran_set)]
        if missing:
            result.kind = "drift"
            result.missing_nodeids = missing
        else:
            result.kind = "ok"
    else:  # unknown / missing status — treat as a gap so it surfaces.
        result.kind = "declared_gap"
    return result


def classify_all(coverage_map: dict[str, Any], ran_set: set[str]) -> list[FlowResult]:
    return [classify_flow(group_id, flow, ran_set) for group_id, flow in iter_flow_dicts(coverage_map)]


# ── rendering ───────────────────────────────────────────────────────────────


def marker(flow_id: str) -> str:
    return f"<!-- coverage-flow:{flow_id} -->"


def parse_flow_id(body: str) -> str | None:
    match = MARKER_RE.search(body or "")
    return match.group("flow_id") if match else None


def issue_title(result: FlowResult) -> str:
    return f"[coverage] {result.title} ({result.flow_id})"


def render_issue_body(result: FlowResult, run_url: str = "") -> str:
    lines = [
        marker(result.flow_id),
        "",
        "_Auto-maintained by `ci/harness/check_nightly_coverage.py`. Do not edit the marker above._",
        "",
        f"- **Flow:** `{result.flow_id}` — {result.title}",
        f"- **Priority:** {result.priority}",
        f"- **Nightly status (coverage-map):** `{result.status}`",
        f"- **Classification:** `{result.kind}`",
    ]
    if result.kind == "drift":
        lines.append("")
        lines.append("### Drift: claimed covered but not run in the latest nightly")
        lines.append("These nodeids are declared under `coverage.nightly` but were not collected/executed:")
        lines.extend(f"- `{nid}`" for nid in result.missing_nodeids)
        lines.append("")
        lines.append(
            "Resolve by either (a) fixing the test/marker so it runs in nightly, or "
            "(b) re-statusing the flow to `manual`/`external`/`not_applicable` with a rationale."
        )
    elif result.kind == "declared_gap":
        lines.append("")
        lines.append(f"### Declared nightly gap (`{result.status}`)")
        lines.append("This flow is documented but its nightly coverage is incomplete.")
    if result.nightly_nodeids:
        lines.append("")
        lines.append("**Declared nightly nodeids:**")
        lines.extend(f"- `{nid}`" for nid in result.nightly_nodeids)
    if result.docs:
        lines.append("")
        lines.append("**Docs:**")
        lines.extend(f"- `{doc}`" for doc in result.docs)
    if result.notes:
        lines.append("")
        lines.append(f"**Map notes:** {result.notes}")
    if result.gaps:
        lines.append("")
        lines.append("**Related gaps (from coverage-map):**")
        lines.extend(f"- {gap}" for gap in result.gaps)
    if run_url:
        lines.append("")
        lines.append(f"Source nightly run: {run_url}")
    return "\n".join(lines)


def render_digest(results: list[FlowResult]) -> str:
    drift = [r for r in results if r.kind == "drift"]
    gaps = [r for r in results if r.kind == "declared_gap"]
    ok = [r for r in results if r.kind == "ok"]

    def table(rows: list[FlowResult]) -> list[str]:
        if not rows:
            return ["_none_", ""]
        out = ["| Flow | Priority | Status | Detail |", "|---|---|---|---|"]
        for r in rows:
            detail = ", ".join(f"`{n}`" for n in r.missing_nodeids) if r.kind == "drift" else (r.notes[:80] or "")
            out.append(f"| `{r.flow_id}` | {r.priority} | `{r.status}` | {detail} |")
        out.append("")
        return out

    lines = [
        "# Nightly Coverage Gap Report",
        "",
        f"- drift (claimed covered, not run): **{len(drift)}**",
        f"- declared gaps (gap/partial): **{len(gaps)}**",
        f"- ok / whitelisted: **{len(ok)}**",
        "",
        "## Drift",
        *table(drift),
        "## Declared gaps",
        *table(gaps),
        "## OK / whitelisted",
        *table(ok),
    ]
    return "\n".join(lines)


# ── GitHub issue sync ───────────────────────────────────────────────────────


class IssueClient(Protocol):
    """Minimal GitHub issue surface, injectable for tests."""

    def list_issues(self, label: str) -> list[dict[str, Any]]:
        """Return open+closed issues with ``label`` (each has number/state/body)."""

    def create_issue(self, title: str, body: str, label: str) -> dict[str, Any]: ...

    def update_issue(self, number: int, title: str, body: str) -> None: ...

    def reopen_issue(self, number: int) -> None: ...

    def comment_issue(self, number: int, body: str) -> None: ...

    def close_issue(self, number: int) -> None: ...


def _index_issues_by_flow(client: IssueClient) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for issue in client.list_issues(ISSUE_LABEL):
        flow_id = parse_flow_id(issue.get("body", "") or "")
        if flow_id and flow_id not in index:
            index[flow_id] = issue
    return index


def sync_issues(
    results: list[FlowResult],
    client: IssueClient,
    run_url: str = "",
) -> list[dict[str, str]]:
    """Reconcile per-flow tracking issues with the current classification.

    Only touches issues carrying the ``coverage-flow`` marker + label; never
    human-authored issues. Returns a list of ``{action, flow_id}`` for logging.
    """
    actions: list[dict[str, str]] = []
    existing = _index_issues_by_flow(client)

    for result in results:
        try:
            actions.append(_sync_one(result, existing.get(result.flow_id), client, run_url))
        except Exception as exc:  # noqa: BLE001 - one flow's API failure must not abort the rest
            actions.append({"action": "error", "flow_id": result.flow_id, "error": str(exc)})

    return [a for a in actions if a["action"] != "noop"]


def _sync_one(
    result: FlowResult,
    issue: dict[str, Any] | None,
    client: IssueClient,
    run_url: str,
) -> dict[str, str]:
    if result.needs_issue:
        body = render_issue_body(result, run_url)
        title = issue_title(result)
        if issue is None:
            client.create_issue(title, body, ISSUE_LABEL)
            return {"action": "created", "flow_id": result.flow_id}
        number = int(issue["number"])
        if str(issue.get("state", "")).lower() == "closed":
            client.reopen_issue(number)
        client.update_issue(number, title, body)
        return {"action": "updated", "flow_id": result.flow_id}

    # Flow is healthy/whitelisted: close any open bot issue.
    if issue is not None and str(issue.get("state", "")).lower() == "open":
        number = int(issue["number"])
        client.comment_issue(number, "Resolved: this flow is now covered or whitelisted. Closing automatically.")
        client.close_issue(number)
        return {"action": "closed", "flow_id": result.flow_id}
    return {"action": "noop", "flow_id": result.flow_id}


# ── CLI ─────────────────────────────────────────────────────────────────────


class GitHubCliIssueClient:
    """IssueClient backed by the ``gh`` CLI for a single repo (used by main())."""

    def __init__(self, repo: str, *, limit: int = 1000) -> None:
        self.repo = repo
        self.limit = limit

    def _run(self, *args: str, capture: bool = False) -> str:
        import subprocess

        result = subprocess.run(["gh", *args, "--repo", self.repo], check=True, text=True, capture_output=capture)
        return result.stdout if capture else ""

    def list_issues(self, label: str) -> list[dict[str, Any]]:
        out = self._run(
            "issue",
            "list",
            "--label",
            label,
            "--state",
            "all",
            "--limit",
            str(self.limit),
            "--json",
            "number,state,title,body",
            capture=True,
        )
        data = json.loads(out or "[]")
        return data if isinstance(data, list) else []

    def create_issue(self, title: str, body: str, label: str) -> dict[str, Any]:
        url = self._run("issue", "create", "--title", title, "--body", body, "--label", label, capture=True)
        return {"url": url.strip()}

    def update_issue(self, number: int, title: str, body: str) -> None:
        self._run("issue", "edit", str(number), "--title", title, "--body", body)

    def reopen_issue(self, number: int) -> None:
        self._run("issue", "reopen", str(number))

    def comment_issue(self, number: int, body: str) -> None:
        self._run("issue", "comment", str(number), "--body", body)

    def close_issue(self, number: int) -> None:
        self._run("issue", "close", str(number))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_coverage_map(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-map", default="ci/harness/coverage-map.yml")
    parser.add_argument("--nightly-manifest", default="nightly-manifest.json")
    parser.add_argument("--output", default="coverage-gap-report.md")
    parser.add_argument("--repo", default="Datus-ai/Datus-agent")
    parser.add_argument("--run-url", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and write the digest, but make no GitHub writes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    coverage_map = load_coverage_map(Path(args.coverage_map))
    manifest = load_json(Path(args.nightly_manifest))
    ran_set = build_ran_set(manifest)
    # Gate on actually-executed nodeids, not just a truthy ``suites`` field: a
    # missing / corrupt / skipped-only / malformed manifest yields an empty ran
    # set, which would make every covered flow look like drift and (in live mode)
    # create a wave of false issues. No executed nodeids → skip issue sync.
    manifest_available = bool(ran_set)
    results = classify_all(coverage_map, ran_set)

    digest = render_digest(results)
    if not manifest_available:
        digest = f"> ⚠️ no executed nodeids in the nightly manifest — drift results below are not reliable.\n\n{digest}"
    Path(args.output).write_text(digest + "\n", encoding="utf-8")
    print(digest)

    if not manifest_available:
        print("\n[warn] nightly manifest unavailable or had no executed nodeids; skipping issue sync")
        return 0

    if args.dry_run:
        needs = [r for r in results if r.needs_issue]
        print(f"\n[dry-run] would sync {len(needs)} issue(s): {[r.flow_id for r in needs]}")
        return 0

    client = GitHubCliIssueClient(repo=args.repo)
    actions = sync_issues(results, client, run_url=args.run_url)
    for action in actions:
        print(f"[issue] {action['action']}: {action['flow_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
