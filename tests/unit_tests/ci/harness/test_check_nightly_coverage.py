# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ci/harness/check_nightly_coverage.py (deterministic, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[4] / "ci" / "harness" / "check_nightly_coverage.py"
_spec = importlib.util.spec_from_file_location("check_nightly_coverage", _MODULE_PATH)
cnc = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(cnc)


# ── ran_covers: granularity matrix ──────────────────────────────────────────


@pytest.mark.parametrize(
    "map_nodeid, ran, expected",
    [
        # file-level claim
        (
            "tests/integration/storage/test_platform_doc.py",
            {"tests/integration/storage/test_platform_doc.py::C::t"},
            True,
        ),
        ("tests/integration/storage/test_platform_doc.py", {"tests/integration/storage/test_other.py::C::t"}, False),
        # prefix boundary: must not match a sibling file with a longer name
        ("tests/a/test_x.py", {"tests/a/test_x_extra.py::C::t"}, False),
        # class-level claim
        ("tests/a/test_x.py::TestA", {"tests/a/test_x.py::TestA::test_y"}, True),
        ("tests/a/test_x.py::TestA", {"tests/a/test_x.py::TestB::test_y"}, False),
        # function-level: exact
        ("tests/a/test_x.py::TestA::test_y", {"tests/a/test_x.py::TestA::test_y"}, True),
        # function-level: parametrized variant
        ("tests/a/test_x.py::TestA::test_y", {"tests/a/test_x.py::TestA::test_y[case1]"}, True),
        # function-level: a different function with a shared prefix must not match
        ("tests/a/test_x.py::TestA::test_y", {"tests/a/test_x.py::TestA::test_y2"}, False),
    ],
)
def test_ran_covers_granularity(map_nodeid: str, ran: set[str], expected: bool) -> None:
    assert cnc.ran_covers(map_nodeid, ran) is expected


# ── build_ran_set ───────────────────────────────────────────────────────────


def test_build_ran_set_unions_executed_suites_and_skips_skipped() -> None:
    manifest = {
        "suites": [
            {"name": "A", "status": "passed", "collection": {"nodeids": ["t/a.py::t1", "t/a.py::t2"]}},
            {"name": "B", "status": "failed", "collection": {"nodeids": ["t/b.py::t1"]}},
            # a whole-suite skip must contribute nothing even if it has a collection
            {"name": "C", "status": "skipped", "collection": {"nodeids": ["t/c.py::t1"]}},
            # a suite without a collection block is ignored
            {"name": "D", "status": "passed"},
        ]
    }
    assert cnc.build_ran_set(manifest) == {"t/a.py::t1", "t/a.py::t2", "t/b.py::t1"}


def test_build_ran_set_handles_empty_or_garbled() -> None:
    assert cnc.build_ran_set({}) == set()
    assert cnc.build_ran_set({"suites": None}) == set()
    assert cnc.build_ran_set({"suites": ["not-a-dict"]}) == set()


# ── classification ──────────────────────────────────────────────────────────


def _flow(flow_id: str, status: str, nodeids: list[str], **extra: Any) -> dict[str, Any]:
    flow: dict[str, Any] = {
        "id": flow_id,
        "title": extra.get("title", flow_id),
        "priority": extra.get("priority", "p1"),
        "coverage": {"nightly": {"status": status, "nodeids": nodeids}},
    }
    if "notes" in extra:
        flow["coverage"]["nightly"]["notes"] = extra["notes"]
    if "docs" in extra:
        flow["docs"] = extra["docs"]
    return flow


def test_classify_covered_and_ran_is_ok() -> None:
    flow = _flow("g.f", "covered", ["t/a.py::t1"])
    result = cnc.classify_flow("g", flow, {"t/a.py::t1"})
    assert result.kind == "ok"
    assert result.needs_issue is False


def test_classify_covered_but_not_run_is_drift() -> None:
    # The platform_doc-orphan scenario: claims covered, file never collected.
    flow = _flow("kb.platform", "covered", ["tests/integration/storage/test_platform_doc.py"])
    result = cnc.classify_flow("kb", flow, ran_set=set())
    assert result.kind == "drift"
    assert result.missing_nodeids == ["tests/integration/storage/test_platform_doc.py"]
    assert result.needs_issue is True


def test_classify_partial_drift_lists_only_missing_nodeids() -> None:
    flow = _flow("g.f", "covered", ["t/a.py::t1", "t/b.py::t2"])
    result = cnc.classify_flow("g", flow, {"t/a.py::t1"})  # only the first ran
    assert result.kind == "drift"
    assert result.missing_nodeids == ["t/b.py::t2"]


@pytest.mark.parametrize("status", ["gap", "partial"])
def test_classify_declared_gap_needs_issue(status: str) -> None:
    result = cnc.classify_flow("g", _flow("g.f", status, []), set())
    assert result.kind == "declared_gap"
    assert result.needs_issue is True


@pytest.mark.parametrize("status", ["manual", "external", "not_applicable"])
def test_classify_whitelist_never_needs_issue(status: str) -> None:
    # Even with declared nodeids that did not run, whitelist statuses are ok.
    result = cnc.classify_flow("g", _flow("g.f", status, ["t/a.py::t1"]), set())
    assert result.kind == "ok"
    assert result.needs_issue is False


def test_classify_unknown_status_surfaces_as_gap() -> None:
    flow = {"id": "g.f", "title": "F", "coverage": {"nightly": {"status": "", "nodeids": []}}}
    assert cnc.classify_flow("g", flow, set()).kind == "declared_gap"


def test_classify_all_iterates_groups_and_flows() -> None:
    coverage_map = {
        "flow_groups": {
            "g1": {"flows": [_flow("g1.a", "covered", ["t/a.py::t1"]), _flow("g1.b", "gap", [])]},
            "g2": {"flows": [_flow("g2.c", "manual", [])]},
        }
    }
    results = cnc.classify_all(coverage_map, {"t/a.py::t1"})
    by_id = {r.flow_id: r.kind for r in results}
    assert by_id == {"g1.a": "ok", "g1.b": "declared_gap", "g2.c": "ok"}


# ── marker round-trip + rendering ───────────────────────────────────────────


def test_marker_round_trips_through_issue_body() -> None:
    flow = _flow("kb.platform_doc", "gap", ["t/a.py::t1"])
    result = cnc.classify_flow("kb", flow, set())
    body = cnc.render_issue_body(result, run_url="https://example/run/1")
    assert cnc.parse_flow_id(body) == "kb.platform_doc"


def test_drift_body_lists_missing_nodeids() -> None:
    flow = _flow("g.f", "covered", ["t/a.py::missing"])
    result = cnc.classify_flow("g", flow, set())
    body = cnc.render_issue_body(result)
    assert "Drift" in body
    assert "t/a.py::missing" in body


def test_parse_flow_id_returns_none_without_marker() -> None:
    assert cnc.parse_flow_id("just some text, no marker") is None


# ── sync_issues decision table ──────────────────────────────────────────────


class FakeIssueClient:
    """In-memory IssueClient double recording every call."""

    def __init__(self, issues: list[dict[str, Any]] | None = None) -> None:
        self.issues = issues or []
        self.calls: list[tuple[str, Any]] = []
        self._next_number = 1000

    def list_issues(self, label: str) -> list[dict[str, Any]]:
        self.calls.append(("list", label))
        return list(self.issues)

    def create_issue(self, title: str, body: str, label: str) -> dict[str, Any]:
        self.calls.append(("create", title))
        self._next_number += 1
        return {"number": self._next_number}

    def update_issue(self, number: int, title: str, body: str) -> None:
        self.calls.append(("update", number))

    def reopen_issue(self, number: int) -> None:
        self.calls.append(("reopen", number))

    def comment_issue(self, number: int, body: str) -> None:
        self.calls.append(("comment", number))

    def close_issue(self, number: int) -> None:
        self.calls.append(("close", number))


def _result(flow_id: str, kind: str) -> "cnc.FlowResult":
    return cnc.FlowResult(flow_id=flow_id, group_id="g", title=flow_id, priority="p1", status=kind, kind=kind)


def test_sync_creates_issue_when_gap_and_none_exists() -> None:
    client = FakeIssueClient(issues=[])
    actions = cnc.sync_issues([_result("g.f", "declared_gap")], client)
    assert actions == [{"action": "created", "flow_id": "g.f"}]
    assert ("create", cnc.issue_title(_result("g.f", "declared_gap"))) in client.calls


def test_sync_updates_open_issue_when_still_gap() -> None:
    issues = [{"number": 7, "state": "open", "body": cnc.marker("g.f")}]
    client = FakeIssueClient(issues=issues)
    actions = cnc.sync_issues([_result("g.f", "drift")], client)
    assert actions == [{"action": "updated", "flow_id": "g.f"}]
    assert ("update", 7) in client.calls
    assert ("reopen", 7) not in client.calls


def test_sync_reopens_then_updates_closed_issue_when_gap_returns() -> None:
    issues = [{"number": 8, "state": "closed", "body": cnc.marker("g.f")}]
    client = FakeIssueClient(issues=issues)
    cnc.sync_issues([_result("g.f", "declared_gap")], client)
    assert ("reopen", 8) in client.calls
    assert ("update", 8) in client.calls


def test_sync_closes_open_issue_when_flow_healthy() -> None:
    issues = [{"number": 9, "state": "open", "body": cnc.marker("g.f")}]
    client = FakeIssueClient(issues=issues)
    actions = cnc.sync_issues([_result("g.f", "ok")], client)
    assert actions == [{"action": "closed", "flow_id": "g.f"}]
    assert ("comment", 9) in client.calls
    assert ("close", 9) in client.calls


def test_sync_ignores_healthy_flow_without_existing_issue() -> None:
    client = FakeIssueClient(issues=[])
    actions = cnc.sync_issues([_result("g.f", "ok")], client)
    assert actions == []  # noop filtered out


def test_sync_never_touches_unmarked_issues() -> None:
    # A human issue carrying the label but no marker must be left alone.
    issues = [{"number": 910, "state": "open", "body": "human-authored, no marker"}]
    client = FakeIssueClient(issues=issues)
    cnc.sync_issues([_result("g.f", "declared_gap")], client)
    # It created a NEW issue (didn't reuse 910) and never updated/closed 910.
    assert ("create", cnc.issue_title(_result("g.f", "declared_gap"))) in client.calls
    assert all(call[0] not in ("update", "close", "comment", "reopen") or call[1] != 910 for call in client.calls)


def test_sync_continues_after_one_flow_errors() -> None:
    class FlakyClient(FakeIssueClient):
        def create_issue(self, title: str, body: str, label: str) -> dict[str, Any]:
            if "boom" in title:
                raise RuntimeError("gh failed")
            return super().create_issue(title, body, label)

    client = FlakyClient(issues=[])
    results = [_result("g.boom", "declared_gap"), _result("g.ok", "declared_gap")]
    actions = cnc.sync_issues(results, client)
    kinds = {a["flow_id"]: a["action"] for a in actions}
    assert kinds["g.boom"] == "error"
    assert kinds["g.ok"] == "created"


# ── digest ──────────────────────────────────────────────────────────────────


def test_render_digest_counts_each_bucket() -> None:
    results = [
        _result("a", "drift"),
        _result("b", "declared_gap"),
        _result("c", "declared_gap"),
        _result("d", "ok"),
    ]
    digest = cnc.render_digest(results)
    assert "drift (claimed covered, not run): **1**" in digest
    assert "declared gaps (gap/partial): **2**" in digest
    assert "ok / whitelisted: **1**" in digest


def test_digest_includes_ok_whitelisted_section() -> None:
    digest = cnc.render_digest([_result("kept.ok", "ok")])
    assert "## OK / whitelisted" in digest
    assert "`kept.ok`" in digest


# ── review fixes ─────────────────────────────────────────────────────────────


def test_build_ran_set_excludes_non_executed_status() -> None:
    # Only passed/failed contribute; an unknown/errored status must not.
    manifest = {
        "suites": [
            {"name": "ok", "status": "passed", "collection": {"nodeids": ["t/a.py::t1"]}},
            {"name": "weird", "status": "errored", "collection": {"nodeids": ["t/b.py::t2"]}},
        ]
    }
    assert cnc.build_ran_set(manifest) == {"t/a.py::t1"}


def test_issue_body_includes_related_gaps() -> None:
    flow = _flow(
        "g.f",
        "gap",
        [],
    )
    result = cnc.classify_flow("g", flow, set())
    result.gaps = ["https://github.com/Datus-ai/Datus-agent/issues/910"]
    body = cnc.render_issue_body(result)
    assert "Related gaps" in body
    assert "issues/910" in body


def _write_cov_map(path: Path) -> Path:
    path.write_text(
        "flow_groups:\n"
        "  g:\n"
        "    flows:\n"
        "      - id: g.f\n"
        "        title: F\n"
        "        coverage:\n"
        "          nightly:\n"
        "            status: covered\n"
        "            nodeids: [t/a.py::t1]\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "manifest_text",
    [
        None,  # missing file
        '{"suites": []}',  # empty
        '{"suites": [{"status": "skipped", "collection": {"nodeids": ["t/a.py::t1"]}}]}',  # skipped-only
        '{"suites": "bad"}',  # malformed but truthy
    ],
)
def test_main_skips_issue_sync_without_executed_nodeids(tmp_path: Path, manifest_text: str | None) -> None:
    # Any manifest that yields no *executed* nodeids must skip issue sync (else
    # live mode would mass-create false-drift issues). No --dry-run: the guard
    # must short-circuit before any GitHub client is constructed.
    cov_map = _write_cov_map(tmp_path / "coverage-map.yml")
    manifest = tmp_path / "manifest.json"
    if manifest_text is not None:
        manifest.write_text(manifest_text, encoding="utf-8")
    out = tmp_path / "report.md"

    rc = cnc.main(["--coverage-map", str(cov_map), "--nightly-manifest", str(manifest), "--output", str(out)])

    assert rc == 0
    assert "no executed nodeids" in out.read_text(encoding="utf-8")
