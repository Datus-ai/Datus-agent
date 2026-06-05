from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

MODULE_PATH = Path(__file__).resolve().parents[4] / "ci" / "harness" / "validate_coverage_map.py"
REPO_ROOT = MODULE_PATH.parents[2]
MODULE_SPEC = importlib.util.spec_from_file_location("validate_coverage_map", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load validate_coverage_map module from {MODULE_PATH}")
validate_coverage_map = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(validate_coverage_map)


def minimal_map(**flow_overrides):
    flow = {
        "id": "onboarding.quickstart",
        "title": "Quickstart",
        "priority": "p0",
        "quality_sensitive": True,
        "docs": ["docs/getting_started/Quickstart.md"],
        "entrypoints": ["CLI: datus"],
        "contracts": ["Quickstart command remains routable."],
        "coverage": {
            "pr_acceptance": {
                "status": "covered",
                "nodeids": ["tests/unit_tests/cli/test_core_entrypoint_acceptance.py"],
            },
            "merge_queue": {"status": "covered"},
            "nightly": {"status": "gap"},
            "weekly_benchmark": {"status": "gap"},
        },
        "gaps": ["https://github.com/Datus-ai/Datus-agent/issues/910"],
    }
    flow.update(flow_overrides)
    return {
        "schema_version": 1,
        "source": {"docs_navigation": "mkdocs.yml"},
        "layers": {
            "pr_acceptance": {"description": "PR"},
            "merge_queue": {"description": "MQ"},
            "nightly": {"description": "Nightly"},
            "weekly_benchmark": {"description": "Weekly"},
        },
        "gap_issue": "https://github.com/Datus-ai/Datus-agent/issues/910",
        "flow_groups": {"onboarding": {"flows": [flow]}},
    }


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def validate(data: dict):
    return validate_coverage_map.validate_coverage_map(REPO_ROOT, data)


def test_minimal_map_with_gap_issue_is_valid():
    errors, flows = validate(minimal_map())

    assert errors == []
    assert [flow.flow_id for flow in flows] == ["onboarding.quickstart"]


def test_missing_docs_path_is_reported():
    errors, _ = validate(minimal_map(docs=["docs/missing.md"]))

    assert errors == ["onboarding.quickstart.docs: docs path does not exist: docs/missing.md"]


def test_missing_nodeid_path_is_reported():
    data = minimal_map()
    data["flow_groups"]["onboarding"]["flows"][0]["coverage"]["pr_acceptance"]["nodeids"] = [
        "tests/unit_tests/missing.py"
    ]

    errors, _ = validate(data)

    assert errors == [
        "onboarding.quickstart.coverage.pr_acceptance: nodeid target does not exist: tests/unit_tests/missing.py"
    ]


def test_p0_requires_deterministic_pr_or_merge_queue_coverage():
    data = minimal_map()
    flow = data["flow_groups"]["onboarding"]["flows"][0]
    flow["coverage"]["pr_acceptance"] = {"status": "external"}
    flow["coverage"]["merge_queue"] = {"status": "gap"}

    errors, _ = validate(data)

    assert "onboarding.quickstart: p0 flow must have deterministic PR or merge queue coverage" in errors


def test_p0_nightly_gap_requires_gap_issue():
    data = minimal_map(gaps=[])

    errors, _ = validate(data)

    assert "onboarding.quickstart: p0 flow must have nightly coverage or an explicit gap issue" in errors


def test_quality_sensitive_weekly_gap_requires_gap_issue():
    data = minimal_map(priority="p1", gaps=[])

    errors, _ = validate(data)

    assert (
        "onboarding.quickstart: quality-sensitive flow must have benchmark coverage or an explicit gap issue" in errors
    )


def test_invalid_gap_reference_is_reported():
    data = minimal_map(gaps=["not-an-issue"])

    errors, _ = validate(data)

    assert "onboarding.quickstart.gaps[0]: invalid GitHub issue reference: not-an-issue" in errors


def test_report_manifest_contains_layer_statuses():
    errors, flows = validate(minimal_map())
    manifest = validate_coverage_map.build_manifest(Path("ci/harness/coverage-map.yml"), flows, errors)

    assert manifest["summary"]["flow_count"] == 1
    assert manifest["flows"][0]["coverage"]["pr_acceptance"]["status"] == "covered"


def test_repo_coverage_map_is_valid():
    coverage = validate_coverage_map.read_yaml(REPO_ROOT / "ci" / "harness" / "coverage-map.yml")

    errors, flows = validate_coverage_map.validate_coverage_map(REPO_ROOT, coverage)

    assert errors == []
    assert len(flows) >= 20


def _docs_coverage_fixture(tmp_path: Path, *, nav_pages: list[str]) -> None:
    """Materialize a tiny mkdocs.yml + docs tree under tmp_path for drift tests."""
    docs_dir = tmp_path / "docs"
    nav_entries = []
    for page in nav_pages:
        target = docs_dir / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# page\n", encoding="utf-8")
        nav_entries.append({page.split("/")[-1]: page})
    (tmp_path / "mkdocs.yml").write_text(yaml.safe_dump({"nav": nav_entries}), encoding="utf-8")


def _docs_coverage_map(*, flow_docs: list[str], exclude: list[dict]) -> dict:
    return {
        "source": {"docs_navigation": "mkdocs.yml"},
        "docs_coverage_policy": {"exclude": exclude},
        "flow_groups": {"onboarding": {"flows": [{"id": "onboarding.quickstart", "docs": flow_docs}]}},
    }


def _flows(coverage_map: dict):
    return validate_coverage_map.iter_flows(coverage_map, [])


def test_docs_coverage_drift_unowned_nav_page_is_reported(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md", "b.md"])
    coverage_map = _docs_coverage_map(flow_docs=["docs/a.md"], exclude=[])

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, _flows(coverage_map), errors)

    assert any("docs/b.md is in mkdocs nav but no flow references it" in error for error in errors)
    assert not any("docs/a.md" in error for error in errors)


def test_docs_coverage_exclude_suppresses_drift(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md", "b.md"])
    coverage_map = _docs_coverage_map(
        flow_docs=["docs/a.md"],
        exclude=[{"path": "docs/b.md", "reason": "Section overview, not a flow."}],
    )

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, _flows(coverage_map), errors)

    assert errors == []


def test_docs_coverage_flow_reference_suppresses_drift(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md", "b.md"])
    coverage_map = _docs_coverage_map(flow_docs=["docs/a.md", "docs/b.md"], exclude=[])

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, _flows(coverage_map), errors)

    assert errors == []


def test_docs_coverage_stale_exclude_is_reported(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md"])
    coverage_map = _docs_coverage_map(
        flow_docs=["docs/a.md"],
        exclude=[{"path": "docs/gone.md", "reason": "no longer documented"}],
    )

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, _flows(coverage_map), errors)

    assert errors == ["docs_coverage_policy.exclude: stale entry not in mkdocs nav: docs/gone.md"]


def test_docs_coverage_exclude_requires_reason(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md", "b.md"])
    coverage_map = _docs_coverage_map(flow_docs=["docs/a.md"], exclude=[{"path": "docs/b.md"}])

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, _flows(coverage_map), errors)

    assert any("reason: required rationale is missing" in error for error in errors)


def test_docs_coverage_is_opt_in_without_policy(tmp_path):
    _docs_coverage_fixture(tmp_path, nav_pages=["a.md", "b.md"])
    coverage_map = {
        "source": {"docs_navigation": "mkdocs.yml"},
        "flow_groups": {"onboarding": {"flows": [{"id": "onboarding.quickstart", "docs": ["docs/a.md"]}]}},
    }

    errors: list[str] = []
    validate_coverage_map.validate_docs_coverage(tmp_path, coverage_map, [], errors)

    assert errors == []
