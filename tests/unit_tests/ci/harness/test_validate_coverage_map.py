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
