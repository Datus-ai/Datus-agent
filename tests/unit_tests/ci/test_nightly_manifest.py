from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "nightly_manifest.py"
MODULE_SPEC = importlib.util.spec_from_file_location("nightly_manifest", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load nightly_manifest module from {MODULE_PATH}")
nightly_manifest = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(nightly_manifest)


def test_parse_collection_output_extracts_nodeids_and_counts():
    parsed = nightly_manifest.parse_collection_output(
        "\n".join(
            [
                "DATUS_MANIFEST_NODEIDS_START",
                "DATUS_MANIFEST_NODEID tests/integration/tools/test_mcp_server.py::test_list_tools",
                "DATUS_MANIFEST_NODEID tests/integration/tools/test_mcp_server.py::test_read_query",
                "DATUS_MANIFEST_NODEIDS_END",
                "===== 2 selected, 10 deselected in 0.12s =====",
            ]
        )
    )

    assert parsed["nodeids"] == [
        "tests/integration/tools/test_mcp_server.py::test_list_tools",
        "tests/integration/tools/test_mcp_server.py::test_read_query",
    ]
    assert parsed["counts"] == {"selected": 2, "deselected": 10}


def test_record_collection_and_finalize_manifest(tmp_path):
    manifest_path = tmp_path / "nightly-manifest.json"
    collection_output = tmp_path / "collect.log"

    assert (
        nightly_manifest.main(
            [
                "init",
                "--output",
                str(manifest_path),
                "--repo-root",
                str(Path.cwd()),
                "--external-repos-root",
                str(tmp_path / "external"),
            ]
        )
        == 0
    )
    collection_output.write_text(
        "DATUS_MANIFEST_NODEID tests/unit_tests/ci/test_nightly_manifest.py::test_example\n"
        "===== 1 selected, 3 deselected in 0.01s =====\n",
        encoding="utf-8",
    )

    assert (
        nightly_manifest.main(
            [
                "record-suite",
                "--output",
                str(manifest_path),
                "--name",
                "Example Suite",
                "--mode",
                "blocking",
                "--kind",
                "command",
                "--status",
                "passed",
                "--exit-code",
                "0",
                "--started-at",
                "2026-05-17T00:00:00Z",
                "--ended-at",
                "2026-05-17T00:00:01Z",
                "--command-json",
                json.dumps(["uv", "run", "pytest"]),
            ]
        )
        == 0
    )
    assert (
        nightly_manifest.main(
            [
                "record-collection",
                "--output",
                str(manifest_path),
                "--name",
                "Example Suite",
                "--exit-code",
                "0",
                "--collection-output",
                str(collection_output),
            ]
        )
        == 0
    )
    assert nightly_manifest.main(["finalize", "--output", str(manifest_path), "--exit-code", "0"]) == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["suite_count"] == 1
    assert manifest["summary"]["collected_nodeid_count"] == 1
    assert manifest["suites"][0]["collection"]["counts"] == {"selected": 1, "deselected": 3}
