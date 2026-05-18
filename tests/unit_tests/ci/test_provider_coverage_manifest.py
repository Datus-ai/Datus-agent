from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

MODULE_PATH = Path(__file__).resolve().parents[3] / "ci" / "provider_coverage_manifest.py"
REPO_ROOT = MODULE_PATH.parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("provider_coverage_manifest", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load provider_coverage_manifest module from {MODULE_PATH}")
provider_coverage_manifest = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(provider_coverage_manifest)


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def make_args(tmp_path: Path, catalog: Path, coverage: Path, nightly_manifest: Path | None = None):
    return type(
        "Args",
        (),
        {
            "repo_root": str(REPO_ROOT),
            "provider_catalog": str(catalog),
            "coverage_config": str(coverage),
            "nightly_manifest": str(nightly_manifest or tmp_path / "missing-nightly-manifest.json"),
            "output": str(tmp_path / "provider-coverage-manifest.json"),
            "strict": False,
        },
    )()


def test_build_manifest_matches_collected_provider_health_nodeids(tmp_path):
    catalog = tmp_path / "providers.yml"
    coverage = tmp_path / "provider-coverage.yml"
    nightly_manifest = tmp_path / "nightly-manifest.json"
    write_yaml(
        catalog,
        {
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-5.5",
                    "models": ["gpt-5.5"],
                }
            }
        },
    )
    write_yaml(
        coverage,
        {
            "version": 1,
            "providers": {
                "openai": {
                    "required_env": ["OPENAI_API_KEY"],
                    "deterministic": {"nodeids": ["tests/unit_tests/models/test_openai_model.py::TestOpenAIModel"]},
                    "live_provider_health": {
                        "mode": "warn-only",
                        "nodeids": ["tests/integration/models/test_other_models.py::TestOpenAIModel"],
                    },
                }
            },
        },
    )
    write_json(
        nightly_manifest,
        {
            "suites": [
                {
                    "name": "Provider Health Tests",
                    "status": "passed",
                    "mode": "warn-only",
                    "collection": {
                        "nodeids": [
                            "tests/integration/models/test_other_models.py::TestOpenAIModel::test_generate_basic"
                        ]
                    },
                }
            ]
        },
    )

    manifest = provider_coverage_manifest.build_manifest(make_args(tmp_path, catalog, coverage, nightly_manifest))

    assert manifest["coverage_errors"] == []
    assert manifest["nightly"]["provider_health_collected_nodeids"] == 1
    assert manifest["summary"]["providers_total"] == 1
    assert manifest["summary"]["deterministic_covered"] == 1
    assert manifest["summary"]["live_provider_health_declared"] == 1
    assert manifest["summary"]["live_provider_health_collected"] == 1
    provider = manifest["providers"][0]
    assert provider["required_env"] == ["OPENAI_API_KEY"]
    assert provider["coverage"]["live_provider_health"]["status"] == "collected"
    assert provider["coverage"]["live_provider_health"]["collected_nodeids"] == [
        "tests/integration/models/test_other_models.py::TestOpenAIModel::test_generate_basic"
    ]


def test_build_manifest_reports_missing_declarations(tmp_path):
    catalog = tmp_path / "providers.yml"
    coverage = tmp_path / "provider-coverage.yml"
    write_yaml(
        catalog,
        {
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-5.5",
                    "models": ["gpt-5.5"],
                }
            }
        },
    )
    write_yaml(coverage, {"version": 1, "providers": {}})

    manifest = provider_coverage_manifest.build_manifest(make_args(tmp_path, catalog, coverage))

    assert manifest["summary"]["coverage_error_count"] == 1
    assert manifest["coverage_errors"] == ["openai: provider has no coverage declaration"]
    assert manifest["summary"]["missing_declared_coverage"] == ["openai"]


def test_class_or_function_exists_matches_async_functions(tmp_path):
    test_file = tmp_path / "tests" / "test_async_provider.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "class TestProvider:\n"
        "    async def test_async_method(self):\n"
        "        pass\n\n"
        "async def test_async_function():\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert provider_coverage_manifest.class_or_function_exists(
        tmp_path,
        "tests/test_async_provider.py::test_async_function",
    )
    assert provider_coverage_manifest.class_or_function_exists(
        tmp_path,
        "tests/test_async_provider.py::TestProvider::test_async_method",
    )


def test_build_manifest_reports_invalid_list_fields_without_raising(tmp_path):
    catalog = tmp_path / "providers.yml"
    coverage = tmp_path / "provider-coverage.yml"
    nightly_manifest = tmp_path / "nightly-manifest.json"
    write_yaml(
        catalog,
        {
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_key_env": "OPENAI_API_KEY",
                    "default_model": "gpt-5.5",
                    "models": "gpt-5.5",
                }
            }
        },
    )
    write_yaml(
        coverage,
        {
            "version": 1,
            "providers": {
                "openai": {
                    "required_env": "OPENAI_API_KEY",
                    "deterministic": {"nodeids": "tests/unit_tests/models/test_openai_model.py"},
                    "live_provider_health": {"mode": "warn-only", "nodeids": [123]},
                }
            },
        },
    )
    write_json(
        nightly_manifest,
        {
            "suites": [
                {
                    "name": "Provider Health Tests",
                    "collection": {"nodeids": "tests/integration/models/test_other_models.py"},
                }
            ]
        },
    )

    manifest = provider_coverage_manifest.build_manifest(make_args(tmp_path, catalog, coverage, nightly_manifest))

    assert manifest["providers"][0]["models"] == []
    assert manifest["providers"][0]["required_env"] == ["OPENAI_API_KEY"]
    assert manifest["nightly"]["provider_health_collected_nodeids"] == 0
    assert manifest["coverage_errors"] == [
        "openai.deterministic.nodeids: expected list[str], got 'tests/unit_tests/models/test_openai_model.py'",
        "openai.live_provider_health.nodeids: expected list[str], got [123]",
        "openai.required_env: expected list[str], got 'OPENAI_API_KEY'",
        "nightly.provider_health.collection.nodeids: expected list[str], got 'tests/integration/models/test_other_models.py'",
        "openai.models: expected list[str], got 'gpt-5.5'",
    ]
    assert manifest["summary"]["coverage_error_count"] == 5


def test_repo_provider_coverage_config_is_valid():
    catalog = provider_coverage_manifest.read_yaml(REPO_ROOT / "conf" / "providers.yml")
    coverage = provider_coverage_manifest.read_yaml(REPO_ROOT / "ci" / "provider-coverage.yml")

    assert provider_coverage_manifest.validate_coverage_config(REPO_ROOT, catalog, coverage) == []
