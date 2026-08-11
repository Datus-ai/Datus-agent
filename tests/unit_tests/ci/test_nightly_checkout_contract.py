import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-nightly.yml"
NIGHTLY_SCRIPT = REPO_ROOT / "ci" / "run-nightly-tests.sh"


def _bash_function_source(name: str) -> str:
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + len("\n}\n")
    return script[start:end]


def test_nightly_preserves_checkout_packages_after_locked_sync():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'UV_NO_SYNC: "1"' in workflow
    assert "uv sync --locked" in workflow
    assert "uv run --no-sync python ci/verify_nightly_adapter_sources.py" in workflow
    assert "uv run --no-sync playwright install --with-deps chromium" in workflow


def test_nightly_kb_cache_tracks_all_adapter_checkouts():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for repo in (
        "external/datus-db-adapters",
        "external/datus-bi-adapters",
        "external/datus-scheduler-adapters",
        "external/datus-semantic-adapter",
        "external/datus-storage-adapters",
    ):
        assert repo in workflow

    assert 'git -C "$repo" rev-parse HEAD' in workflow
    assert "kb-v6-datus_agent_nightly" in workflow


def test_nightly_installs_storage_packages_from_latest_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: Datus-ai/datus-storage-adapters" in workflow
    assert "ref: main" in workflow
    assert "path: external/datus-storage-adapters" in workflow
    for package_name, package_path in (
        ("datus-storage-base", "./external/datus-storage-adapters/datus-storage-base"),
        ("datus-storage-postgresql", '"./external/datus-storage-adapters/datus-storage-postgresql[dev]"'),
    ):
        assert f"--reinstall-package {package_name}" in workflow
        assert package_path in workflow


def test_nightly_installs_new_database_adapters_from_latest_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for package_name in ("datus-doris", "datus-hologres", "datus-oracle"):
        assert f"--reinstall-package {package_name}" in workflow
        assert f"./external/datus-db-adapters/{package_name}" in workflow


def test_nightly_runs_postgresql_storage_adapter_tests_from_checkout():
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'STORAGE_ADAPTERS_ROOT="$(default_repo_root' in script
    assert 'run_logged "PostgreSQL Storage Adapter Tests"' in script
    assert 'uv run --no-sync pytest "$STORAGE_ADAPTERS_ROOT/datus-storage-postgresql/tests"' in script
    assert '"PostgreSQL Storage Adapter Tests"' in script.split("DOCKER_GROUPS=(", maxsplit=1)[1]


def test_nightly_runs_doris_agent_contract_from_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "ADAPTERS_DORIS=1" >> $GITHUB_ENV' in workflow
    assert 'echo "DORIS_QUERY_HOST_PORT=29031" >> $GITHUB_ENV' in workflow
    assert 'echo "DORIS_HTTP_HOST_PORT=28031" >> $GITHUB_ENV' in workflow
    assert 'DORIS_COMPOSE="${DORIS_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-doris/docker-compose.yml}"' in script
    assert 'run_compose_suite "Doris Adapter Tests"' in script
    assert 'uv run --no-sync python "$DB_ADAPTERS_ROOT/datus-doris/scripts/wait_for_doris.py"' in script
    assert 'wait_for_doris_client_readiness "${DORIS_READY_TIMEOUT:-600}"' in script
    assert 'wait_for_tcp_readiness "Doris"' not in script
    assert "tests/integration/adapters/test_doris.py" in script


def test_doris_readiness_failure_is_logged_and_propagated(tmp_path):
    log_file = tmp_path / "nightly.log"
    function_source = _bash_function_source("wait_for_doris_client_readiness")
    harness = f"""
set -u
set -o pipefail
test_exit_code=0
uv() {{
  printf 'uv_args=%s\n' "$*"
  echo "simulated Doris readiness failure"
  return 23
}}
{function_source}
wait_for_doris_client_readiness 17
readiness_status=$?
printf 'readiness_status=%s test_exit_code=%s\n' "$readiness_status" "$test_exit_code"
"""

    result = subprocess.run(
        ["bash", "-c", harness],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DB_ADAPTERS_ROOT": str(tmp_path / "db-adapters"),
            "LOG_FILE": str(log_file),
        },
    )

    assert "readiness_status=23 test_exit_code=23" in result.stdout
    assert "--timeout 17" in result.stdout
    assert "simulated Doris readiness failure" in log_file.read_text(encoding="utf-8")
