import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-nightly.yml"
RELEASE_CANDIDATE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-candidate.yml"
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


def test_nightly_feishu_notification_drains_response_and_has_timeout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    notification = workflow.split("- name: Send Feishu notification", maxsplit=1)[1]

    assert "await new Promise((resolve) =>" in notification
    assert "res.resume();" in notification
    assert "res.on('end'" in notification
    assert "req.setTimeout(10_000" in notification


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


def test_nightly_installs_p0_plugins_only_through_managed_store():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for repository, path in (
        ("Datus-ai/Datus-Plugins", "external/Datus-Plugins"),
        ("Datus-ai/datus-sql-policies", "external/datus-sql-policies"),
    ):
        assert f"repository: {repository}" in workflow
        assert f"path: {path}" in workflow

    private_checkout = workflow.split("- name: Checkout datus-sql-policies main", maxsplit=1)[1].split(
        "- name:", maxsplit=1
    )[0]
    assert "token: ${{ secrets.RELEASE_BOT_TOKEN }}" in private_checkout
    assert "github.token" not in private_checkout
    assert "- name: Require private repository checkout token" in workflow

    install_block = workflow.split("- name: Install dependencies", maxsplit=1)[1].split(
        "- name: Ensure Docker runtime", maxsplit=1
    )[0]
    for package_name, package_path in (
        ("datus-superset-plugin", "./external/Datus-Plugins/datus-superset-plugin"),
        ("datus-sql-policies", "./external/datus-sql-policies"),
    ):
        assert f"--reinstall-package {package_name}" not in install_block
        assert package_path not in install_block

    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")
    assert "export LOG_FILE" in script
    assert "EXTERNAL_REPOS_ROOT" in next(line for line in script.splitlines() if line.startswith("export LOG_FILE"))


def test_nightly_installs_new_database_adapters_from_latest_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for package_name in ("datus-doris", "datus-hologres", "datus-oracle", "datus-gaussdb", "datus-tidb"):
        assert f"--reinstall-package {package_name}" in workflow
        assert f"./external/datus-db-adapters/{package_name}" in workflow


def test_nightly_runs_postgresql_storage_adapter_tests_from_checkout():
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'STORAGE_ADAPTERS_ROOT="$(default_repo_root' in script
    assert 'run_logged "PostgreSQL Storage Adapter Tests"' in script
    assert 'uv run --no-sync pytest "$STORAGE_ADAPTERS_ROOT/datus-storage-postgresql/tests"' in script
    assert '"PostgreSQL Storage Adapter Tests"' in script.split("DOCKER_GROUPS=(", maxsplit=1)[1]


def test_nightly_runs_p0_contracts_without_reruns_or_skips():
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")
    logical_lines = script.replace("\\\n", " ").splitlines()

    for group in (
        "P0 PostgreSQL Agent Storage Contracts",
        "P0 SQL Policy Plugin Contracts",
        "P0 Dosi Semantic Modeling E2E",
        "P0 Dashboard Bootstrap Skill E2E",
    ):
        command = next(line for line in logical_lines if line.startswith("run_") and f'"{group}"' in line)
        assert "--fail-on-skip" in command
        assert "--reruns" not in command
        assert "uv run python -m pytest" in command

    assert '"P0 PostgreSQL Agent Storage Contracts"' in script.split("DOCKER_GROUPS=(", maxsplit=1)[1]
    assert '"P0 Dashboard Bootstrap Skill E2E"' in script.split("COMPOSE_GROUPS=(", maxsplit=1)[1]


def test_nightly_tracebacks_do_not_publish_local_credentials():
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    pytest_addopts = [line for line in script.splitlines() if 'export PYTEST_ADDOPTS="' in line]
    assert len(pytest_addopts) == 2
    assert all("--no-showlocals" in line for line in pytest_addopts)


def test_release_candidate_reuses_p0_nightly_on_the_release_ref():
    nightly = WORKFLOW.read_text(encoding="utf-8")
    release = RELEASE_CANDIDATE_WORKFLOW.read_text(encoding="utf-8")
    prepare = (REPO_ROOT / ".github" / "workflows" / "prepare-release.yml").read_text(encoding="utf-8")

    assert "workflow_call:" in nightly
    assert "\npermissions:\n  contents: read\n" in nightly
    assert "DATUS_AGENT_REF: ${{ inputs.datus_agent_ref || github.ref_name }}" in nightly
    assert "NIGHTLY_GROUP_FILTER: ${{ inputs.nightly_group_filter || '' }}" in nightly
    assert "uses: ./.github/workflows/run-nightly.yml" in release
    assert "if: ${{ github.event_name != 'pull_request' }}" in release
    assert "datus_agent_ref: ${{ inputs.ref || github.ref }}" in release
    assert "nightly_group_filter: '^P0 '" in release
    assert "secrets: inherit" not in release
    assert "RELEASE_BOT_TOKEN: ${{ secrets.RELEASE_BOT_TOKEN }}" in release
    assert "uses: ./.github/workflows/release-candidate.yml" in prepare
    assert "RELEASE_BOT_TOKEN: ${{ secrets.RELEASE_BOT_TOKEN }}" in prepare


def test_every_adapter_suite_is_deselected_from_the_main_nightly_run():
    """An adapter suite with its own compose lifecycle must be excluded from the
    generic nightly command, or it runs a second time with no container up and
    fails on connection rather than skipping."""
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")
    deselect_block = script.split("NIGHTLY_DEDICATED_SUITE_DESELECTS=(", maxsplit=1)[1].split("\n)", maxsplit=1)[0]

    pattern = r"tests/integration/adapters/test_\w+\.py"
    deselected = set(re.findall(pattern, deselect_block))
    referenced = set(re.findall(pattern, script))

    assert referenced <= deselected, f"adapter suites missing a --deselect: {sorted(referenced - deselected)}"


def test_nightly_runs_doris_agent_contract_from_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "ADAPTERS_DORIS=1" >> $GITHUB_ENV' in workflow
    assert 'echo "DORIS_QUERY_HOST_PORT=29031" >> $GITHUB_ENV' in workflow
    assert 'echo "DORIS_HTTP_HOST_PORT=28031" >> $GITHUB_ENV' in workflow
    assert 'DORIS_COMPOSE="${DORIS_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-doris/docker-compose.yml}"' in script
    # The db-adapters compose file runs the all-in-one image as a single
    # `doris` service; waiting on the retired `doris-fe`/`doris-be` names
    # aborts the suite before pytest starts.
    assert 'run_compose_suite "Doris Adapter Tests" "$DORIS_COMPOSE" "doris:600" --' in script
    assert 'uv run --no-sync python "$DB_ADAPTERS_ROOT/datus-doris/scripts/wait_for_doris.py"' in script
    assert 'wait_for_doris_client_readiness "${DORIS_READY_TIMEOUT:-600}"' in script
    assert 'wait_for_tcp_readiness "Doris"' not in script
    assert "tests/integration/adapters/test_doris.py" in script


def test_nightly_runs_tidb_agent_contract_from_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "ADAPTERS_TIDB=1" >> $GITHUB_ENV' in workflow
    assert 'echo "TIDB_HOST_PORT=24000" >> $GITHUB_ENV' in workflow
    assert 'TIDB_COMPOSE="${TIDB_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-tidb/docker-compose.yml}"' in script
    # One container: the adapter's compose runs TiDB's built-in unistore engine.
    assert 'run_compose_suite "TiDB Adapter Tests" "$TIDB_COMPOSE" "tidb:120" --' in script
    # Assert the whole readiness call, not just the function name: a wrong port
    # or path would still satisfy a fragment search.
    assert (
        'wait_for_http_readiness "TiDB" "http://${TIDB_HOST:-127.0.0.1}:${TIDB_STATUS_HOST_PORT:-20080}/status" 300'
    ) in script
    assert "tests/integration/adapters/test_tidb.py" in script
    assert '"TiDB Adapter Tests"' in script.split("COMPOSE_GROUPS=(", maxsplit=1)[1]


def test_nightly_runs_gaussdb_agent_contract_from_checkout():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = NIGHTLY_SCRIPT.read_text(encoding="utf-8")

    assert 'echo "ADAPTERS_GAUSSDB=1" >> $GITHUB_ENV' in workflow
    assert 'GAUSSDB_COMPOSE="${GAUSSDB_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-gaussdb/docker-compose.yml}"' in script
    assert 'run_compose_suite "GaussDB Adapter Tests" "$GAUSSDB_COMPOSE" "gaussdb:600"' in script
    assert "tests/integration/adapters/test_gaussdb.py" in script
    assert '"GaussDB Adapter Tests"' in script.split("COMPOSE_GROUPS=(", maxsplit=1)[1]


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
