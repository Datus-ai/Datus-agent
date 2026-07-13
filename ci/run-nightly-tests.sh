#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT" || {
  echo "Failed to enter repository root: $REPO_ROOT" >&2
  exit 1
}

LOG_FILE="${NIGHTLY_LOG_FILE:-test_output_nightly_$(date +%Y%m%d_%H%M%S).log}"
NIGHTLY_MANIFEST_FILE="${NIGHTLY_MANIFEST_FILE:-nightly-manifest.json}"
NIGHTLY_MANIFEST_ENABLED="${NIGHTLY_MANIFEST_ENABLED:-1}"
NIGHTLY_MANIFEST_FINALIZED=0
NIGHTLY_FAILURE_CLASSIFICATION_FILE="${NIGHTLY_FAILURE_CLASSIFICATION_FILE:-nightly-failure-classification.json}"
NIGHTLY_FAILURE_CLASSIFICATION_ENABLED="${NIGHTLY_FAILURE_CLASSIFICATION_ENABLED:-1}"
NIGHTLY_FAILURE_CLASSIFICATION_FINALIZED=0
PROVIDER_COVERAGE_MANIFEST_FILE="${PROVIDER_COVERAGE_MANIFEST_FILE:-provider-coverage-manifest.json}"
PROVIDER_COVERAGE_MANIFEST_ENABLED="${PROVIDER_COVERAGE_MANIFEST_ENABLED:-1}"
PROVIDER_COVERAGE_MANIFEST_FINALIZED=0
NIGHTLY_TRACE_REFERENCES_FILE="${NIGHTLY_TRACE_REFERENCES_FILE:-nightly-trace-references.jsonl}"
NIGHTLY_TRACE_SUMMARY_FILE="${NIGHTLY_TRACE_SUMMARY_FILE:-nightly-trace-summary.jsonl}"
NIGHTLY_PROCESS_DIAGNOSTICS_FILE="${NIGHTLY_PROCESS_DIAGNOSTICS_FILE:-nightly-process-diagnostics.json}"
NIGHTLY_TRACE_DIAGNOSTICS_FINALIZED=0
test_exit_code=0
last_command_exit_code=0
NIGHTLY_GROUP_FILTER="${NIGHTLY_GROUP_FILTER:-}"
AGENT_TEST_CONFIG="${AGENT_TEST_CONFIG:-tests/conf/agent.yml}"
DATUS_TEST_PROJECT_NAME="${DATUS_TEST_PROJECT_NAME:-datus_agent_nightly}"
export DATUS_TEST_PROJECT_NAME
NIGHTLY_REQUIRE_LANGFUSE_TRACING="${NIGHTLY_REQUIRE_LANGFUSE_TRACING:-0}"
NIGHTLY_STARTED_AT="${NIGHTLY_STARTED_AT:-}"
NIGHTLY_COMPOSE_PROJECT_PREFIX="${NIGHTLY_COMPOSE_PROJECT_PREFIX:-datus-nightly-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
EXTERNAL_REPOS_ROOT="${EXTERNAL_REPOS_ROOT:-${REPO_ROOT}/external}"
NIGHTLY_HOME="${DATUS_TEST_HOME:-${REPO_ROOT}/.datus_test_data}"
NIGHTLY_PROJECT_ROOT="${NIGHTLY_PROJECT_ROOT:-${NIGHTLY_HOME}/workspace}"
UNIT_TEST_HOME="${NIGHTLY_UNIT_TEST_HOME:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-agent-nightly-unit-${GITHUB_RUN_ID:-$$}}"
UNIT_TEST_PROJECT_ROOT="${NIGHTLY_UNIT_TEST_PROJECT_ROOT:-${UNIT_TEST_HOME}/workspace}"
NIGHTLY_PYTEST_BASETEMP="${NIGHTLY_PYTEST_BASETEMP:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-agent-nightly-pytest-${GITHUB_RUN_ID:-$$}-${GITHUB_RUN_ATTEMPT:-0}}"
AGENT_TEST_CONFIG_BACKUP="${AGENT_TEST_CONFIG_BACKUP:-${TMPDIR:-/tmp}/datus-agent-nightly-config-${GITHUB_RUN_ID:-$$}.bak}"
export LOG_FILE NIGHTLY_MANIFEST_FILE NIGHTLY_FAILURE_CLASSIFICATION_FILE PROVIDER_COVERAGE_MANIFEST_FILE NIGHTLY_TRACE_REFERENCES_FILE NIGHTLY_TRACE_SUMMARY_FILE NIGHTLY_PROCESS_DIAGNOSTICS_FILE NIGHTLY_HOME NIGHTLY_PROJECT_ROOT UNIT_TEST_HOME NIGHTLY_PYTEST_BASETEMP NIGHTLY_STARTED_AT

NIGHTLY_PYTEST_ROOTS=(tests/integration tests/regression)

default_repo_root() {
  local explicit_root="$1"
  local repo_name="$2"

  if [ -n "$explicit_root" ]; then
    echo "$explicit_root"
  elif [ -d "${EXTERNAL_REPOS_ROOT}/${repo_name}" ]; then
    echo "${EXTERNAL_REPOS_ROOT}/${repo_name}"
  elif [ -d "${WORKSPACE_ROOT}/${repo_name}" ]; then
    echo "${WORKSPACE_ROOT}/${repo_name}"
  else
    echo "${EXTERNAL_REPOS_ROOT}/${repo_name}"
  fi
}

DB_ADAPTERS_ROOT="$(default_repo_root "${DB_ADAPTERS_ROOT:-}" datus-db-adapters)"
BI_ADAPTERS_ROOT="$(default_repo_root "${BI_ADAPTERS_ROOT:-}" datus-bi-adapters)"
SCHEDULER_ADAPTERS_ROOT="$(default_repo_root "${SCHEDULER_ADAPTERS_ROOT:-}" datus-scheduler-adapters)"
STORAGE_ADAPTERS_ROOT="$(default_repo_root "${STORAGE_ADAPTERS_ROOT:-}" datus-storage-adapters)"

POSTGRES_COMPOSE="${POSTGRES_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-postgresql/docker-compose.yml}"
MYSQL_COMPOSE="${MYSQL_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-mysql/docker-compose.yml}"
CLICKHOUSE_COMPOSE="${CLICKHOUSE_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-clickhouse/docker-compose.yml}"
STARROCKS_COMPOSE="${STARROCKS_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-starrocks/docker-compose.yml}"
TRINO_COMPOSE="${TRINO_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-trino/docker-compose.yml}"
GREENPLUM_COMPOSE="${GREENPLUM_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-greenplum/docker-compose.yml}"
HIVE_COMPOSE="${HIVE_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-hive/docker-compose.yml}"
SPARK_COMPOSE="${SPARK_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-spark/docker-compose.yml}"
SUPERSET_COMPOSE="${SUPERSET_COMPOSE:-${BI_ADAPTERS_ROOT}/datus-bi-superset/tests/integration/docker-compose.yml}"
GRAFANA_COMPOSE="${GRAFANA_COMPOSE:-${BI_ADAPTERS_ROOT}/datus-bi-grafana/tests/integration/docker-compose.yml}"
AIRFLOW_COMPOSE="${AIRFLOW_COMPOSE:-${SCHEDULER_ADAPTERS_ROOT}/datus-scheduler-airflow/tests/integration/docker-compose.yml}"

COMPOSE_FILES=(
  "$POSTGRES_COMPOSE"
  "$MYSQL_COMPOSE"
  "$CLICKHOUSE_COMPOSE"
  "$STARROCKS_COMPOSE"
  "$TRINO_COMPOSE"
  "$GREENPLUM_COMPOSE"
  "$HIVE_COMPOSE"
  "$SPARK_COMPOSE"
  "$SUPERSET_COMPOSE"
  "$GRAFANA_COMPOSE"
  "$AIRFLOW_COMPOSE"
)

COMPOSE_GROUPS=(
  "Superset Nightly Tests"
  "Grafana Nightly Tests"
  "Airflow Nightly Tests"
  "PostgreSQL Adapter Tests"
  "MySQL Adapter Tests"
  "ClickHouse Adapter Tests"
  "StarRocks Adapter Tests"
  "Trino Adapter Tests"
  "Greenplum Adapter Tests"
  "Hive Adapter Tests"
  "Spark Adapter Tests"
)

DOCKER_GROUPS=(
  "PostgreSQL Storage Adapter Tests"
  "${COMPOSE_GROUPS[@]}"
)

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

utc_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

manifest_command_json() {
  python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}

manifest_update() {
  if [ "$NIGHTLY_MANIFEST_ENABLED" != "1" ]; then
    return 0
  fi

  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    "${REPO_ROOT}/.venv/bin/python" ci/nightly_manifest.py "$@" 2>>"$LOG_FILE"
  elif command -v uv >/dev/null 2>&1; then
    uv run python ci/nightly_manifest.py "$@" 2>>"$LOG_FILE"
  else
    python3 ci/nightly_manifest.py "$@" 2>>"$LOG_FILE"
  fi
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: nightly manifest update failed: ci/nightly_manifest.py $*" | tee -a "$LOG_FILE" >&2
    return 0
  fi
}

manifest_init() {
  manifest_update init \
    --output "$NIGHTLY_MANIFEST_FILE" \
    --repo-root "$REPO_ROOT" \
    --external-repos-root "$EXTERNAL_REPOS_ROOT"
}

manifest_record_suite() {
  local group_name="$1"
  local mode="$2"
  local kind="$3"
  local status="$4"
  local exit_code="$5"
  local started_at="$6"
  local ended_at="$7"
  shift 7

  local command_json
  command_json="$(manifest_command_json "$@")"

  manifest_update record-suite \
    --output "$NIGHTLY_MANIFEST_FILE" \
    --name "$group_name" \
    --mode "$mode" \
    --kind "$kind" \
    --status "$status" \
    --exit-code "$exit_code" \
    --started-at "$started_at" \
    --ended-at "$ended_at" \
    --command-json "$command_json" \
    --compose-file "${NIGHTLY_CURRENT_COMPOSE_FILE:-}" \
    --compose-project "${NIGHTLY_CURRENT_COMPOSE_PROJECT:-}" \
    --host-ports "${NIGHTLY_CURRENT_HOST_PORTS:-}"
}

manifest_record_collection() {
  local group_name="$1"
  local exit_code="$2"
  local output_file="$3"

  manifest_update record-collection \
    --output "$NIGHTLY_MANIFEST_FILE" \
    --name "$group_name" \
    --exit-code "$exit_code" \
    --collection-output "$output_file"
}

manifest_record_compose_project() {
  local group_name="$1"
  local project_name="$2"
  local compose_file="$3"
  local host_ports="$4"

  manifest_update record-compose-project \
    --output "$NIGHTLY_MANIFEST_FILE" \
    --group "$group_name" \
    --project "$project_name" \
    --compose-file "$compose_file" \
    --host-ports "$host_ports"
}

manifest_finalize() {
  if [ "$NIGHTLY_MANIFEST_FINALIZED" = "1" ]; then
    return 0
  fi
  NIGHTLY_MANIFEST_FINALIZED=1
  manifest_update finalize --output "$NIGHTLY_MANIFEST_FILE" --exit-code "$test_exit_code"
}

failure_classification_update() {
  if [ "$NIGHTLY_FAILURE_CLASSIFICATION_ENABLED" != "1" ]; then
    return 0
  fi

  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    "${REPO_ROOT}/.venv/bin/python" ci/classify_nightly_failures.py "$@" 2>>"$LOG_FILE"
  elif command -v uv >/dev/null 2>&1; then
    uv run python ci/classify_nightly_failures.py "$@" 2>>"$LOG_FILE"
  else
    python3 ci/classify_nightly_failures.py "$@" 2>>"$LOG_FILE"
  fi
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: nightly failure classification update failed: ci/classify_nightly_failures.py $*" | tee -a "$LOG_FILE" >&2
    return "$status"
  fi
  return 0
}

failure_classification_finalize() {
  if [ "$NIGHTLY_FAILURE_CLASSIFICATION_FINALIZED" = "1" ]; then
    return 0
  fi
  if failure_classification_update \
    --manifest "$NIGHTLY_MANIFEST_FILE" \
    --log-file "$LOG_FILE" \
    --registry ci/flaky-registry.yml \
    --output "$NIGHTLY_FAILURE_CLASSIFICATION_FILE" \
    --exit-code "$test_exit_code" &&
    [ -s "$NIGHTLY_FAILURE_CLASSIFICATION_FILE" ]; then
    NIGHTLY_FAILURE_CLASSIFICATION_FINALIZED=1
  else
    echo "WARNING: nightly failure classification artifact was not written; will retry if cleanup runs again." | tee -a "$LOG_FILE" >&2
  fi
  return 0
}

provider_coverage_update() {
  if [ "$PROVIDER_COVERAGE_MANIFEST_ENABLED" != "1" ]; then
    return 0
  fi

  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    "${REPO_ROOT}/.venv/bin/python" ci/provider_coverage_manifest.py "$@" 2>>"$LOG_FILE"
  elif command -v uv >/dev/null 2>&1; then
    uv run python ci/provider_coverage_manifest.py "$@" 2>>"$LOG_FILE"
  else
    python3 ci/provider_coverage_manifest.py "$@" 2>>"$LOG_FILE"
  fi
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: provider coverage manifest update failed: ci/provider_coverage_manifest.py $*" | tee -a "$LOG_FILE" >&2
    return "$status"
  fi
  return 0
}

provider_coverage_finalize() {
  if [ "$PROVIDER_COVERAGE_MANIFEST_FINALIZED" = "1" ]; then
    return 0
  fi
  if provider_coverage_update \
    --repo-root "$REPO_ROOT" \
    --provider-catalog conf/providers.yml \
    --coverage-config ci/provider-coverage.yml \
    --nightly-manifest "$NIGHTLY_MANIFEST_FILE" \
    --output "$PROVIDER_COVERAGE_MANIFEST_FILE" &&
    [ -s "$PROVIDER_COVERAGE_MANIFEST_FILE" ]; then
    PROVIDER_COVERAGE_MANIFEST_FINALIZED=1
  else
    echo "WARNING: provider coverage manifest artifact was not written; will retry if cleanup runs again." | tee -a "$LOG_FILE" >&2
  fi
  return 0
}

command_contains_pytest() {
  local arg
  for arg in "$@"; do
    if [ "$arg" = "pytest" ]; then
      return 0
    fi
  done
  return 1
}

collect_pytest_suite() {
  local group_name="$1"
  shift

  if ! command_contains_pytest "$@"; then
    return 0
  fi

  local output_file="${NIGHTLY_PYTEST_BASETEMP}/manifest-collect-${group_name//[^A-Za-z0-9_-]/_}.log"
  mkdir -p "$NIGHTLY_PYTEST_BASETEMP"
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" "$@" -p ci.pytest_manifest_plugin --collect-only >"$output_file" 2>&1
  local collect_status=$?
  manifest_record_collection "$group_name" "$collect_status" "$output_file"
  if [ "$collect_status" -ne 0 ]; then
    log "WARNING: collection failed for ${group_name}; continuing and preserving failure in manifest."
  fi
  return 0
}

should_run_group() {
  local group_name="$1"

  if [ -z "$NIGHTLY_GROUP_FILTER" ]; then
    return 0
  fi

  [[ "$group_name" =~ $NIGHTLY_GROUP_FILTER ]]
}

validate_nightly_group_filter() {
  if [ -z "$NIGHTLY_GROUP_FILTER" ]; then
    return 0
  fi

  [[ "__datus_filter_probe__" =~ $NIGHTLY_GROUP_FILTER ]]
  local status=$?
  if [ "$status" -eq 2 ]; then
    echo "Invalid NIGHTLY_GROUP_FILTER regex: $NIGHTLY_GROUP_FILTER" | tee -a "$LOG_FILE" >&2
    return 1
  fi
  return 0
}

is_under_dir() {
  local path="${1%/}"
  local parent="${2%/}"

  if [ -z "$parent" ] || [ "$parent" = "/" ]; then
    return 1
  fi

  [[ "$path" == "$parent"/* ]]
}

validate_unit_test_home() {
  local path="${1%/}"
  local user_home="${HOME:-}"
  local tmp_root="${TMPDIR:-/tmp}"
  tmp_root="${tmp_root%/}"
  local runner_temp="${RUNNER_TEMP:-}"
  runner_temp="${runner_temp%/}"

  case "$path" in
    "" | "." | "/" | "$user_home" | "$REPO_ROOT" | "$WORKSPACE_ROOT" | *"/.."* | *"/../"* | "../"* | *"/."* | *"/./"*)
      echo "Refusing to remove unsafe UNIT_TEST_HOME: $UNIT_TEST_HOME" | tee -a "$LOG_FILE" >&2
      return 1
      ;;
  esac

  case "$path" in
    /*) ;;
    *)
      echo "Refusing to remove non-absolute UNIT_TEST_HOME: $UNIT_TEST_HOME" | tee -a "$LOG_FILE" >&2
      return 1
      ;;
  esac

  if is_under_dir "$path" "$runner_temp" || is_under_dir "$path" "$tmp_root" || is_under_dir "$path" "/tmp"; then
    return 0
  fi

  echo "Refusing to remove UNIT_TEST_HOME outside temp directories: $UNIT_TEST_HOME" | tee -a "$LOG_FILE" >&2
  return 1
}

validate_pytest_basetemp() {
  local path="${1%/}"
  local user_home="${HOME:-}"
  local tmp_root="${TMPDIR:-/tmp}"
  tmp_root="${tmp_root%/}"
  local runner_temp="${RUNNER_TEMP:-}"
  runner_temp="${runner_temp%/}"

  case "$path" in
    "" | "." | "/" | "$user_home" | "$REPO_ROOT" | "$WORKSPACE_ROOT" | *"/.."* | *"/../"* | "../"* | *"/."* | *"/./"*)
      echo "Refusing to remove unsafe NIGHTLY_PYTEST_BASETEMP: $NIGHTLY_PYTEST_BASETEMP" | tee -a "$LOG_FILE" >&2
      return 1
      ;;
  esac

  case "$path" in
    /*) ;;
    *)
      echo "Refusing to remove non-absolute NIGHTLY_PYTEST_BASETEMP: $NIGHTLY_PYTEST_BASETEMP" | tee -a "$LOG_FILE" >&2
      return 1
      ;;
  esac

  if is_under_dir "$path" "$runner_temp" || is_under_dir "$path" "$tmp_root" || is_under_dir "$path" "/tmp"; then
    return 0
  fi

  echo "Refusing to remove NIGHTLY_PYTEST_BASETEMP outside temp directories: $NIGHTLY_PYTEST_BASETEMP" | tee -a "$LOG_FILE" >&2
  return 1
}

has_docker_compose() {
  docker compose version >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1
}

will_run_any_compose_suite() {
  local group_name
  for group_name in "${COMPOSE_GROUPS[@]}"; do
    if should_run_group "$group_name"; then
      return 0
    fi
  done
  return 1
}

will_run_any_docker_suite() {
  local group_name
  for group_name in "${DOCKER_GROUPS[@]}"; do
    if should_run_group "$group_name"; then
      return 0
    fi
  done
  return 1
}

require_docker_runtime() {
  if ! will_run_any_docker_suite; then
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for Docker-backed nightly suites" | tee -a "$LOG_FILE" >&2
    test_exit_code=127
    return 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not available for Docker-backed nightly suites" | tee -a "$LOG_FILE" >&2
    test_exit_code=127
    return 1
  fi

  if will_run_any_compose_suite && ! has_docker_compose; then
    echo "Docker Compose is required for nightly compose-backed suites" | tee -a "$LOG_FILE" >&2
    test_exit_code=127
    return 1
  fi
}

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "Docker Compose is not available" >&2
    return 127
  fi
}

compose_project_slug() {
  local group_name="$1"

  case "$group_name" in
    "Superset Nightly Tests") echo "superset" ;;
    "Grafana Nightly Tests") echo "grafana" ;;
    "Airflow Nightly Tests") echo "airflow" ;;
    "PostgreSQL Adapter Tests") echo "postgresql" ;;
    "MySQL Adapter Tests") echo "mysql" ;;
    "ClickHouse Adapter Tests") echo "clickhouse" ;;
    "StarRocks Adapter Tests") echo "starrocks" ;;
    "Trino Adapter Tests") echo "trino" ;;
    "Greenplum Adapter Tests") echo "greenplum" ;;
    "Hive Adapter Tests") echo "hive" ;;
    "Spark Adapter Tests") echo "spark" ;;
    *)
      echo "$group_name" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//; s/-+$//'
      ;;
  esac
}

compose_project_name() {
  local group_name="$1"
  local slug
  slug="$(compose_project_slug "$group_name")"
  local project_name="${NIGHTLY_COMPOSE_PROJECT_PREFIX}${slug}"

  echo "$project_name" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9_-]+/-/g; s/^-+//'
}

compose_cmd() {
  local project_name="$1"
  local compose_file="$2"
  shift 2

  docker_compose -p "$project_name" -f "$compose_file" "$@"
}

compose_down() {
  local compose_file="$1"
  local project_name="$2"
  compose_cmd "$project_name" "$compose_file" down -v --remove-orphans >/dev/null 2>&1 || true
}

cleanup_all_compose() {
  set +e
  if ! has_docker_compose; then
    echo "Docker Compose is not available; skipping compose cleanup"
    return 0
  fi
  local group_name
  local compose_file
  local project_name
  for group_name in "${COMPOSE_GROUPS[@]}"; do
    case "$group_name" in
      "Superset Nightly Tests") compose_file="$SUPERSET_COMPOSE" ;;
      "Grafana Nightly Tests") compose_file="$GRAFANA_COMPOSE" ;;
      "Airflow Nightly Tests") compose_file="$AIRFLOW_COMPOSE" ;;
      "PostgreSQL Adapter Tests") compose_file="$POSTGRES_COMPOSE" ;;
      "MySQL Adapter Tests") compose_file="$MYSQL_COMPOSE" ;;
      "ClickHouse Adapter Tests") compose_file="$CLICKHOUSE_COMPOSE" ;;
      "StarRocks Adapter Tests") compose_file="$STARROCKS_COMPOSE" ;;
      "Trino Adapter Tests") compose_file="$TRINO_COMPOSE" ;;
      "Greenplum Adapter Tests") compose_file="$GREENPLUM_COMPOSE" ;;
      "Hive Adapter Tests") compose_file="$HIVE_COMPOSE" ;;
      "Spark Adapter Tests") compose_file="$SPARK_COMPOSE" ;;
      *) continue ;;
    esac
    if [ -f "$compose_file" ]; then
      project_name="$(compose_project_name "$group_name")"
      echo "Stopping services from $compose_file (project=$project_name)"
      compose_cmd "$project_name" "$compose_file" down -v --remove-orphans || true
    fi
  done
}

backup_agent_test_config() {
  if [ ! -f "$AGENT_TEST_CONFIG_BACKUP" ]; then
    cp "$AGENT_TEST_CONFIG" "$AGENT_TEST_CONFIG_BACKUP"
  fi
}

restore_agent_test_config() {
  if [ -n "$AGENT_TEST_CONFIG_BACKUP" ] && [ -f "$AGENT_TEST_CONFIG_BACKUP" ]; then
    cp "$AGENT_TEST_CONFIG_BACKUP" "$AGENT_TEST_CONFIG"
  fi
}

set_agent_test_config_paths() {
  local home="$1"
  local project_root="$2"
  local project_name="${3:-$DATUS_TEST_PROJECT_NAME}"

  python3 - "$AGENT_TEST_CONFIG" "$home" "$project_root" "$project_name" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
home = sys.argv[2]
project_root = sys.argv[3]
project_name = sys.argv[4]

updated = []
updated_home = False
updated_project_root = False
updated_project_name = False
for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if line == "agent:":
        updated.append(line)
        updated.append(f"  project_name: {project_name}")
        updated_project_name = True
    elif line.startswith("  project_name: ") and stripped.startswith("project_name:"):
        if not updated_project_name:
            updated.append(f"  project_name: {project_name}")
            updated_project_name = True
    elif line.startswith("  home: ") and stripped.startswith("home:"):
        updated.append(f"  home: {home}")
        updated_home = True
    elif line.startswith("  project_root: ") and stripped.startswith("project_root:"):
        updated.append(f"  project_root: {project_root}")
        updated_project_root = True
    else:
        updated.append(line)

if not updated_home or not updated_project_root or not updated_project_name:
    missing = []
    if not updated_home:
        missing.append("home")
    if not updated_project_root:
        missing.append("project_root")
    if not updated_project_name:
        missing.append("project_name")
    print(f"Failed to rewrite required keys in {path}: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
}

run_with_agent_home() {
  local home="$1"
  local project_root="$2"
  shift 2

  if ! backup_agent_test_config; then
    echo "Failed to back up $AGENT_TEST_CONFIG to $AGENT_TEST_CONFIG_BACKUP" >&2
    return 1
  fi
  mkdir -p "$home" "$project_root" || return 1
  if ! set_agent_test_config_paths "$home" "$project_root" "$DATUS_TEST_PROJECT_NAME"; then
    restore_agent_test_config
    return 1
  fi
  DATUS_TEST_HOME="$home" DATUS_TEST_PROJECT_NAME="$DATUS_TEST_PROJECT_NAME" DATUS_TUI=0 "$@"
  local status=$?
  restore_agent_test_config
  return "$status"
}

nightly_requires_langfuse_tracing() {
  case "$(printf '%s' "$NIGHTLY_REQUIRE_LANGFUSE_TRACING" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

prepare_nightly_langfuse_tracing() {
  if ! nightly_requires_langfuse_tracing; then
    return 0
  fi

  export LANGFUSE_BASE_URL="${LANGFUSE_BASE_URL:-https://us.cloud.langfuse.com}"

  local missing=()
  if [ -z "${LANGFUSE_PUBLIC_KEY:-}" ]; then
    missing+=(LANGFUSE_PUBLIC_KEY)
  fi
  if [ -z "${LANGFUSE_SECRET_KEY:-}" ]; then
    missing+=(LANGFUSE_SECRET_KEY)
  fi

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "Langfuse tracing requested for nightly suites, but missing: ${missing[*]}" | tee -a "$LOG_FILE" >&2
    return 1
  fi

  log "Langfuse tracing enabled for nightly suites: base_url=$LANGFUSE_BASE_URL"
}

nightly_trace_expected_for_group() {
  case "$1" in
    "Gen Agent Tests" | \
      "Reference Template Nightly Tests" | \
      "Web UI Nightly Tests" | \
      "Product E2E Nightly Tests" | \
      "Superset Nightly Tests" | \
      "Grafana Nightly Tests" | \
      "Airflow Nightly Tests" | \
      "Provider Health Tests")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

collect_nightly_trace_diagnostics() {
  if [ "$NIGHTLY_TRACE_DIAGNOSTICS_FINALIZED" = "1" ]; then
    return 0
  fi
  NIGHTLY_TRACE_DIAGNOSTICS_FINALIZED=1

  local ended_at
  ended_at="$(utc_now)"

  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    "${REPO_ROOT}/.venv/bin/python" ci/collect_nightly_trace_summary.py \
      --trace-references-jsonl "$NIGHTLY_TRACE_REFERENCES_FILE" \
      --output-jsonl "$NIGHTLY_TRACE_SUMMARY_FILE" \
      --diagnostics-json "$NIGHTLY_PROCESS_DIAGNOSTICS_FILE" \
      --from-start-time "${NIGHTLY_STARTED_AT:-}" \
      --to-start-time "$ended_at" 2>>"$LOG_FILE"
  elif command -v uv >/dev/null 2>&1; then
    uv run python ci/collect_nightly_trace_summary.py \
      --trace-references-jsonl "$NIGHTLY_TRACE_REFERENCES_FILE" \
      --output-jsonl "$NIGHTLY_TRACE_SUMMARY_FILE" \
      --diagnostics-json "$NIGHTLY_PROCESS_DIAGNOSTICS_FILE" \
      --from-start-time "${NIGHTLY_STARTED_AT:-}" \
      --to-start-time "$ended_at" 2>>"$LOG_FILE"
  else
    python3 ci/collect_nightly_trace_summary.py \
      --trace-references-jsonl "$NIGHTLY_TRACE_REFERENCES_FILE" \
      --output-jsonl "$NIGHTLY_TRACE_SUMMARY_FILE" \
      --diagnostics-json "$NIGHTLY_PROCESS_DIAGNOSTICS_FILE" \
      --from-start-time "${NIGHTLY_STARTED_AT:-}" \
      --to-start-time "$ended_at" 2>>"$LOG_FILE"
  fi

  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "WARNING: nightly trace diagnostics collection failed; continuing because trace diagnostics are non-blocking." | tee -a "$LOG_FILE" >&2
    return 0
  fi
}

nightly_kb_ready_dir() {
  echo "${NIGHTLY_HOME}/data/${DATUS_TEST_PROJECT_NAME}/datus_db"
}

nightly_kb_dataset_ready() {
  local dataset_dir="$1"

  [ -d "${dataset_dir}/data" ] || return 1
  find "${dataset_dir}/data" -type f -size +0 -print -quit | grep -q .
}

nightly_kb_data_ready() {
  local ready_dir
  ready_dir="$(nightly_kb_ready_dir)"

  local dataset
  for dataset in schema_metadata.lance schema_value.lance metrics.lance reference_sql.lance reference_template.lance; do
    nightly_kb_dataset_ready "${ready_dir}/${dataset}" || return 1
  done
}

will_run_kb_dependent_suite() {
  local group_name
  for group_name in \
    "Gen Agent Tests" \
    "Reference Template Nightly Tests" \
    "Web UI Nightly Tests" \
    "Main Nightly Tests" \
    "Product E2E Nightly Tests"; do
    if should_run_group "$group_name"; then
      return 0
    fi
  done
  return 1
}

ensure_nightly_kb_data() {
  if ! will_run_kb_dependent_suite; then
    return 0
  fi

  local ready_dir
  ready_dir="$(nightly_kb_ready_dir)"
  if nightly_kb_data_ready; then
    log "Knowledge base test data ready: ${ready_dir}"
    return 0
  fi

  log "Knowledge base test data missing or incomplete at ${ready_dir}; rebuilding before nightly tests"
  run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" bash ./build_scripts/build_test_data.sh 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    test_exit_code="$status"
    return "$status"
  fi

  if ! nightly_kb_data_ready; then
    echo "Knowledge base test data build completed but required datasets are still missing under ${ready_dir}" | tee -a "$LOG_FILE" >&2
    test_exit_code=1
    return 1
  fi
  return 0
}

cleanup_all() {
  set +e
  if [ "${NIGHTLY_SKIP_MANIFEST_FINALIZE:-0}" != "1" ]; then
    manifest_finalize
    provider_coverage_finalize
    failure_classification_finalize
    collect_nightly_trace_diagnostics
  fi
  restore_agent_test_config
  rm -f "$AGENT_TEST_CONFIG_BACKUP"
  if validate_pytest_basetemp "$NIGHTLY_PYTEST_BASETEMP"; then
    rm -rf "$NIGHTLY_PYTEST_BASETEMP"
  fi
  cleanup_all_compose
}

if [ "${1:-}" = "--cleanup-only" ]; then
  NIGHTLY_SKIP_MANIFEST_FINALIZE=1
  cleanup_all
  exit 0
fi

handle_interrupt() {
  cleanup_all
  exit 130
}

trap cleanup_all EXIT
trap handle_interrupt INT TERM
rm -f "$AGENT_TEST_CONFIG_BACKUP"

export DATUS_STRICT_NIGHTLY_REQUIREMENTS="${DATUS_STRICT_NIGHTLY_REQUIREMENTS:-1}"
export ADAPTERS_PG="${ADAPTERS_PG:-1}"
export ADAPTERS_MYSQL="${ADAPTERS_MYSQL:-1}"
export ADAPTERS_CH="${ADAPTERS_CH:-1}"
export ADAPTERS_SR="${ADAPTERS_SR:-1}"
export ADAPTERS_TRINO="${ADAPTERS_TRINO:-1}"
export ADAPTERS_GP="${ADAPTERS_GP:-1}"
export ADAPTERS_HIVE="${ADAPTERS_HIVE:-1}"
export ADAPTERS_SPARK="${ADAPTERS_SPARK:-1}"
export ADAPTERS_METRICFLOW_DUCKDB="${ADAPTERS_METRICFLOW_DUCKDB:-1}"
export ADAPTERS_METRICFLOW_MYSQL="${ADAPTERS_METRICFLOW_MYSQL:-1}"
export ADAPTERS_METRICFLOW_PG="${ADAPTERS_METRICFLOW_PG:-1}"
export SUPERSET_PORT="${SUPERSET_PORT:-18088}"
export SUPERSET_POSTGRES_HOST="${SUPERSET_POSTGRES_HOST:-127.0.0.1}"
export SUPERSET_POSTGRES_PORT="${SUPERSET_POSTGRES_PORT:-15433}"
export SUPERSET_URL="${SUPERSET_URL:-http://127.0.0.1:${SUPERSET_PORT}}"
export SUPERSET_USER="${SUPERSET_USER:-admin}"
export SUPERSET_PASS="${SUPERSET_PASS:-admin}"
export GRAFANA_PORT="${GRAFANA_PORT:-13000}"
export GRAFANA_POSTGRES_HOST="${GRAFANA_POSTGRES_HOST:-127.0.0.1}"
export GRAFANA_POSTGRES_PORT="${GRAFANA_POSTGRES_PORT:-15435}"
export GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:${GRAFANA_PORT}}"
export GRAFANA_USER="${GRAFANA_USER:-admin}"
export GRAFANA_PASS="${GRAFANA_PASS:-admin123}"
export AIRFLOW_HOST_PORT="${AIRFLOW_HOST_PORT:-18080}"
export AIRFLOW_URL="${AIRFLOW_URL:-http://127.0.0.1:${AIRFLOW_HOST_PORT}/api/v1}"
export AIRFLOW_USER="${AIRFLOW_USER:-admin}"
export AIRFLOW_USERNAME="${AIRFLOW_USERNAME:-$AIRFLOW_USER}"
export AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-admin}"

if [ "${NIGHTLY_FORCE_ADAPTER_ENV:-1}" = "1" ]; then
  export POSTGRESQL_HOST=localhost
  export POSTGRESQL_HOST_PORT="${POSTGRESQL_HOST_PORT:-25432}"
  export POSTGRESQL_PORT="$POSTGRESQL_HOST_PORT"
  export POSTGRESQL_USER=test_user
  export POSTGRESQL_PASSWORD=test_password
  export POSTGRESQL_DATABASE=test
  export POSTGRESQL_SCHEMA=public

  export MYSQL_HOST=localhost
  export MYSQL_HOST_PORT="${MYSQL_HOST_PORT:-23306}"
  export MYSQL_PORT="$MYSQL_HOST_PORT"
  export MYSQL_USER=test_user
  export MYSQL_PASSWORD=test_password
  export MYSQL_DATABASE=test

  export CLICKHOUSE_HTTP_HOST_PORT="${CLICKHOUSE_HTTP_HOST_PORT:-28123}"
  export CLICKHOUSE_NATIVE_HOST_PORT="${CLICKHOUSE_NATIVE_HOST_PORT:-29000}"
  export CLICKHOUSE_HOST=127.0.0.1
  export CLICKHOUSE_PORT="$CLICKHOUSE_HTTP_HOST_PORT"
  export CLICKHOUSE_USER=default_user
  export CLICKHOUSE_PASSWORD=default_test
  export CLICKHOUSE_DATABASE=default_test

  export STARROCKS_QUERY_HOST_PORT="${STARROCKS_QUERY_HOST_PORT:-29030}"
  export STARROCKS_HTTP_HOST_PORT="${STARROCKS_HTTP_HOST_PORT:-28030}"
  export STARROCKS_HOST=127.0.0.1
  export STARROCKS_PORT="$STARROCKS_QUERY_HOST_PORT"
  export STARROCKS_USER=root
  export STARROCKS_PASSWORD=
  export STARROCKS_CATALOG=default_catalog
  export STARROCKS_DATABASE=test

  export TRINO_HOST_PORT="${TRINO_HOST_PORT:-28080}"
  export TRINO_HOST=127.0.0.1
  export TRINO_PORT="$TRINO_HOST_PORT"
  export TRINO_USER=trino
  export TRINO_PASSWORD=
  export TRINO_HTTP_SCHEME=http

  export GREENPLUM_HOST_PORT="${GREENPLUM_HOST_PORT:-15434}"
  export GREENPLUM_HOST=localhost
  export GREENPLUM_PORT="$GREENPLUM_HOST_PORT"
  export GREENPLUM_USER=gpadmin
  export GREENPLUM_PASSWORD=pivotal
  export GREENPLUM_DATABASE=postgres
  export GREENPLUM_SCHEMA=public

  export HIVE_METASTORE_HOST_PORT="${HIVE_METASTORE_HOST_PORT:-29083}"
  export HIVE_THRIFT_HOST_PORT="${HIVE_THRIFT_HOST_PORT:-21000}"
  export HIVE_WEBUI_HOST_PORT="${HIVE_WEBUI_HOST_PORT:-21002}"
  export HIVE_HOST=localhost
  export HIVE_PORT="$HIVE_THRIFT_HOST_PORT"
  export HIVE_USERNAME=hive
  export HIVE_PASSWORD=
  export HIVE_DATABASE=default

  export SPARK_THRIFT_HOST_PORT="${SPARK_THRIFT_HOST_PORT:-31000}"
  export SPARK_UI_HOST_PORT="${SPARK_UI_HOST_PORT:-24040}"
  export SPARK_HOST=localhost
  export SPARK_PORT="$SPARK_THRIFT_HOST_PORT"
  export SPARK_USER=spark
  export SPARK_PASSWORD=
  export SPARK_DATABASE=default
  export SPARK_AUTH_MECHANISM=NONE
fi

run_logged_unfiltered() {
  local group_name="$1"
  shift

  log ""
  log "=== ${group_name} ==="
  local started_at
  started_at="$(utc_now)"
  collect_pytest_suite "$group_name" "$@"
  local trace_expected=0
  if nightly_trace_expected_for_group "$group_name"; then
    trace_expected=1
  fi
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    DATUS_NIGHTLY_SUITE_NAME="$group_name" \
    DATUS_NIGHTLY_TRACE_EXPECTED="$trace_expected" \
    DATUS_NIGHTLY_TRACE_REFERENCES_FILE="$NIGHTLY_TRACE_REFERENCES_FILE" \
    "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  local ended_at
  ended_at="$(utc_now)"
  last_command_exit_code="$cmd_status"
  if [ "$cmd_status" -ne 0 ]; then
    test_exit_code="$cmd_status"
  fi
  local status="passed"
  if [ "$cmd_status" -ne 0 ]; then
    status="failed"
  fi
  manifest_record_suite "$group_name" "blocking" "command" "$status" "$cmd_status" "$started_at" "$ended_at" "$@"
  return 0
}

run_logged() {
  local group_name="$1"
  shift
  if ! should_run_group "$group_name"; then
    log ""
    log "=== Skipping ${group_name} (NIGHTLY_GROUP_FILTER=${NIGHTLY_GROUP_FILTER}) ==="
    last_command_exit_code=0
    local now
    now="$(utc_now)"
    manifest_record_suite "$group_name" "blocking" "command" "skipped" 0 "$now" "$now" "$@"
    return 0
  fi

  run_logged_unfiltered "$group_name" "$@"
}

run_logged_warn_only_unfiltered() {
  local group_name="$1"
  shift

  log ""
  log "=== ${group_name} (warn-only) ==="
  local started_at
  started_at="$(utc_now)"
  collect_pytest_suite "$group_name" "$@"
  local trace_expected=0
  if nightly_trace_expected_for_group "$group_name"; then
    trace_expected=1
  fi
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
    DATUS_NIGHTLY_SUITE_NAME="$group_name" \
    DATUS_NIGHTLY_TRACE_EXPECTED="$trace_expected" \
    DATUS_NIGHTLY_TRACE_REFERENCES_FILE="$NIGHTLY_TRACE_REFERENCES_FILE" \
    "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  local ended_at
  ended_at="$(utc_now)"
  last_command_exit_code="$cmd_status"
  if [ "$cmd_status" -ne 0 ]; then
    log "WARNING: ${group_name} failed with exit code ${cmd_status}; continuing because this group is non-blocking."
  fi
  local status="passed"
  if [ "$cmd_status" -ne 0 ]; then
    status="failed"
  fi
  manifest_record_suite "$group_name" "warn-only" "command" "$status" "$cmd_status" "$started_at" "$ended_at" "$@"
  return 0
}

run_logged_warn_only() {
  local group_name="$1"
  shift
  if ! should_run_group "$group_name"; then
    log ""
    log "=== Skipping ${group_name} (NIGHTLY_GROUP_FILTER=${NIGHTLY_GROUP_FILTER}) ==="
    last_command_exit_code=0
    local now
    now="$(utc_now)"
    manifest_record_suite "$group_name" "warn-only" "command" "skipped" 0 "$now" "$now" "$@"
    return 0
  fi

  run_logged_warn_only_unfiltered "$group_name" "$@"
}

compose_up() {
  local project_name="$1"
  local compose_file="$2"
  shift 2
  if [ ! -f "$compose_file" ]; then
    echo "Missing compose file: $compose_file" | tee -a "$LOG_FILE" >&2
    test_exit_code=1
    return 1
  fi
  compose_cmd "$project_name" "$compose_file" up -d --build "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  if [ "$cmd_status" -ne 0 ]; then
    test_exit_code="$cmd_status"
    return 1
  fi
  return 0
}

wait_for_service_health() {
  local project_name="$1"
  local compose_file="$2"
  local service_name="$3"
  local timeout_seconds="$4"
  local container_id=""
  local has_health=""
  local status=""
  local deadline=$((SECONDS + timeout_seconds))

  container_id="$(compose_cmd "$project_name" "$compose_file" ps -q "$service_name")"
  if [ -z "$container_id" ]; then
    echo "No container found for service '$service_name' in $compose_file (project=$project_name)" | tee -a "$LOG_FILE" >&2
    compose_cmd "$project_name" "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
    test_exit_code=1
    return 1
  fi
  has_health="$(docker inspect --format '{{if .State.Health}}1{{else}}0{{end}}' "$container_id" 2>/dev/null || echo 0)"

  while [ "$SECONDS" -lt "$deadline" ]; do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || echo unknown)"
    if [ "$status" = "healthy" ]; then
      log "Service '$service_name' is $status"
      return 0
    fi
    if [ "$has_health" != "1" ] && [ "$status" = "running" ]; then
      log "Service '$service_name' is running and has no container healthcheck"
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for service '$service_name' from $compose_file (project=$project_name)" | tee -a "$LOG_FILE" >&2
  compose_cmd "$project_name" "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
  compose_cmd "$project_name" "$compose_file" logs --tail=200 2>&1 | tee -a "$LOG_FILE" || true
  test_exit_code=1
  return 1
}

dump_compose_diagnostics() {
  local project_name="$1"
  local compose_file="$2"
  local group_name="$3"

  log ""
  log "=== ${group_name} Service Diagnostics (project=${project_name}) ==="
  compose_cmd "$project_name" "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
  compose_cmd "$project_name" "$compose_file" logs --tail=200 2>&1 | tee -a "$LOG_FILE" || true
}

can_bind_host_port() {
  local port="$1"

  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
finally:
    sock.close()
PY
}

log_host_port_owner() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>&1 | tee -a "$LOG_FILE" || true
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp "sport = :$port" 2>&1 | tee -a "$LOG_FILE" || true
  fi
  if ! command -v lsof >/dev/null 2>&1 && ! command -v ss >/dev/null 2>&1; then
    netstat -an 2>/dev/null | grep "[.:]${port}[[:space:]].*LISTEN" | tee -a "$LOG_FILE" || true
  fi
}

compose_host_port_specs() {
  local group_name="$1"

  case "$group_name" in
    "Superset Nightly Tests")
      printf 'Superset web:%s\nSuperset PostgreSQL:%s\n' "${SUPERSET_PORT:-18088}" "${SUPERSET_POSTGRES_PORT:-15433}"
      ;;
    "Grafana Nightly Tests")
      printf 'Grafana web:%s\nGrafana PostgreSQL:%s\n' "${GRAFANA_PORT:-13000}" "${GRAFANA_POSTGRES_PORT:-15435}"
      ;;
    "Airflow Nightly Tests")
      printf 'Airflow web:%s\n' "${AIRFLOW_HOST_PORT:-18080}"
      ;;
    "PostgreSQL Adapter Tests")
      printf 'PostgreSQL:%s\n' "${POSTGRESQL_HOST_PORT:-${POSTGRESQL_PORT:-25432}}"
      ;;
    "MySQL Adapter Tests")
      printf 'MySQL:%s\n' "${MYSQL_HOST_PORT:-${MYSQL_PORT:-23306}}"
      ;;
    "ClickHouse Adapter Tests")
      printf 'ClickHouse HTTP:%s\nClickHouse native:%s\n' "${CLICKHOUSE_HTTP_HOST_PORT:-28123}" "${CLICKHOUSE_NATIVE_HOST_PORT:-29000}"
      ;;
    "StarRocks Adapter Tests")
      printf 'StarRocks query:%s\nStarRocks HTTP:%s\n' "${STARROCKS_QUERY_HOST_PORT:-29030}" "${STARROCKS_HTTP_HOST_PORT:-28030}"
      ;;
    "Trino Adapter Tests")
      printf 'Trino HTTP:%s\n' "${TRINO_HOST_PORT:-28080}"
      ;;
    "Greenplum Adapter Tests")
      printf 'Greenplum:%s\n' "${GREENPLUM_HOST_PORT:-15434}"
      ;;
    "Hive Adapter Tests")
      printf 'Hive metastore:%s\nHive thrift:%s\nHive web UI:%s\n' "${HIVE_METASTORE_HOST_PORT:-29083}" "${HIVE_THRIFT_HOST_PORT:-21000}" "${HIVE_WEBUI_HOST_PORT:-21002}"
      ;;
    "Spark Adapter Tests")
      printf 'Spark thrift:%s\nSpark UI:%s\n' "${SPARK_THRIFT_HOST_PORT:-31000}" "${SPARK_UI_HOST_PORT:-24040}"
      ;;
  esac
}

check_compose_host_ports_available() {
  local group_name="$1"
  local failed=0
  local spec
  local label
  local port

  while IFS= read -r spec; do
    [ -n "$spec" ] || continue
    label="${spec%%:*}"
    port="${spec##*:}"
    if [ -z "$port" ]; then
      continue
    fi
    if ! can_bind_host_port "$port"; then
      echo "Host port is already in use for ${group_name}: ${label} port ${port}" | tee -a "$LOG_FILE" >&2
      log_host_port_owner "$port"
      failed=1
    fi
  done < <(compose_host_port_specs "$group_name")

  if [ "$failed" -ne 0 ]; then
    test_exit_code=1
    return 1
  fi
  return 0
}

wait_for_tcp_readiness() {
  local label="$1"
  local host="$2"
  local port="$3"
  local timeout_seconds="${4:-300}"
  local deadline=$((SECONDS + timeout_seconds))
  local probe_output="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-${label//[^A-Za-z0-9_-]/_}-tcp-readiness-${GITHUB_RUN_ID:-$$}.log"

  log "Waiting for ${label} TCP readiness at ${host}:${port}"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if python3 - "$host" "$port" >"$probe_output" 2>&1 <<'PY'; then
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=5):
    pass
PY
      log "${label} TCP readiness probe succeeded"
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for ${label} TCP readiness at ${host}:${port}" | tee -a "$LOG_FILE" >&2
  if [ -s "$probe_output" ]; then
    log "Last ${label} TCP readiness probe output:"
    sed 's/^/  /' "$probe_output" | tee -a "$LOG_FILE"
  fi
  test_exit_code=1
  return 1
}

wait_for_http_readiness() {
  local label="$1"
  local url="$2"
  local timeout_seconds="${3:-300}"
  local deadline=$((SECONDS + timeout_seconds))
  local probe_output="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-${label//[^A-Za-z0-9_-]/_}-http-readiness-${GITHUB_RUN_ID:-$$}.log"

  log "Waiting for ${label} HTTP readiness at ${url}"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if python3 - "$url" >"$probe_output" 2>&1 <<'PY'; then
import sys
import urllib.request

url = sys.argv[1]
request = urllib.request.Request(url, headers={"User-Agent": "datus-nightly-readiness"})
with urllib.request.urlopen(request, timeout=10) as response:
    status = response.getcode()
    if status < 200 or status >= 400:
        raise RuntimeError(f"unexpected HTTP status {status}")
PY
      log "${label} HTTP readiness probe succeeded"
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for ${label} HTTP readiness at ${url}" | tee -a "$LOG_FILE" >&2
  if [ -s "$probe_output" ]; then
    log "Last ${label} HTTP readiness probe output:"
    sed 's/^/  /' "$probe_output" | tee -a "$LOG_FILE"
  fi
  test_exit_code=1
  return 1
}

wait_for_mysql_client_readiness() {
  local timeout_seconds="${1:-300}"
  local deadline=$((SECONDS + timeout_seconds))
  local probe_output="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-mysql-readiness-${GITHUB_RUN_ID:-$$}.log"

  log "Waiting for MySQL client readiness at ${MYSQL_HOST:-localhost}:${MYSQL_PORT:-3306}/${MYSQL_DATABASE:-test}"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if uv run python - <<'PY' >"$probe_output" 2>&1; then
import os

import pymysql

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST", "localhost"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "test_user"),
    password=os.getenv("MYSQL_PASSWORD", "test_password"),
    database=os.getenv("MYSQL_DATABASE", "test"),
    charset="utf8mb4",
    autocommit=True,
    connect_timeout=5,
    read_timeout=5,
    write_timeout=5,
)
try:
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        row = cursor.fetchone()
        if not row or row[0] != 1:
            raise RuntimeError(f"unexpected readiness result: {row!r}")
finally:
    conn.close()
PY
      log "MySQL client readiness probe succeeded"
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for MySQL client readiness" | tee -a "$LOG_FILE" >&2
  if [ -s "$probe_output" ]; then
    log "Last MySQL readiness probe output:"
    sed 's/^/  /' "$probe_output" | tee -a "$LOG_FILE"
  fi
  test_exit_code=1
  return 1
}

wait_for_starrocks_client_readiness() {
  local timeout_seconds="${1:-300}"
  local deadline=$((SECONDS + timeout_seconds))
  local probe_output="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-starrocks-readiness-${GITHUB_RUN_ID:-$$}.log"

  log "Waiting for StarRocks client readiness at ${STARROCKS_HOST:-127.0.0.1}:${STARROCKS_PORT:-9030}/${STARROCKS_DATABASE:-test}"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if uv run python - <<'PY' >"$probe_output" 2>&1; then
import os

import pymysql


def quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def is_alive(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


ACCESS_DENIED_ERROR_CODES = {1045, 1227, 5203}
database = os.getenv("STARROCKS_DATABASE", "test")
probe_table = f"__datus_starrocks_readiness_probe_{os.getpid()}"
conn = pymysql.connect(
    host=os.getenv("STARROCKS_HOST", "127.0.0.1"),
    port=int(os.getenv("STARROCKS_PORT", "9030")),
    user=os.getenv("STARROCKS_USER", "root"),
    password=os.getenv("STARROCKS_PASSWORD", ""),
    charset="utf8mb4",
    autocommit=True,
    connect_timeout=5,
    read_timeout=5,
    write_timeout=5,
)
try:
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        row = cursor.fetchone()
        if not row or row[0] != 1:
            raise RuntimeError(f"unexpected SELECT 1 result: {row!r}")

        try:
            cursor.execute("SHOW BACKENDS")
        except pymysql.err.OperationalError as exc:
            # StarRocks 5203 is the privilege-denied error seen when SHOW BACKENDS
            # lacks SYSTEM OPERATE/NODE privileges; MySQL 1045/1227 are equivalent
            # access-denied signals for this optional backend-status probe.
            error_code = exc.args[0] if exc.args else None
            if error_code not in ACCESS_DENIED_ERROR_CODES:
                raise
        else:
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description or []]
            alive_index = next((index for index, column in enumerate(columns) if column.lower() == "alive"), None)
            if alive_index is None:
                raise RuntimeError(f"SHOW BACKENDS did not return an Alive column: {columns!r}")
            alive_rows = [row for row in rows if is_alive(row[alive_index])]
            if not alive_rows:
                raise RuntimeError(f"SHOW BACKENDS has no alive backend: columns={columns!r} rows={rows!r}")

        if database:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(database)}")
            cursor.execute(f"USE {quote_identifier(database)}")
            cursor.execute(
                f"""
                CREATE TABLE {quote_identifier(probe_table)} (
                    `id` INT
                )
                ENGINE=OLAP
                DUPLICATE KEY(`id`)
                DISTRIBUTED BY HASH(`id`) BUCKETS 1
                PROPERTIES ("replication_num" = "1")
                """
            )
            cursor.execute(f"DROP TABLE IF EXISTS {quote_identifier(probe_table)}")
finally:
    conn.close()
PY
      log "StarRocks client readiness probe succeeded"
      return 0
    fi
    sleep 5
  done

  echo "Timed out waiting for StarRocks client readiness" | tee -a "$LOG_FILE" >&2
  if [ -s "$probe_output" ]; then
    log "Last StarRocks readiness probe output:"
    sed 's/^/  /' "$probe_output" | tee -a "$LOG_FILE"
  fi
  test_exit_code=1
  return 1
}

wait_for_compose_client_readiness() {
  local group_name="$1"
  local airflow_base

  case "$group_name" in
    "Superset Nightly Tests")
      wait_for_http_readiness "Superset" "${SUPERSET_URL%/}/health" 300
      ;;
    "Grafana Nightly Tests")
      wait_for_http_readiness "Grafana" "${GRAFANA_URL%/}/api/health" 300
      ;;
    "Airflow Nightly Tests")
      airflow_base="${AIRFLOW_URL%/}"
      airflow_base="${airflow_base%/api/v1}"
      wait_for_http_readiness "Airflow" "${airflow_base}/api/v1/health" 300
      ;;
    "PostgreSQL Adapter Tests")
      wait_for_tcp_readiness "PostgreSQL" "${POSTGRESQL_HOST:-localhost}" "${POSTGRESQL_PORT:-5432}" 300
      ;;
    "MySQL Adapter Tests")
      wait_for_mysql_client_readiness 300
      ;;
    "ClickHouse Adapter Tests")
      wait_for_tcp_readiness "ClickHouse" "${CLICKHOUSE_HOST:-127.0.0.1}" "${CLICKHOUSE_PORT:-8123}" 300
      ;;
    "StarRocks Adapter Tests")
      wait_for_starrocks_client_readiness 300
      ;;
    "Trino Adapter Tests")
      wait_for_http_readiness "Trino" "http://${TRINO_HOST:-127.0.0.1}:${TRINO_PORT:-8080}/v1/info" 300
      ;;
    "Greenplum Adapter Tests")
      wait_for_tcp_readiness "Greenplum" "${GREENPLUM_HOST:-localhost}" "${GREENPLUM_PORT:-5432}" 300
      ;;
    "Hive Adapter Tests")
      wait_for_tcp_readiness "HiveServer2" "${HIVE_HOST:-localhost}" "${HIVE_PORT:-10000}" 300
      ;;
    "Spark Adapter Tests")
      wait_for_tcp_readiness "Spark Thrift" "${SPARK_HOST:-localhost}" "${SPARK_PORT:-10000}" 300
      ;;
  esac
}

run_compose_suite() {
  local group_name="$1"
  local compose_file="$2"
  shift 2
  local service_specs=()
  local project_name
  local started_at
  started_at="$(utc_now)"

  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    service_specs+=("$1")
    shift
  done
  if [ "${1:-}" != "--" ]; then
    echo "Internal error: missing -- before command for ${group_name}" | tee -a "$LOG_FILE" >&2
    test_exit_code=1
    return 0
  fi
  shift

  if ! should_run_group "$group_name"; then
    log ""
    log "=== Skipping ${group_name} (NIGHTLY_GROUP_FILTER=${NIGHTLY_GROUP_FILTER}) ==="
    local now
    now="$(utc_now)"
    manifest_record_suite "$group_name" "blocking" "compose" "skipped" 0 "$now" "$now" "$@"
    return 0
  fi

  project_name="$(compose_project_name "$group_name")"
  local host_ports
  host_ports="$(compose_host_port_specs "$group_name")"
  manifest_record_compose_project "$group_name" "$project_name" "$compose_file" "$host_ports"
  log ""
  log "=== Starting ${group_name} Services (project=${project_name}) ==="
  log "Compose file: ${compose_file}"
  log "Host ports:"
  printf '%s\n' "$host_ports" | sed 's/^/  /' | tee -a "$LOG_FILE"

  compose_down "$compose_file" "$project_name"
  if ! check_compose_host_ports_available "$group_name"; then
    log "Skipping ${group_name} startup because required host ports are unavailable"
    manifest_record_suite "$group_name" "blocking" "compose" "failed" "$test_exit_code" "$started_at" "$(utc_now)" "$@"
    return 0
  fi

  if ! compose_up "$project_name" "$compose_file"; then
    compose_down "$compose_file" "$project_name"
    manifest_record_suite "$group_name" "blocking" "compose" "failed" "$test_exit_code" "$started_at" "$(utc_now)" "$@"
    return 0
  fi

  local spec
  for spec in "${service_specs[@]}"; do
    local service_name="${spec%%:*}"
    local timeout_seconds="${spec##*:}"
    if ! wait_for_service_health "$project_name" "$compose_file" "$service_name" "$timeout_seconds"; then
      compose_down "$compose_file" "$project_name"
      manifest_record_suite "$group_name" "blocking" "compose" "failed" "$test_exit_code" "$started_at" "$(utc_now)" "$@"
      return 0
    fi
  done

  if ! wait_for_compose_client_readiness "$group_name"; then
    dump_compose_diagnostics "$project_name" "$compose_file" "$group_name"
    compose_down "$compose_file" "$project_name"
    manifest_record_suite "$group_name" "blocking" "compose" "failed" "$test_exit_code" "$started_at" "$(utc_now)" "$@"
    return 0
  fi

  NIGHTLY_CURRENT_COMPOSE_FILE="$compose_file"
  NIGHTLY_CURRENT_COMPOSE_PROJECT="$project_name"
  NIGHTLY_CURRENT_HOST_PORTS="$host_ports"
  run_logged "$group_name" "$@"
  NIGHTLY_CURRENT_COMPOSE_FILE=""
  NIGHTLY_CURRENT_COMPOSE_PROJECT=""
  NIGHTLY_CURRENT_HOST_PORTS=""
  if [ "$last_command_exit_code" -ne 0 ]; then
    dump_compose_diagnostics "$project_name" "$compose_file" "$group_name"
  fi

  log ""
  log "=== Stopping ${group_name} Services (project=${project_name}) ==="
  compose_down "$compose_file" "$project_name"
  return 0
}

NIGHTLY_STARTED_AT="$(utc_now)"
export NIGHTLY_STARTED_AT
rm -f "$NIGHTLY_TRACE_REFERENCES_FILE" "$NIGHTLY_TRACE_SUMMARY_FILE" "$NIGHTLY_PROCESS_DIAGNOSTICS_FILE"

manifest_init

log "Nightly log: $LOG_FILE"
log "Nightly manifest: $NIGHTLY_MANIFEST_FILE"
log "Nightly failure classification: $NIGHTLY_FAILURE_CLASSIFICATION_FILE"
log "Nightly trace references: $NIGHTLY_TRACE_REFERENCES_FILE"
log "Nightly trace summary: $NIGHTLY_TRACE_SUMMARY_FILE"
log "Nightly process diagnostics: $NIGHTLY_PROCESS_DIAGNOSTICS_FILE"
log "DB_ADAPTERS_ROOT=$DB_ADAPTERS_ROOT"
log "BI_ADAPTERS_ROOT=$BI_ADAPTERS_ROOT"
log "SCHEDULER_ADAPTERS_ROOT=$SCHEDULER_ADAPTERS_ROOT"
log "STORAGE_ADAPTERS_ROOT=$STORAGE_ADAPTERS_ROOT"
log "NIGHTLY_HOME=$NIGHTLY_HOME"
log "DATUS_TEST_PROJECT_NAME=$DATUS_TEST_PROJECT_NAME"
log "UNIT_TEST_HOME=$UNIT_TEST_HOME"
log "NIGHTLY_PYTEST_BASETEMP=$NIGHTLY_PYTEST_BASETEMP"
log "NIGHTLY_PYTEST_ROOTS=${NIGHTLY_PYTEST_ROOTS[*]}"
log "NIGHTLY_COMPOSE_PROJECT_PREFIX=$NIGHTLY_COMPOSE_PROJECT_PREFIX"
log "SUPERSET_URL=$SUPERSET_URL SUPERSET_PORT=$SUPERSET_PORT SUPERSET_POSTGRES_HOST=$SUPERSET_POSTGRES_HOST SUPERSET_POSTGRES_PORT=$SUPERSET_POSTGRES_PORT"
log "GRAFANA_URL=$GRAFANA_URL GRAFANA_PORT=$GRAFANA_PORT GRAFANA_POSTGRES_HOST=$GRAFANA_POSTGRES_HOST GRAFANA_POSTGRES_PORT=$GRAFANA_POSTGRES_PORT"
log "AIRFLOW_URL=$AIRFLOW_URL AIRFLOW_HOST_PORT=$AIRFLOW_HOST_PORT"
log "POSTGRESQL_HOST=${POSTGRESQL_HOST:-} POSTGRESQL_PORT=${POSTGRESQL_PORT:-} POSTGRESQL_HOST_PORT=${POSTGRESQL_HOST_PORT:-}"
log "MYSQL_HOST=${MYSQL_HOST:-} MYSQL_PORT=${MYSQL_PORT:-} MYSQL_HOST_PORT=${MYSQL_HOST_PORT:-}"
log "CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-} CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-} CLICKHOUSE_HTTP_HOST_PORT=${CLICKHOUSE_HTTP_HOST_PORT:-} CLICKHOUSE_NATIVE_HOST_PORT=${CLICKHOUSE_NATIVE_HOST_PORT:-}"
log "STARROCKS_HOST=${STARROCKS_HOST:-} STARROCKS_PORT=${STARROCKS_PORT:-} STARROCKS_QUERY_HOST_PORT=${STARROCKS_QUERY_HOST_PORT:-} STARROCKS_HTTP_HOST_PORT=${STARROCKS_HTTP_HOST_PORT:-}"
log "TRINO_HOST=${TRINO_HOST:-} TRINO_PORT=${TRINO_PORT:-}"
log "GREENPLUM_HOST=${GREENPLUM_HOST:-} GREENPLUM_PORT=${GREENPLUM_PORT:-} GREENPLUM_HOST_PORT=${GREENPLUM_HOST_PORT:-}"
log "HIVE_HOST=${HIVE_HOST:-} HIVE_PORT=${HIVE_PORT:-} HIVE_METASTORE_HOST_PORT=${HIVE_METASTORE_HOST_PORT:-} HIVE_THRIFT_HOST_PORT=${HIVE_THRIFT_HOST_PORT:-} HIVE_WEBUI_HOST_PORT=${HIVE_WEBUI_HOST_PORT:-}"
log "SPARK_HOST=${SPARK_HOST:-} SPARK_PORT=${SPARK_PORT:-} SPARK_THRIFT_HOST_PORT=${SPARK_THRIFT_HOST_PORT:-} SPARK_UI_HOST_PORT=${SPARK_UI_HOST_PORT:-}"
if [ -n "$NIGHTLY_GROUP_FILTER" ]; then
  log "NIGHTLY_GROUP_FILTER=$NIGHTLY_GROUP_FILTER"
fi

if ! prepare_nightly_langfuse_tracing; then
  test_exit_code=1
  exit "$test_exit_code"
fi

validate_nightly_group_filter || exit 1
validate_pytest_basetemp "$NIGHTLY_PYTEST_BASETEMP" || exit 1
rm -rf "$NIGHTLY_PYTEST_BASETEMP"
if [ -n "${PYTEST_ADDOPTS:-}" ]; then
  export PYTEST_ADDOPTS="${PYTEST_ADDOPTS} --basetemp=$NIGHTLY_PYTEST_BASETEMP -p ci.pytest_trace_reference_plugin"
else
  export PYTEST_ADDOPTS="--basetemp=$NIGHTLY_PYTEST_BASETEMP -p ci.pytest_trace_reference_plugin"
fi
require_docker_runtime || exit "$test_exit_code"

run_logged_unfiltered "Flaky Registry Check" uv run python ci/check_flaky_registry.py --registry ci/flaky-registry.yml --strict

validate_unit_test_home "$UNIT_TEST_HOME" || exit 1
rm -rf "$UNIT_TEST_HOME"
run_logged "Full Unit Tests" run_with_agent_home "$UNIT_TEST_HOME" "$UNIT_TEST_PROJECT_ROOT" env DATUS_TEST_LAYER=unit uv run pytest tests/unit_tests/ -m "not nightly and not quarantine" --tb=short --verbose --timeout=300 --dist=loadscope -n auto

ensure_nightly_kb_data

run_logged "MCP Server Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/tools/test_mcp_server.py --tb=short --verbose --timeout=60 --timeout-method=thread

run_logged "Gen Agent Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/agent/test_gen_semantic_model_agentic.py tests/integration/agent/test_gen_metrics_agentic.py --tb=short --verbose --timeout=600 --timeout-method=thread --reruns 1 --reruns-delay 5

run_logged "Reference Template Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/tools/test_reference_template.py --tb=short --verbose --timeout=600 --timeout-method=thread --reruns 1 --reruns-delay 5

run_logged "Web UI Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/regression/test_regression_web_e2e.py --tb=short --verbose --timeout=300 --timeout-method=thread --reruns 1 --reruns-delay 5

# These suites are not skipped. They are run by dedicated groups above/below
# so the broad "tests/" collection used by Main/Product E2E does not duplicate
# them before their required server/compose setup is ready.
NIGHTLY_DEDICATED_SUITE_DESELECTS=(
  --deselect tests/integration/tools/test_mcp_server.py
  --deselect tests/integration/agent/test_gen_semantic_model_agentic.py
  --deselect tests/integration/agent/test_gen_metrics_agentic.py
  --deselect tests/integration/agent/test_gen_dashboard_agentic.py
  --deselect tests/integration/agent/test_scheduler_agentic.py
  --deselect tests/integration/tools/test_bi_dashboard.py
  --deselect tests/integration/tools/test_bi_grafana.py
  --deselect tests/integration/tools/test_reference_template.py
  --deselect tests/integration/adapters/test_postgresql.py
  --deselect tests/integration/adapters/test_mysql.py
  --deselect tests/integration/adapters/test_clickhouse.py
  --deselect tests/integration/adapters/test_starrocks.py
  --deselect tests/integration/adapters/test_trino.py
  --deselect tests/integration/adapters/test_greenplum.py
  --deselect tests/integration/adapters/test_hive.py
  --deselect tests/integration/adapters/test_spark.py
  --deselect tests/integration/adapters/test_semantic_metricflow_duckdb.py
  --deselect tests/integration/adapters/test_semantic_metricflow_mysql.py
  --deselect tests/integration/adapters/test_semantic_metricflow_postgresql.py
  --deselect tests/regression/test_regression_web_e2e.py
)

run_logged "Main Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m "nightly and not provider_health and not product_e2e" "${NIGHTLY_PYTEST_ROOTS[@]}" "${NIGHTLY_DEDICATED_SUITE_DESELECTS[@]}" --tb=short --verbose --timeout=300 --timeout-method=thread --reruns 1 --reruns-delay 5 --dist=loadscope -n auto

run_logged "Product E2E Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m "nightly and product_e2e and not provider_health" "${NIGHTLY_PYTEST_ROOTS[@]}" "${NIGHTLY_DEDICATED_SUITE_DESELECTS[@]}" --tb=short --verbose --timeout=600 --timeout-method=thread --reruns 1 --reruns-delay 5

run_compose_suite "Superset Nightly Tests" "$SUPERSET_COMPOSE" "postgres:300" "superset:1200" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/agent/test_gen_dashboard_agentic.py tests/integration/tools/test_bi_dashboard.py --tb=short --verbose --timeout=600 --timeout-method=thread --reruns 1 --reruns-delay 5

run_compose_suite "Grafana Nightly Tests" "$GRAFANA_COMPOSE" "postgres:300" "grafana:600" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/tools/test_bi_grafana.py --tb=short --verbose --timeout=300 --timeout-method=thread --reruns 1 --reruns-delay 5

run_compose_suite "Airflow Nightly Tests" "$AIRFLOW_COMPOSE" "airflow:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/agent/test_scheduler_agentic.py --tb=short --verbose --timeout=600 --timeout-method=thread --reruns 1 --reruns-delay 5

run_logged "PostgreSQL Storage Adapter Tests" env DATUS_TEST_LAYER=nightly uv run --no-sync pytest "$STORAGE_ADAPTERS_ROOT/datus-storage-postgresql/tests" --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "PostgreSQL Adapter Tests" "$POSTGRES_COMPOSE" "postgres:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_postgresql.py tests/integration/adapters/test_semantic_metricflow_postgresql.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "MySQL Adapter Tests" "$MYSQL_COMPOSE" "mysql:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_mysql.py tests/integration/adapters/test_semantic_metricflow_mysql.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "ClickHouse Adapter Tests" "$CLICKHOUSE_COMPOSE" "clickhouse:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_clickhouse.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "StarRocks Adapter Tests" "$STARROCKS_COMPOSE" "starrocks:600" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_starrocks.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "Trino Adapter Tests" "$TRINO_COMPOSE" "trino:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_trino.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "Greenplum Adapter Tests" "$GREENPLUM_COMPOSE" "greenplum:600" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_greenplum.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "Hive Adapter Tests" "$HIVE_COMPOSE" "hive-metastore:600" "hive-server:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_hive.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_compose_suite "Spark Adapter Tests" "$SPARK_COMPOSE" "spark-thrift:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_spark.py --tb=short --verbose --timeout=300 --timeout-method=thread
run_logged "MetricFlow DuckDB Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m nightly tests/integration/adapters/test_semantic_metricflow_duckdb.py --tb=short --verbose --timeout=300 --timeout-method=thread

run_logged_warn_only "Provider Health Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" env DATUS_TEST_LAYER=nightly uv run pytest -m "nightly and provider_health" "${NIGHTLY_PYTEST_ROOTS[@]}" --tb=short --verbose --timeout=300 --timeout-method=thread --reruns 1 --reruns-delay 5

run_logged_unfiltered "Flaky Log Classification" uv run python ci/check_flaky_registry.py --registry ci/flaky-registry.yml --log-file "$LOG_FILE" --warn-only

manifest_finalize
provider_coverage_finalize
failure_classification_finalize
collect_nightly_trace_diagnostics

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "log_file=$LOG_FILE" >> "$GITHUB_OUTPUT"
  echo "manifest_file=$NIGHTLY_MANIFEST_FILE" >> "$GITHUB_OUTPUT"
  echo "failure_classification_file=$NIGHTLY_FAILURE_CLASSIFICATION_FILE" >> "$GITHUB_OUTPUT"
  echo "provider_coverage_manifest_file=$PROVIDER_COVERAGE_MANIFEST_FILE" >> "$GITHUB_OUTPUT"
  echo "nightly_trace_references_file=$NIGHTLY_TRACE_REFERENCES_FILE" >> "$GITHUB_OUTPUT"
  echo "nightly_trace_summary_file=$NIGHTLY_TRACE_SUMMARY_FILE" >> "$GITHUB_OUTPUT"
  echo "nightly_process_diagnostics_file=$NIGHTLY_PROCESS_DIAGNOSTICS_FILE" >> "$GITHUB_OUTPUT"
  echo "test_exit_code=$test_exit_code" >> "$GITHUB_OUTPUT"
fi

exit "$test_exit_code"
