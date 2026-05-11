#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT" || {
  echo "Failed to enter repository root: $REPO_ROOT" >&2
  exit 1
}

LOG_FILE="${NIGHTLY_LOG_FILE:-test_output_nightly_$(date +%Y%m%d_%H%M%S).log}"
test_exit_code=0
last_command_exit_code=0
NIGHTLY_GROUP_FILTER="${NIGHTLY_GROUP_FILTER:-}"
AGENT_TEST_CONFIG="${AGENT_TEST_CONFIG:-tests/conf/agent.yml}"
DATUS_TEST_PROJECT_NAME="${DATUS_TEST_PROJECT_NAME:-datus_agent_nightly}"
export DATUS_TEST_PROJECT_NAME

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
EXTERNAL_REPOS_ROOT="${EXTERNAL_REPOS_ROOT:-${REPO_ROOT}/external}"
NIGHTLY_HOME="${DATUS_TEST_HOME:-${REPO_ROOT}/.datus_test_data}"
NIGHTLY_PROJECT_ROOT="${NIGHTLY_PROJECT_ROOT:-${NIGHTLY_HOME}/workspace}"
UNIT_TEST_HOME="${NIGHTLY_UNIT_TEST_HOME:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/datus-agent-nightly-unit-${GITHUB_RUN_ID:-$$}}"
UNIT_TEST_PROJECT_ROOT="${NIGHTLY_UNIT_TEST_PROJECT_ROOT:-${UNIT_TEST_HOME}/workspace}"
AGENT_TEST_CONFIG_BACKUP="${AGENT_TEST_CONFIG_BACKUP:-${TMPDIR:-/tmp}/datus-agent-nightly-config-${GITHUB_RUN_ID:-$$}.bak}"

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

POSTGRES_COMPOSE="${POSTGRES_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-postgresql/docker-compose.yml}"
MYSQL_COMPOSE="${MYSQL_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-mysql/docker-compose.yml}"
CLICKHOUSE_COMPOSE="${CLICKHOUSE_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-clickhouse/docker-compose.yml}"
STARROCKS_COMPOSE="${STARROCKS_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-starrocks/docker-compose.yml}"
TRINO_COMPOSE="${TRINO_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-trino/docker-compose.yml}"
GREENPLUM_COMPOSE="${GREENPLUM_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-greenplum/docker-compose.yml}"
HIVE_COMPOSE="${HIVE_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-hive/docker-compose.yml}"
SPARK_COMPOSE="${SPARK_COMPOSE:-${DB_ADAPTERS_ROOT}/datus-spark/docker-compose.yml}"
SUPERSET_COMPOSE="${SUPERSET_COMPOSE:-${BI_ADAPTERS_ROOT}/datus-bi-superset/tests/integration/docker-compose.yml}"
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
  "$AIRFLOW_COMPOSE"
)

COMPOSE_GROUPS=(
  "Superset Nightly Tests"
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

log() {
  echo "$@" | tee -a "$LOG_FILE"
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

require_docker_runtime() {
  if ! will_run_any_compose_suite; then
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for nightly compose-backed suites" | tee -a "$LOG_FILE" >&2
    test_exit_code=127
    return 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not available for nightly compose-backed suites" | tee -a "$LOG_FILE" >&2
    test_exit_code=127
    return 1
  fi

  if ! has_docker_compose; then
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

compose_down() {
  local compose_file="$1"
  docker_compose -f "$compose_file" down -v --remove-orphans >/dev/null 2>&1 || true
}

cleanup_all_compose() {
  set +e
  if ! has_docker_compose; then
    echo "Docker Compose is not available; skipping compose cleanup"
    return 0
  fi
  local compose_file
  for compose_file in "${COMPOSE_FILES[@]}"; do
    if [ -f "$compose_file" ]; then
      echo "Stopping services from $compose_file"
      docker_compose -f "$compose_file" down -v --remove-orphans || true
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
  for dataset in schema_metadata.lance schema_value.lance metrics.lance reference_sql.lance ext_knowledge.lance reference_template.lance; do
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
  restore_agent_test_config
  rm -f "$AGENT_TEST_CONFIG_BACKUP"
  cleanup_all_compose
}

if [ "${1:-}" = "--cleanup-only" ]; then
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
export SUPERSET_PORT="${SUPERSET_PORT:-8088}"
export SUPERSET_POSTGRES_HOST="${SUPERSET_POSTGRES_HOST:-127.0.0.1}"
export SUPERSET_POSTGRES_PORT="${SUPERSET_POSTGRES_PORT:-5433}"
export SUPERSET_URL="${SUPERSET_URL:-http://127.0.0.1:${SUPERSET_PORT}}"
export SUPERSET_USER="${SUPERSET_USER:-admin}"
export SUPERSET_PASS="${SUPERSET_PASS:-admin}"
export AIRFLOW_HOST_PORT="${AIRFLOW_HOST_PORT:-8080}"
export AIRFLOW_URL="${AIRFLOW_URL:-http://127.0.0.1:${AIRFLOW_HOST_PORT}/api/v1}"
export AIRFLOW_USER="${AIRFLOW_USER:-admin}"
export AIRFLOW_USERNAME="${AIRFLOW_USERNAME:-$AIRFLOW_USER}"
export AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-admin}"

if [ "${NIGHTLY_FORCE_ADAPTER_ENV:-1}" = "1" ]; then
  export POSTGRESQL_HOST=localhost
  export POSTGRESQL_PORT=5432
  export POSTGRESQL_USER=test_user
  export POSTGRESQL_PASSWORD=test_password
  export POSTGRESQL_DATABASE=test
  export POSTGRESQL_SCHEMA=public

  export MYSQL_HOST=localhost
  export MYSQL_PORT=3306
  export MYSQL_USER=test_user
  export MYSQL_PASSWORD=test_password
  export MYSQL_DATABASE=test

  export CLICKHOUSE_HTTP_HOST_PORT="${CLICKHOUSE_HTTP_HOST_PORT:-8123}"
  export CLICKHOUSE_NATIVE_HOST_PORT="${CLICKHOUSE_NATIVE_HOST_PORT:-9000}"
  export CLICKHOUSE_HOST=127.0.0.1
  export CLICKHOUSE_PORT="$CLICKHOUSE_HTTP_HOST_PORT"
  export CLICKHOUSE_USER=default_user
  export CLICKHOUSE_PASSWORD=default_test
  export CLICKHOUSE_DATABASE=default_test

  export STARROCKS_QUERY_HOST_PORT="${STARROCKS_QUERY_HOST_PORT:-9030}"
  export STARROCKS_HTTP_HOST_PORT="${STARROCKS_HTTP_HOST_PORT:-8030}"
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

  export GREENPLUM_HOST_PORT="${GREENPLUM_HOST_PORT:-15432}"
  export GREENPLUM_HOST=localhost
  export GREENPLUM_PORT="$GREENPLUM_HOST_PORT"
  export GREENPLUM_USER=gpadmin
  export GREENPLUM_PASSWORD=pivotal
  export GREENPLUM_DATABASE=postgres
  export GREENPLUM_SCHEMA=public

  export HIVE_HOST=localhost
  export HIVE_PORT=10000
  export HIVE_USERNAME=hive
  export HIVE_PASSWORD=
  export HIVE_DATABASE=default

  export SPARK_HOST=localhost
  export SPARK_PORT=10000
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
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  last_command_exit_code="$cmd_status"
  if [ "$cmd_status" -ne 0 ]; then
    test_exit_code="$cmd_status"
  fi
  return 0
}

run_logged() {
  local group_name="$1"
  shift
  if ! should_run_group "$group_name"; then
    log ""
    log "=== Skipping ${group_name} (NIGHTLY_GROUP_FILTER=${NIGHTLY_GROUP_FILTER}) ==="
    last_command_exit_code=0
    return 0
  fi

  run_logged_unfiltered "$group_name" "$@"
}

run_logged_warn_only_unfiltered() {
  local group_name="$1"
  shift

  log ""
  log "=== ${group_name} (warn-only) ==="
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  last_command_exit_code="$cmd_status"
  if [ "$cmd_status" -ne 0 ]; then
    log "WARNING: ${group_name} failed with exit code ${cmd_status}; continuing because this group is non-blocking."
  fi
  return 0
}

run_logged_warn_only() {
  local group_name="$1"
  shift
  if ! should_run_group "$group_name"; then
    log ""
    log "=== Skipping ${group_name} (NIGHTLY_GROUP_FILTER=${NIGHTLY_GROUP_FILTER}) ==="
    last_command_exit_code=0
    return 0
  fi

  run_logged_warn_only_unfiltered "$group_name" "$@"
}

compose_up() {
  local compose_file="$1"
  shift
  if [ ! -f "$compose_file" ]; then
    echo "Missing compose file: $compose_file" | tee -a "$LOG_FILE" >&2
    test_exit_code=1
    return 1
  fi
  docker_compose -f "$compose_file" up -d --build "$@" 2>&1 | tee -a "$LOG_FILE"
  local cmd_status=${PIPESTATUS[0]}
  if [ "$cmd_status" -ne 0 ]; then
    test_exit_code="$cmd_status"
    return 1
  fi
  return 0
}

wait_for_service_health() {
  local compose_file="$1"
  local service_name="$2"
  local timeout_seconds="$3"
  local container_id=""
  local has_health=""
  local status=""
  local deadline=$((SECONDS + timeout_seconds))

  container_id="$(docker_compose -f "$compose_file" ps -q "$service_name")"
  if [ -z "$container_id" ]; then
    echo "No container found for service '$service_name' in $compose_file" | tee -a "$LOG_FILE" >&2
    docker_compose -f "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
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

  echo "Timed out waiting for service '$service_name' from $compose_file" | tee -a "$LOG_FILE" >&2
  docker_compose -f "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
  docker_compose -f "$compose_file" logs --tail=200 2>&1 | tee -a "$LOG_FILE" || true
  test_exit_code=1
  return 1
}

dump_compose_diagnostics() {
  local compose_file="$1"
  local group_name="$2"

  log ""
  log "=== ${group_name} Service Diagnostics ==="
  docker_compose -f "$compose_file" ps 2>&1 | tee -a "$LOG_FILE" || true
  docker_compose -f "$compose_file" logs --tail=200 2>&1 | tee -a "$LOG_FILE" || true
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

wait_for_compose_client_readiness() {
  local group_name="$1"
  local airflow_base

  case "$group_name" in
    "Superset Nightly Tests")
      wait_for_http_readiness "Superset" "${SUPERSET_URL%/}/health" 300
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
      wait_for_tcp_readiness "StarRocks" "${STARROCKS_HOST:-127.0.0.1}" "${STARROCKS_PORT:-9030}" 300
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
    return 0
  fi

  log ""
  log "=== Starting ${group_name} Services ==="
  compose_down "$compose_file"
  if ! compose_up "$compose_file"; then
    compose_down "$compose_file"
    return 0
  fi

  local spec
  for spec in "${service_specs[@]}"; do
    local service_name="${spec%%:*}"
    local timeout_seconds="${spec##*:}"
    if ! wait_for_service_health "$compose_file" "$service_name" "$timeout_seconds"; then
      compose_down "$compose_file"
      return 0
    fi
  done

  if ! wait_for_compose_client_readiness "$group_name"; then
    dump_compose_diagnostics "$compose_file" "$group_name"
    compose_down "$compose_file"
    return 0
  fi

  run_logged "$group_name" "$@"
  if [ "$last_command_exit_code" -ne 0 ]; then
    dump_compose_diagnostics "$compose_file" "$group_name"
  fi

  log ""
  log "=== Stopping ${group_name} Services ==="
  compose_down "$compose_file"
  return 0
}

log "Nightly log: $LOG_FILE"
log "DB_ADAPTERS_ROOT=$DB_ADAPTERS_ROOT"
log "BI_ADAPTERS_ROOT=$BI_ADAPTERS_ROOT"
log "SCHEDULER_ADAPTERS_ROOT=$SCHEDULER_ADAPTERS_ROOT"
log "NIGHTLY_HOME=$NIGHTLY_HOME"
log "DATUS_TEST_PROJECT_NAME=$DATUS_TEST_PROJECT_NAME"
log "UNIT_TEST_HOME=$UNIT_TEST_HOME"
log "SUPERSET_URL=$SUPERSET_URL SUPERSET_PORT=$SUPERSET_PORT SUPERSET_POSTGRES_HOST=$SUPERSET_POSTGRES_HOST SUPERSET_POSTGRES_PORT=$SUPERSET_POSTGRES_PORT"
log "AIRFLOW_URL=$AIRFLOW_URL AIRFLOW_HOST_PORT=$AIRFLOW_HOST_PORT"
log "CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-} CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-} CLICKHOUSE_NATIVE_HOST_PORT=${CLICKHOUSE_NATIVE_HOST_PORT:-}"
log "STARROCKS_HOST=${STARROCKS_HOST:-} STARROCKS_PORT=${STARROCKS_PORT:-} STARROCKS_HTTP_HOST_PORT=${STARROCKS_HTTP_HOST_PORT:-}"
log "TRINO_HOST=${TRINO_HOST:-} TRINO_PORT=${TRINO_PORT:-}"
log "GREENPLUM_HOST=${GREENPLUM_HOST:-} GREENPLUM_PORT=${GREENPLUM_PORT:-} GREENPLUM_HOST_PORT=${GREENPLUM_HOST_PORT:-}"
if [ -n "$NIGHTLY_GROUP_FILTER" ]; then
  log "NIGHTLY_GROUP_FILTER=$NIGHTLY_GROUP_FILTER"
fi

validate_nightly_group_filter || exit 1
require_docker_runtime || exit "$test_exit_code"

run_logged_unfiltered "Flaky Registry Check" uv run python ci/check_flaky_registry.py --registry ci/flaky-registry.yml --strict

validate_unit_test_home "$UNIT_TEST_HOME" || exit 1
rm -rf "$UNIT_TEST_HOME"
run_logged "Full Unit Tests" run_with_agent_home "$UNIT_TEST_HOME" "$UNIT_TEST_PROJECT_ROOT" uv run pytest tests/unit_tests/ -m "not nightly and not quarantine" --tb=short --verbose --timeout=300 --dist=loadscope -n auto

ensure_nightly_kb_data

run_logged "MCP Server Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/tools/test_mcp_server.py --tb=short --verbose --timeout=60

run_logged "Gen Agent Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/agent/test_gen_semantic_model_agentic.py tests/integration/agent/test_gen_metrics_agentic.py tests/integration/agent/test_gen_ext_knowledge_agentic.py --tb=short --verbose --timeout=600 --reruns 1 --reruns-delay 5

run_logged "Reference Template Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/tools/test_reference_template.py --tb=short --verbose --timeout=600 --reruns 1 --reruns-delay 5

run_logged "Web UI Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/regression/test_regression_web_e2e.py --tb=short --verbose --timeout=300 --reruns 1 --reruns-delay 5

# These suites are not skipped. They are run by dedicated groups above/below
# so the broad "tests/" collection used by Main/Product E2E does not duplicate
# them before their required server/compose setup is ready.
NIGHTLY_DEDICATED_SUITE_DESELECTS=(
  --deselect tests/integration/tools/test_mcp_server.py
  --deselect tests/integration/agent/test_gen_semantic_model_agentic.py
  --deselect tests/integration/agent/test_gen_metrics_agentic.py
  --deselect tests/integration/agent/test_gen_ext_knowledge_agentic.py
  --deselect tests/integration/agent/test_gen_dashboard_agentic.py
  --deselect tests/integration/agent/test_scheduler_agentic.py
  --deselect tests/integration/tools/test_bi_dashboard.py
  --deselect tests/integration/tools/test_reference_template.py
  --deselect tests/integration/adapters/test_postgresql.py
  --deselect tests/integration/adapters/test_mysql.py
  --deselect tests/integration/adapters/test_clickhouse.py
  --deselect tests/integration/adapters/test_starrocks.py
  --deselect tests/integration/adapters/test_trino.py
  --deselect tests/integration/adapters/test_greenplum.py
  --deselect tests/integration/adapters/test_hive.py
  --deselect tests/integration/adapters/test_spark.py
  --deselect tests/regression/test_regression_web_e2e.py
)

run_logged "Main Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m "nightly and not provider_health and not product_e2e" tests/ "${NIGHTLY_DEDICATED_SUITE_DESELECTS[@]}" --tb=short --verbose --timeout=300 --reruns 1 --reruns-delay 5 --dist=loadscope -n auto

run_logged "Product E2E Nightly Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m "nightly and product_e2e and not provider_health" tests/ "${NIGHTLY_DEDICATED_SUITE_DESELECTS[@]}" --tb=short --verbose --timeout=600 --reruns 1 --reruns-delay 5

run_compose_suite "Superset Nightly Tests" "$SUPERSET_COMPOSE" "postgres:300" "superset:1200" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/agent/test_gen_dashboard_agentic.py tests/integration/tools/test_bi_dashboard.py --tb=short --verbose --timeout=600 --reruns 1 --reruns-delay 5

run_compose_suite "Airflow Nightly Tests" "$AIRFLOW_COMPOSE" "airflow:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/agent/test_scheduler_agentic.py --tb=short --verbose --timeout=600 --reruns 1 --reruns-delay 5

run_compose_suite "PostgreSQL Adapter Tests" "$POSTGRES_COMPOSE" "postgres:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_postgresql.py --tb=short --verbose --timeout=300
run_compose_suite "MySQL Adapter Tests" "$MYSQL_COMPOSE" "mysql:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_mysql.py --tb=short --verbose --timeout=300
run_compose_suite "ClickHouse Adapter Tests" "$CLICKHOUSE_COMPOSE" "clickhouse:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_clickhouse.py --tb=short --verbose --timeout=300
run_compose_suite "StarRocks Adapter Tests" "$STARROCKS_COMPOSE" "starrocks:600" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_starrocks.py --tb=short --verbose --timeout=300
run_compose_suite "Trino Adapter Tests" "$TRINO_COMPOSE" "trino:300" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_trino.py --tb=short --verbose --timeout=300
run_compose_suite "Greenplum Adapter Tests" "$GREENPLUM_COMPOSE" "greenplum:600" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_greenplum.py --tb=short --verbose --timeout=300
run_compose_suite "Hive Adapter Tests" "$HIVE_COMPOSE" "hive-metastore:600" "hive-server:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_hive.py --tb=short --verbose --timeout=300
run_compose_suite "Spark Adapter Tests" "$SPARK_COMPOSE" "spark-thrift:900" -- run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m nightly tests/integration/adapters/test_spark.py --tb=short --verbose --timeout=300

run_logged_warn_only "Provider Health Tests" run_with_agent_home "$NIGHTLY_HOME" "$NIGHTLY_PROJECT_ROOT" uv run pytest -m "nightly and provider_health" tests/ --tb=short --verbose --timeout=300 --reruns 1 --reruns-delay 5

run_logged_unfiltered "Flaky Log Classification" uv run python ci/check_flaky_registry.py --registry ci/flaky-registry.yml --log-file "$LOG_FILE" --warn-only

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "log_file=$LOG_FILE" >> "$GITHUB_OUTPUT"
  echo "test_exit_code=$test_exit_code" >> "$GITHUB_OUTPUT"
fi

exit "$test_exit_code"
