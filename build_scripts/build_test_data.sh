#!/usr/bin/env bash
set -euo pipefail

DATUS_TEST_HOME="${DATUS_TEST_HOME:-$HOME/.datus/tests}"
export DATUS_TEST_HOME
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

validate_test_home() {
  local path="${DATUS_TEST_HOME%/}"
  local home="${HOME:-}"
  local github_workspace="${GITHUB_WORKSPACE:-}"

  case "$path" in
    "" | "." | "/" | "~" | "$home" | "$PWD" | ".." | "../"* | *"/.." | *"/../"* | "./"* | *"/." | *"/./"*)
      echo "Refusing to remove unsafe DATUS_TEST_HOME: '$DATUS_TEST_HOME'" >&2
      exit 1
      ;;
  esac

  case "$path" in
    /*) ;;
    *)
      echo "Refusing to remove non-absolute DATUS_TEST_HOME: '$DATUS_TEST_HOME'" >&2
      exit 1
      ;;
  esac

  if [ "$path" = "$home/.datus/tests" ] || [ "$path" = "$home/.datus_test_data" ]; then
    DATUS_TEST_HOME="$path"
    export DATUS_TEST_HOME
    return
  fi

  if [ "$path" = "${REPO_ROOT%/}/.datus_test_data" ]; then
    DATUS_TEST_HOME="$path"
    export DATUS_TEST_HOME
    return
  fi

  if [ -n "$github_workspace" ] && [ "$path" = "${github_workspace%/}/.datus_test_data" ]; then
    DATUS_TEST_HOME="$path"
    export DATUS_TEST_HOME
    return
  fi

  echo "Refusing to remove DATUS_TEST_HOME outside expected test roots: '$DATUS_TEST_HOME'" >&2
  exit 1
}

run_bootstrap_kb() {
  uv run python - "$@" <<'PY'
from agents import set_tracing_disabled

set_tracing_disabled(True)

from datus.main import main

raise SystemExit(main())
PY
}

# Clean old data before creating a cacheable, deterministic fixture set.
validate_test_home
rm -rf "$DATUS_TEST_HOME"
mkdir -p "$DATUS_TEST_HOME"

# Phase 1: create datasource metadata. Keep this serial because the storage
# backend writes shared tables and indexes under the same test home.
run_bootstrap_kb bootstrap-kb --config tests/conf/agent.yml --datasource bird_school --kb_update_strategy overwrite --debug --yes
run_bootstrap_kb bootstrap-kb --config tests/conf/agent.yml --datasource ssb_sqlite --kb_update_strategy overwrite --debug --yes

# Phase 2: build bird_school contextual stores. These target the same project
# storage and subject tree, so serial execution is more important than speed.
run_bootstrap_kb bootstrap-kb --config tests/conf/agent.yml --datasource bird_school --components reference_sql --sql_dir datus/sample_data/california_schools/reference_sql --subject_tree "california_schools/Continuation/Free_Rate,california_schools/Charter/Education_Location,california_schools/Charter-Fund/Phone,california_schools/SAT_Score/Average,california_schools/SAT_Score/Excellence_Rate,california_schools/FRPM_Enrollment/Rate,california_schools/Enrollment/Total" --kb_update_strategy overwrite --yes
run_bootstrap_kb bootstrap-kb --config tests/conf/agent.yml --datasource bird_school --kb_update_strategy overwrite --components metrics --success_story datus/sample_data/california_schools/success_story.csv --subject_tree "california_schools/Students_K-12/Free_Rate,california_schools/Education/Location" --yes
run_bootstrap_kb bootstrap-kb --config tests/conf/agent.yml --datasource bird_school --components reference_template --template_dir datus/sample_data/california_schools/reference_template --subject_tree "california_schools/Free_Rate/Query,california_schools/Charter/Zip,california_schools/SAT_Score/Phone,california_schools/Enrollment/Summary,california_schools/Stats/School_Count" --kb_update_strategy overwrite --yes

CACHE_READY_DIR="$DATUS_TEST_HOME/data"
if [ -n "${DATUS_TEST_PROJECT_NAME:-}" ]; then
  CACHE_READY_DIR="$CACHE_READY_DIR/$DATUS_TEST_PROJECT_NAME/datus_db"
fi

if [ ! -d "$CACHE_READY_DIR" ] || ! find "$CACHE_READY_DIR" -mindepth 1 -maxdepth 5 -type f -size +0 -print -quit | grep -q .; then
  echo "Expected test data under $CACHE_READY_DIR, but no cacheable data was produced" >&2
  exit 1
fi

echo "Test data created under $CACHE_READY_DIR"
