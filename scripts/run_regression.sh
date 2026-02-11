#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Result tracking
PASSED=0
FAILED=0

SKIPPED=0

run_step() {
    local name="$1"; shift
    echo -e "\n${YELLOW}[STEP] $name${NC}"
    printf "\n[STEP] %s  (%s)\n" "$name" "$(date '+%H:%M:%S')" >> "$LOG_FILE"

    # Run command, tee output to log, capture real exit code via temp file
    local rc_file="/tmp/.regression_rc_$$"
    ( set +e; "$@" 2>&1; echo $? > "$rc_file" ) | tee -a "$LOG_FILE"
    local rc
    rc=$(cat "$rc_file" 2>/dev/null || echo 1)
    rm -f "$rc_file"

    if [[ $rc -eq 0 ]]; then
        echo -e "${GREEN}[PASS] $name${NC}"
        printf "[PASS] %s\n" "$name" >> "$LOG_FILE"
        PASSED=$((PASSED + 1))
    elif [[ $rc -eq 5 ]]; then
        # pytest exit code 5 = no tests collected (all skipped due to missing deps)
        echo -e "${YELLOW}[SKIP] $name (no tests collected)${NC}"
        printf "[SKIP] %s (no tests collected)\n" "$name" >> "$LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
    else
        echo -e "${RED}[FAIL] $name${NC}"
        printf "[FAIL] %s\n" "$name" >> "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
}

cd "$PROJECT_ROOT"

# ========== Log setup ==========
LOG_DIR="$PROJECT_ROOT/logs/regression"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/regression_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo "Regression test started at $(date)" > "$LOG_FILE"

# ========== LLM Regression Tests ==========
echo "=========================================="
echo "  LLM Regression Tests"
echo "=========================================="

run_step "LLM compatibility (R01-R05)" \
    pytest -m regression --tb=short --log-cli-level=INFO -k "LLMCompatibility" tests/regression/test_regression_llm.py -v

run_step "Model switching (R06)" \
    pytest -m regression --tb=short --log-cli-level=INFO -k "ModelSwitching" tests/regression/test_regression_llm.py -v

# ========== DB Connector Regression Tests ==========
echo ""
echo "=========================================="
echo "  DB Connector Regression Tests"
echo "=========================================="

run_step "SQLite + DuckDB (builtin)" \
    pytest -m regression --tb=short --log-cli-level=INFO tests/regression/test_regression_db.py -v

# ========== Web UI Regression Tests ==========
echo ""
echo "=========================================="
echo "  Web UI Regression Tests"
echo "=========================================="

run_step "Web UI components (R13)" \
    pytest -m regression --tb=short --log-cli-level=INFO tests/regression/test_regression_web.py -v

run_step "Web UI E2E (R13)" \
    pytest -m regression --tb=short --log-cli-level=INFO tests/regression/test_regression_web_e2e.py -v

# ========== Summary ==========
echo ""
echo "=========================================="
echo "  REGRESSION TEST SUMMARY"
echo "=========================================="
echo -e "  Passed:  ${GREEN}$PASSED${NC}"
echo -e "  Skipped: ${YELLOW}$SKIPPED${NC}"
echo -e "  Failed:  ${RED}$FAILED${NC}"
echo "=========================================="

# Write summary to log file (without color codes)
{
    printf "\n==========================================\n"
    printf "  REGRESSION TEST SUMMARY\n"
    printf "==========================================\n"
    printf "  Passed:  %d\n" "$PASSED"
    printf "  Skipped: %d\n" "$SKIPPED"
    printf "  Failed:  %d\n" "$FAILED"
    printf "==========================================\n"
    printf "Finished at %s\n" "$(date)"
} >> "$LOG_FILE"

echo "Full log saved to: $LOG_FILE"

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
