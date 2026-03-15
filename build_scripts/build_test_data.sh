#!/bin/bash
set -e
DATUS_TEST_HOME="${DATUS_TEST_HOME:-$HOME/.datus/tests}"
# clean old data
rm -rf "$DATUS_TEST_HOME"

# Phase 1: Create namespace metadata in parallel (no LLM calls, fast)
uv run python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --debug --yes &
pid_bird=$!
uv run python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace ssb_sqlite --kb_update_strategy overwrite --debug --yes &
pid_ssb=$!
wait $pid_bird || exit 1
wait $pid_ssb || exit 1

# Phase 2: Build reference_sql and metrics in parallel (different tables, safe)
uv run python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --components reference_sql --sql_dir sample_data/california_schools/reference_sql --subject_tree "california_schools/Continuation/Free_Rate,california_schools/Charter/Education_Location,california_schools/Charter-Fund/Phone,california_schools/SAT_Score/Average,california_schools/SAT_Score/Excellence_Rate,california_schools/FRPM_Enrollment/Rate,california_schools/Enrollment/Total" --kb_update_strategy overwrite --yes &
pid_ref=$!
uv run python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --components metrics --success_story sample_data/california_schools/success_story.csv --subject_tree "california_schools/Students_K-12/Free_Rate,california_schools/Education/Location" --yes &
pid_met=$!
wait $pid_ref || exit 1
wait $pid_met || exit 1

# Phase 3: Build ext_knowledge (depends on reference_sql + metrics as context)
uv run python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --components ext_knowledge --success_story sample_data/california_schools/success_story.csv --subject_tree "california_schools/Students_K-12/Free_Rate,california_schools/Education/Location" --yes
