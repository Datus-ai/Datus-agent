#!/bin/bash
set -e

python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --debug
python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --components reference_sql --sql_dir sample_data/california_schools/reference_sql --subject_tree "bird/california_schools/FRPM_Meal_Analysis,bird/california_schools/Enrollment_Demographics,bird/california_schools/SAT_Academic_Performance,bird/debit_card_specializing,bird/student_club" --kb_update_strategy overwrite
python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --components metrics --success_story sample_data/california_schools/success_story.csv

python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace ssb_sqlite --kb_update_strategy overwrite --debug