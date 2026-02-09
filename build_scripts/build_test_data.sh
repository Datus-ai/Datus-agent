#!/bin/bash
set -e
# clean old data
rm -rf ~/.datus/tests

python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --debug --yes
python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --components reference_sql --sql_dir sample_data/california_schools/reference_sql --subject_tree "california_schools/Continuation/Free_Rate,california_schools/Charter/Education_Location,california_schools/Charter-Fund/Phone,california_schools/SAT_Score/Average,california_schools/SAT_Score/Excellence_Rate,california_schools/FRPM_Enrollment/Rate,california_schools/Enrollment/Total" --kb_update_strategy overwrite --yes
python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --components metrics --success_story sample_data/california_schools/success_story.csv --subject_tree "california_schools/Students_K-12/Free_Rate,california_schools/Education/Location" --yes
python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace bird_school --kb_update_strategy overwrite --components ext_knowledge --success_story sample_data/california_schools/success_story.csv --subject_tree "california_schools/Students_K-12/Free_Rate,california_schools/Education/Location" --yes

python -m datus.main bootstrap-kb --config tests/conf/agent.yml --namespace ssb_sqlite --kb_update_strategy overwrite --debug --yes