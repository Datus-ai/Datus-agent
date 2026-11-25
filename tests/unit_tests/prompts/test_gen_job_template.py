# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from datus.prompts.prompt_manager import PromptManager
from datus.utils.path_manager import DatusPathManager


def test_gen_job_prompt_routes_explicit_append_around_ddl(tmp_path):
    manager = PromptManager(path_manager=DatusPathManager(tmp_path / "home"))
    prompt = manager.render_template("gen_job_system")

    target_decision = prompt.split("### Phase 2: Inspect target", 1)[1].split("### Phase 3:", 1)[0]
    append = "For an explicitly requested append, verify schema compatibility and skip to Phase 4"
    replace = "For replacement, overwrite, truncation, or drop/recreate, require explicit authorization"
    assert append in target_decision
    assert replace in target_decision
    assert target_decision.index(append) < target_decision.index(replace)
