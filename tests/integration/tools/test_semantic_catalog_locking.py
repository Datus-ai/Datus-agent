# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Native model loading must not reacquire a publisher's lock on another thread."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.acceptance, pytest.mark.nightly]


@pytest.mark.parametrize("legacy_version", [False, True])
def test_native_catalog_compilation_while_holding_the_artifact_lock(tmp_path, legacy_version):
    fixture = Path(__file__).parents[2] / "data/semantic_models/window_filters/model.yaml"
    model = tmp_path / "subject/semantic_models/warehouse/model.yaml"
    model.parent.mkdir(parents=True)
    text = fixture.read_text()
    if legacy_version:
        document = yaml.safe_load(text)
        version = document.pop("version")
        text = yaml.safe_dump({"version": version, "semantic_model": [document]}, sort_keys=False)
    model.write_text(text)
    # A subprocess bounds native flock/GIL deadlocks as well as Python RLocks.
    # Run the real GenerationTools -> DosiRuntime -> native Engine chain while
    # holding the same writer lock used by reconcile and semantic publication.
    program = """
import sys
from types import SimpleNamespace
from pathlib import Path
import yaml
from datus.storage.semantic_model.artifact_file import semantic_artifact_lock
from datus.tools.func_tool.generation_tools import GenerationTools

model = Path(sys.argv[1])
tools = GenerationTools.__new__(GenerationTools)
tools.agent_config = SimpleNamespace(current_datasource='warehouse', current_db_config=lambda _: None)
with semantic_artifact_lock(model):
    catalog = tools._compiled_metric_catalog(str(model))
assert 'score_total' in catalog, sorted(catalog)
assert yaml.safe_load(model.read_text())['name'] == 'window_filters_model'
print('native catalog compiled under writer lock')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(model)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native catalog compiled under writer lock" in result.stdout
