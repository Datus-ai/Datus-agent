"""CI catches policy drift even when runtime filtering safely hides the tool."""

import json
import re
import subprocess
import sys
from dataclasses import replace

import pytest

from datus.tools.registry import policy_validation as validation
from datus.tools.registry.capabilities import TOOL_CAPABILITIES
from datus.tools.registry.contracts import ToolCapability


@pytest.fixture
def baseline():
    return json.loads(validation.BASELINE_PATH.read_text())


def test_current_policy_contracts_pass_ci():
    result = subprocess.run(
        [sys.executable, "-m", "datus.tools.registry.policy_validation", "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("tool", ["semantic_tools.get_metric", "db_tools.list_tables", "filesystem_tools.write_file"])
def test_deleted_contract_is_not_silently_filtered(monkeypatch, baseline, tool):
    monkeypatch.delitem(TOOL_CAPABILITIES, tool)
    with pytest.raises(ValueError, match=re.escape(tool)):
        validation.validate_policy_contracts(baseline)


@pytest.mark.parametrize("tool", ["semantic_tools.get_metric", "db_tools.list_tables", "filesystem_tools.write_file"])
def test_read_write_changes_require_policy_review(monkeypatch, baseline, tool):
    previous = TOOL_CAPABILITIES[tool]
    access = {"read": "write", "write": "read"}[previous.access]
    monkeypatch.setitem(TOOL_CAPABILITIES, tool, replace(previous, access=access))
    with pytest.raises(ValueError, match=re.escape(tool)):
        validation.validate_policy_contracts(baseline)


def test_wildcard_tool_removal_is_detected_even_with_valid_catalog(monkeypatch, baseline):
    # gen_sql's db_tools.* has no literal entry for this method to go stale.
    tool = "db_tools.export_query"
    assert tool in baseline["gen_sql"]
    providers = validation._provider_tools
    monkeypatch.setattr(validation, "_provider_tools", lambda kind: [name for name in providers(kind) if name != tool])
    with pytest.raises(ValueError, match=re.escape(tool)):
        validation.validate_policy_contracts(baseline)


def test_wildcard_access_change_is_detected_without_read_only_type_constraint(monkeypatch, baseline):
    tool = "db_tools.get_datasource_identity"
    monkeypatch.setitem(TOOL_CAPABILITIES, tool, ToolCapability("write"))
    with pytest.raises(ValueError, match=re.escape(tool)):
        validation.validate_policy_contracts(baseline)


@pytest.fixture
def future_reader(monkeypatch):
    tool = "db_tools.future_reader"
    monkeypatch.setitem(TOOL_CAPABILITIES, tool, ToolCapability("read"))
    providers = validation._provider_tools
    additions = [tool]
    monkeypatch.setattr(validation, "_provider_tools", lambda kind: [*providers(kind), *additions])
    return tool, additions


def test_new_tool_must_be_recorded_before_ci_passes(monkeypatch, baseline, future_reader, tmp_path):
    tool, _ = future_reader
    assert tool not in baseline["gen_sql"]
    with pytest.raises(ValueError, match=re.escape(tool)):
        validation.validate_policy_contracts(baseline)

    path = tmp_path / "baseline.json"
    monkeypatch.setattr(validation, "BASELINE_PATH", path)
    monkeypatch.setattr(sys, "argv", ["policy_validation", "--update"])
    validation.main()
    recorded = json.loads(path.read_text())
    assert recorded["gen_sql"][tool] == "read"
    validation.validate_policy_contracts(recorded)


def test_new_tool_access_change_is_detected_after_recording(monkeypatch, future_reader):
    tool, _ = future_reader
    recorded = validation.current_policy_contracts()
    assert recorded["gen_sql"][tool] == "read"
    validation.validate_policy_contracts(recorded)

    monkeypatch.setitem(TOOL_CAPABILITIES, tool, ToolCapability("write"))
    with pytest.raises(ValueError, match=rf"{re.escape(tool)} access changed from read to write"):
        validation.validate_policy_contracts(recorded)


def test_new_tool_deletion_is_detected_after_recording(monkeypatch, future_reader):
    tool, additions = future_reader
    recorded = validation.current_policy_contracts()
    assert recorded["gen_sql"][tool] == "read"
    validation.validate_policy_contracts(recorded)

    additions.clear()
    monkeypatch.delitem(TOOL_CAPABILITIES, tool)
    with pytest.raises(ValueError, match=rf"{re.escape(tool)} was removed"):
        validation.validate_policy_contracts(recorded)


def test_new_explicit_typo_is_rejected_before_baseline_comparison(monkeypatch, baseline):
    policy = validation.SUBAGENT_POLICIES["analytics_explorer"]
    monkeypatch.setitem(
        validation.SUBAGENT_POLICIES, "analytics_explorer", replace(policy, tools=(*policy.tools, "db_tools.typo"))
    )
    with pytest.raises(ValueError, match="db_tools.typo"):
        validation.validate_policy_contracts(baseline)


def test_ci_command_returns_failure_on_stale_baseline(monkeypatch, baseline, tmp_path, capsys):
    baseline["gen_sql"]["db_tools.retired_reader"] = "read"
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    monkeypatch.setattr(validation, "BASELINE_PATH", path)
    monkeypatch.setattr(sys, "argv", ["policy_validation", "--check"])
    with pytest.raises(SystemExit) as failure:
        validation.main()
    assert failure.value.code == 1
    assert "db_tools.retired_reader" in capsys.readouterr().err
