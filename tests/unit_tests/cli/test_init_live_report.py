"""Live acceptance artifacts remain safe when a follow-up rewrites the report."""

import json

from tests.integration.cli.test_init_live import write_live_report


def test_followup_report_redacts_both_provider_keys(tmp_path, monkeypatch):
    dashscope, qwen = 'synthetic-dashscope-"secret', "synthetic-qwen-secret"
    monkeypatch.setenv("DASHSCOPE_API_KEY", dashscope)
    monkeypatch.setenv("QWEN_API_KEY", qwen)
    report = {"error": f"Failed with {dashscope}", "actions": [{"output": qwen}]}
    path = tmp_path / "report.json"
    write_live_report(path, report)
    first = json.loads(path.read_text())
    assert first == {"error": "Failed with [REDACTED]", "actions": [{"output": "[REDACTED]"}]}
    report["second_scope"] = {"status": "partial", "error": f"{qwen} / {dashscope}"}
    write_live_report(path, report)
    second = json.loads(path.read_text())
    assert second["error"] == first["error"] and second["actions"] == first["actions"]
    assert second["second_scope"] == {"status": "partial", "error": "[REDACTED] / [REDACTED]"}
