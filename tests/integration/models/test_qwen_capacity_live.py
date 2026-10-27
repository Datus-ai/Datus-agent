# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Real Qwen long-input acceptance and automatic context preparation.

Opt in with DATUS_RUN_LIVE_CAPACITY=1. This deliberately sends a long
synthetic transcript, configured with a 262,144-token client budget. It does
not assert that the hosted provider itself enforces that same window.
"""

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from datus.agent.node.context_rewriter import MidTurnCompactor
from datus.cli.chat_commands import ChatCommands
from datus.configuration.agent_config import AgentConfig
from datus.models.request_capacity import ContextCapacityError
from datus.utils.path_manager import reset_path_manager
from tests.integration.cli.test_init_live import observe_live_calls
from tests.unit_tests.cli.test_chat_commands import MinimalCLI


@pytest.mark.nightly
@pytest.mark.product_e2e
@pytest.mark.timeout(900)
@pytest.mark.skipif(os.environ.get("DATUS_RUN_LIVE_CAPACITY") != "1", reason="Explicit live capacity opt-in required")
def test_qwen_long_input_and_automatic_compaction(tmp_path, monkeypatch, request):
    key_name = next((n for n in ("DASHSCOPE_API_KEY", "QWEN_API_KEY") if os.environ.get(n)), None)
    if key_name is None:
        pytest.skip("Set DASHSCOPE_API_KEY or QWEN_API_KEY")
    monkeypatch.chdir(tmp_path)
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    from agents import set_tracing_disabled

    set_tracing_disabled(True)
    reset_path_manager()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY)")
    config = AgentConfig(
        nodes={},
        home=str(tmp_path / "home"),
        project_name="synthetic-qwen-capacity",
        project_root=str(workspace),
        target="live",
        models={
            "live": {
                "type": "qwen",
                "model": "qwen3.5-27b",
                "api_key": "${" + key_name + "}",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "context_length": 262144,
                "max_tokens": 8192,
                "tokenizer_ref": "Qwen/Qwen3.5-27B",
                "enable_thinking": False,
                "max_retry": 1,
                "temperature": 0,
            }
        },
        services={
            "datasources": {"fixture": {"type": "sqlite", "uri": str(database), "name": "fixture", "default": True}}
        },
        agentic_nodes={"chat": {"model": "live", "workspace_root": str(workspace), "tools": "filesystem_tools.*"}},
        observability={"tracing": {"enabled": False}},
    )
    config.current_datasource = "fixture"
    from datus.utils.loggings import configure_logging

    configure_logging(
        level="WARNING",
        console_output=False,
        log_dir=str(tmp_path / "logs"),
        agent_config=config,
        redact={"enabled": True, "patterns": [re.escape(os.environ[key_name])]},
    )
    calls = observe_live_calls(monkeypatch)
    runner = asyncio.Runner()

    def cleanup():
        from datus.storage.registry import clear_storage_registry
        from datus.tools.db_tools.db_manager import set_db_manager_factory

        runner.close()
        set_db_manager_factory(None)
        clear_storage_registry()
        reset_path_manager()

    request.addfinalizer(cleanup)
    node = ChatCommands(MinimalCLI(config))._create_new_node()
    model = node.model
    capacity = model.request_capacity()
    instruction = "You maintain database project notes. Preserve approved decisions and markers."
    marker = "CAPACITY_OK_73A1"
    tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3.5-27B")
    corpus = "\n".join(
        f"Obsolete catalog record {i:06d}: tenant archive ledger event snapshot; not selected for core analysis."
        for i in range(22000)
    )
    encoded = tokenizer.encode(corpus).ids
    assert len(encoded) > 280000
    near_text = tokenizer.decode(encoded[:238000]) + f"\nApproved project marker: {marker}.\n"
    question = "What is the approved project marker? Reply only with that marker."
    near = [{"role": "system", "content": instruction}, {"role": "user", "content": near_text + question}]
    near_tokens = capacity.check({"messages": near}, 8192)
    assert 235000 < near_tokens < capacity.input_limit()
    report = {"model": "qwen3.5-27b", "context_length": 262144, "input_limit": capacity.input_limit(), "calls": calls}
    report_path = Path(os.environ.get("DATUS_LIVE_CAPACITY_REPORT", str(tmp_path / "report.json")))
    started = time.monotonic()
    try:
        near_response = model.generate(near, max_tokens=8192, temperature=0)
        report.update(near_input_tokens=near_tokens, near_response=near_response)
        assert marker in near_response
        history = [
            {"role": "user", "content": "Record this approved project decision and retain it for later turns."},
            {
                "role": "assistant",
                "content": f"Approved project marker: {marker}.\n" + tokenizer.decode(encoded[:280000]),
            },
            {"role": "user", "content": question},
        ]
        original_tokens = capacity.count({"instructions": instruction, "input": history, "tools": []})
        assert original_tokens > 262144
        with pytest.raises(ContextCapacityError, match="CONTEXT_LIMIT_EXCEEDED"):
            model.check_request({"messages": history, "max_tokens": 8192})
        report["original_tokens"] = original_tokens
        session = node._get_or_create_session()
        runner.run(session.add_items(history))
        compactor = MidTurnCompactor(node, system_instruction=instruction)
        compactor.set_request_context(instruction, [], 8192)
        rewritten = runner.run(compactor.rewrite_sdk_input(history))
        compacted_tokens = capacity.check({"instructions": instruction, "input": rewritten, "tools": []})
        report.update(compactions=compactor.compactions, compacted_tokens=compacted_tokens)
        assert compactor.compactions > 0
        assert compacted_tokens < original_tokens // 10
        assert runner.run(session.get_items()) == rewritten
        result = runner.run(
            model.generate_with_tools(prompt=rewritten, instruction=instruction, tools=[], max_turns=1, max_tokens=8192)
        )
        response = result["content"]
        report["compacted_response"] = response
        assert marker in response, "Automatic summarization must retain the approved project decision"
        checked = [c for c in calls if "capacity_input_tokens" in c]
        assert checked and all(c["capacity_input_tokens"] <= c["capacity_input_limit"] for c in checked)
        assert not any(c.get("error_type") for c in calls)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
