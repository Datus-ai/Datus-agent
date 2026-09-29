"""Benchmark scripts should read datasets from agent.home, outside the checkout."""

import json
from types import SimpleNamespace

import pytest

from benchmark.scripts.gen_benchmark import get_benchmark_file_path
from benchmark.scripts.schema_recall_spider2 import load_gold_tables, spider2_question_file

pytestmark = pytest.mark.ci


def test_spider2_script_paths_follow_agent_home(tmp_path):
    spider2_dir = tmp_path / "benchmark" / "spider2"
    snow_dir = spider2_dir / "spider2-snow"
    gold_tables_file = spider2_dir / "methods" / "gold-tables" / "spider2-snow-gold-tables.jsonl"
    gold_tables_file.parent.mkdir(parents=True)
    gold_tables_file.write_text(json.dumps({"instance_id": "sf_test", "gold_tables": ["DB.TABLE"]}) + "\n")
    config = SimpleNamespace(benchmark_path=lambda name: str(snow_dir))

    assert spider2_question_file(config) == snow_dir / "spider2-snow.jsonl"
    assert load_gold_tables(config) == {"sf_test": {"DB.TABLE"}}
    assert get_benchmark_file_path({"agent": {"home": str(tmp_path)}}, "spider2", "unused") == str(
        snow_dir / "spider2-snow.jsonl"
    )
