from pathlib import Path
from unittest.mock import patch

import pytest

from datus.configuration.agent_config_loader import load_agent_config
from datus.utils.constants import DBType
from datus.utils.path_utils import get_files_from_glob_pattern


# test datus.models.base
@pytest.mark.parametrize(
    "platform_name, expected_method",
    [
        ("Windows", "spawn"),  # Windows
        ("Linux", "fork"),  # not Windows
        ("Darwin", "fork"),  # macOS
    ],
)
def test_multiprocessing_start_method_base(platform_name, expected_method):
    with patch("platform.system", return_value=platform_name):
        with patch("multiprocessing.set_start_method") as mock_set:
            import importlib

            import datus.models.base

            importlib.reload(datus.models.base)

            mock_set.assert_called_once_with(expected_method, force=True)


# test datus.storage.embedding_models
@pytest.mark.parametrize(
    "platform_name, expected_method",
    [
        ("Windows", "spawn"),
        ("Linux", "fork"),
        ("Darwin", "fork"),
    ],
)
def test_multiprocessing_start_method_embedding(platform_name, expected_method):
    with patch("platform.system", return_value=platform_name):
        with patch("multiprocessing.set_start_method") as mock_set:
            import importlib

            import datus.storage.embedding_models

            importlib.reload(datus.storage.embedding_models)

            mock_set.assert_called_once_with(expected_method, force=True)


def test_detect_toxicology_db(tmp_path):
    test_files = [
        "benchmark/bird/dev_20240627/dev_databases/medical/toxicology.sqlite",
        "benchmark/bird/dev_20240627/dev_databases/chemical/untested.sqlite",
        "benchmark/bird/dev_20240627/dev_databases/empty.sqlite",
    ]

    for file in test_files:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    pattern = "benchmark/bird/dev_20240627/dev_databases/**/*.sqlite"
    full_pattern = str(tmp_path / pattern)
    results = get_files_from_glob_pattern(full_pattern, DBType.SQLITE)

    toxicology_files = [r for r in results if r["name"] == "toxicology" and r["uri"].endswith("toxicology.sqlite")]

    assert len(toxicology_files) == 1, "1 toxicology database should be detected"

    expected_uri = (
        f"{DBType.SQLITE}:///" f"{tmp_path}/benchmark/bird/dev_20240627/dev_databases/medical/toxicology.sqlite"
    ).replace("\\", "/")

    assert toxicology_files[0]["uri"] == expected_uri


def test_load_agent_config_utf8_with_real_args():
    cfg = load_agent_config(
        config=str(Path(__file__).with_suffix("").parent.parent / "conf" / "agent.yml.qs"),
        debug=False,
        save_llm_trace=True,
        action="benchmark",
        benchmark="bird_dev",
        benchmark_task_ids=["0"],
        namespace="bird_sqlite",
    )
    assert cfg is not None
