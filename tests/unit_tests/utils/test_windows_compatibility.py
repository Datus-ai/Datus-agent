from types import SimpleNamespace
from unittest.mock import patch

import pytest

from datus import entrypoints
from datus.utils.constants import DBType
from datus.utils.multiprocessing_utils import configure_multiprocessing_start_method
from datus.utils.path_utils import get_files_from_glob_pattern


def test_multiprocessing_start_method_defaults_to_spawn():
    with patch("multiprocessing.get_start_method", return_value=None):
        with patch("multiprocessing.set_start_method") as mock_set:
            assert configure_multiprocessing_start_method() == "spawn"
            mock_set.assert_called_once_with("spawn")


def test_multiprocessing_start_method_preserves_host_choice():
    with patch("multiprocessing.get_start_method", return_value="fork"):
        with patch("multiprocessing.set_start_method") as mock_set:
            assert configure_multiprocessing_start_method() == "fork"
            mock_set.assert_not_called()


def test_multiprocessing_start_method_handles_selection_race():
    with patch("multiprocessing.get_start_method", side_effect=[None, "forkserver"]):
        with patch("multiprocessing.set_start_method", side_effect=RuntimeError("already set")):
            assert configure_multiprocessing_start_method() == "forkserver"


@pytest.mark.parametrize(
    ("entrypoint", "module_name"),
    [
        (entrypoints.agent_main, "datus.main"),
        (entrypoints.cli_main, "datus.cli.main"),
        (entrypoints.api_main, "datus.api.main"),
        (entrypoints.mcp_main, "datus.mcp_server"),
        (entrypoints.gateway_main, "datus.gateway.main"),
    ],
)
def test_console_entrypoint_configures_before_import(entrypoint, module_name):
    events = []
    module = SimpleNamespace(main=lambda: events.append("run") or 17)

    with (
        patch.object(
            entrypoints, "configure_multiprocessing_start_method", side_effect=lambda: events.append("configure")
        ),
        patch.object(entrypoints, "import_module", side_effect=lambda name: events.append(f"import:{name}") or module),
    ):
        assert entrypoint() == 17

    assert events == ["configure", f"import:{module_name}", "run"]


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

    # Glob against the temp tree (not "~/...") so the test is hermetic and never
    # depends on a real BIRD dataset under the developer's home directory.
    pattern = str(tmp_path / "benchmark/bird/dev_20240627/dev_databases/**/*.sqlite")
    results = get_files_from_glob_pattern(pattern, DBType.SQLITE)

    toxicology_files = [r for r in results if r["name"] == "toxicology" and r["uri"].endswith("toxicology.sqlite")]

    assert len(toxicology_files) == 1, "1 toxicology database should be detected"

    assert toxicology_files[0]["name"] == "toxicology"
    # For a wildcard directory pattern the datasource is the parent directory name,
    # which is "medical" for medical/toxicology.sqlite.
    assert toxicology_files[0]["datasource"] == "medical"
