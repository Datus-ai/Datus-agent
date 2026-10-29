"""Check profile gating together with pytest's other collection ignore rules."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("profile", [None, "inceptor_ce_v2"])
@pytest.mark.parametrize("ignore", ["none", "path", "glob", "parent"])
def test_hs2_collection_defers_to_other_ignore_rules(tmp_path, profile, ignore):
    suite = tmp_path / "hs2"
    suite.mkdir()
    source = Path(__file__).resolve().parents[1] / "integration/tools/hs2/conftest.py"
    (suite / "conftest.py").write_bytes(source.read_bytes())
    (suite / "test_probe.py").write_text("def test_probe(): pass\n", encoding="utf-8")
    options = []
    if ignore == "path":
        options = ["--ignore=hs2/test_probe.py"]
    elif ignore == "glob":
        options = ["--ignore-glob=*probe.py"]
    elif ignore == "parent":
        (tmp_path / "conftest.py").write_text(
            "def pytest_ignore_collect(collection_path):\n"
            "    if collection_path.name == 'test_probe.py':\n"
            "        return True\n"
            "    return None\n",
            encoding="utf-8",
        )
    env = {key: value for key, value in os.environ.items() if key not in {"HS2_TEST_PROFILE", "PYTEST_ADDOPTS"}}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if profile:
        env["HS2_TEST_PROFILE"] = profile
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *options, "hs2"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    collected = bool(profile) and ignore == "none"
    assert result.returncode == (0 if collected else 5), result.stdout + result.stderr
    assert ("hs2/test_probe.py::test_probe" in result.stdout) == collected
