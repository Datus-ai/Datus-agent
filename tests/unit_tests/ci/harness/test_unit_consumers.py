from pathlib import Path

import pytest

from ci.harness.unit_consumers import audit_unit_consumers, validate_unit_consumers


def consumer_repo(root: Path, source: str, *, exports: str = "") -> dict:
    (root / "datus").mkdir()
    (root / "datus/__init__.py").write_text(exports)
    (root / "datus/example.py").write_text("VALUE = 1\n")
    (root / "tests/unit_tests/owned").mkdir(parents=True)
    (root / "tests/unit_tests/foreign").mkdir()
    (root / "tests/unit_tests/foreign/test_consumer.py").write_text(source)
    return {
        "execution": {
            "unit_rules": [
                {"id": "fallback", "paths": ["datus/"], "targets": ["tests/unit_tests/"]},
                {"id": "leaf", "paths": ["datus/example.py"], "targets": ["tests/unit_tests/owned/"]},
            ]
        }
    }


@pytest.mark.parametrize(
    "source,exports",
    [
        ("import datus.example as example\n", ""),
        ("from datus.example import VALUE\n", ""),
        ("def test_local():\n    from datus.example import VALUE\n    assert VALUE\n", ""),
        ("from datus import example\n", ""),
        ("from datus import exported\n", "from .example import VALUE as exported\n"),
        ("from datus import exported\n", "import datus.example as exported\n"),
        (
            "from datus import VALUE\n",
            "try:\n    from .example import VALUE\nexcept ImportError:\n    from math import pi as VALUE\n",
        ),
        ("from datus import *\n", "from .example import VALUE\n"),
        ("import importlib\nmodule = importlib.import_module('datus.example')\n", ""),
        ("from unittest.mock import patch\np = patch('datus.example.VALUE')\n", ""),
        ("ENTRYPOINT = 'datus.example:VALUE'\n", ""),
    ],
)
def test_audit_finds_import_and_literal_consumers(tmp_path, source, exports):
    data = consumer_repo(tmp_path, source, exports=exports)
    gap = {
        "rule": "leaf",
        "source": "datus/example.py",
        "test": "tests/unit_tests/foreign/test_consumer.py",
    }
    assert gap in audit_unit_consumers(data, tmp_path)["missing"]
    assert "datus/example.py -> tests/unit_tests/foreign/test_consumer.py" in validate_unit_consumers(data, tmp_path)[0]
    data["execution"]["unit_rules"][1]["targets"].append("tests/unit_tests/foreign/")
    assert validate_unit_consumers(data, tmp_path) == []


def test_function_target_does_not_claim_a_whole_consumer_file(tmp_path):
    data = consumer_repo(tmp_path, "from datus.example import VALUE\n")
    data["execution"]["unit_rules"][1]["targets"].append("tests/unit_tests/foreign/test_consumer.py::test_one")
    assert audit_unit_consumers(data, tmp_path)["missing"]
    data["execution"]["unit_rules"][1]["targets"].append("tests/unit_tests/foreign/test_consumer.py")
    assert validate_unit_consumers(data, tmp_path) == []


def test_audit_requires_ownership_even_without_a_direct_consumer(tmp_path):
    data = consumer_repo(tmp_path, "from datus.example import VALUE\n")
    (tmp_path / "datus/unowned.py").write_text("VALUE = 2\n")
    data["execution"]["unit_rules"] = data["execution"]["unit_rules"][1:]
    report = audit_unit_consumers(data, tmp_path)
    assert {"rule": "<unowned>", "source": "datus/unowned.py", "test": ""} in report["missing"]


def test_audit_does_not_silently_skip_unparseable_tests(tmp_path):
    data = consumer_repo(tmp_path, "def broken(\n")
    with pytest.raises(SyntaxError):
        audit_unit_consumers(data, tmp_path)


def test_reexport_cycles_terminate_without_importing_application_code(tmp_path):
    data = consumer_repo(tmp_path, "from datus import VALUE\n", exports="from .alias import VALUE\n")
    (tmp_path / "datus/alias").mkdir()
    (tmp_path / "datus/alias/__init__.py").write_text(
        "from datus import VALUE\nraise RuntimeError('must not import')\n"
    )
    assert validate_unit_consumers(data, tmp_path) == []
