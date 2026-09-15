# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""The gen-datasource skill ships executable helpers, which packaging can silently drop.

``[tool.setuptools.package-data]`` lists suffixes, and a skill's ``scripts/`` directory is
data rather than an importable package, so without an explicit ``*.py`` entry the wheel
carries SKILL.md and no engine - and the failure only shows up at run time, in a customer's
install. These tests pin the bundle's shape and keep the pyproject entry honest.
"""

import tomllib
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[4] / "datus" / "resources" / "skills" / "gen-datasource"
REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.acceptance
def test_bundle_has_engine_and_references():
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert (SKILL_DIR / "scripts" / "ddl_engine.py").is_file()
    assert (SKILL_DIR / "scripts" / "genlib.py").is_file()
    assert (SKILL_DIR / "references" / "profile-spec.md").is_file()
    assert (SKILL_DIR / "references" / "pitfalls.md").is_file()


@pytest.mark.acceptance
def test_package_data_covers_skill_scripts():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["datus"]

    assert "resources/skills/*/scripts/*.py" in patterns, (
        "Without this pattern the wheel ships SKILL.md and no engine: a skill's scripts/ "
        "directory is package data, and the default suffix list has no *.py."
    )
    assert "resources/skills/*/references/*.md" in patterns


@pytest.mark.acceptance
def test_skill_frontmatter_follows_builtin_conventions():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = text.split("---\n", 2)[1]

    for key in ("name:", "description:", "version:", "tags:", "user_invocable:"):
        assert key in front, f"built-in skills declare {key}"
    assert "name: gen-datasource" in front


@pytest.mark.acceptance
def test_skill_content_is_english():
    """The skill ships inside the package, so it follows the repo's English-only rule."""
    for path in SKILL_DIR.rglob("*"):
        if path.suffix not in (".md", ".py"):
            continue
        text = path.read_text(encoding="utf-8")
        cjk = [ch for ch in text if "一" <= ch <= "鿿"]
        assert not cjk, f"{path.relative_to(SKILL_DIR)} contains CJK characters: {''.join(cjk[:20])}"


@pytest.mark.acceptance
def test_engine_imports_and_parses_ddl():
    """A syntax error or a missing helper in the bundle must fail here, not in a customer run."""
    import sys

    # The generator puts scripts/ on sys.path exactly like this; ddl_engine imports genlib from it.
    scripts = str(SKILL_DIR / "scripts")
    sys.path.insert(0, scripts)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_gd_ddl_engine", SKILL_DIR / "scripts" / "ddl_engine.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)

    engine = module.DDLEngine(
        "CREATE TABLE customers (customer_id BIGINT PRIMARY KEY, customer_name VARCHAR);"
        "CREATE TABLE orders (order_id BIGINT PRIMARY KEY, "
        "customer_id BIGINT REFERENCES customers(customer_id), "
        "order_time TIMESTAMP, paid_amount DECIMAL(18,2));",
        rows=2000,
    )

    assert set(engine.schema) == {"customers", "orders"}
    assert engine.decl_pk["orders"] == ["order_id"]
    assert engine.decl_fk[("orders", "customer_id")] == ("customers", "customer_id")
    # The declared CREATE TABLE text is what lets the built database keep its keys.
    assert "PRIMARY KEY" in engine.decl_sql["orders"]
