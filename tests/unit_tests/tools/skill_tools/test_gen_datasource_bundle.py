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
    assert (SKILL_DIR / "references" / "design-from-scratch.md").is_file()


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
        # Ideographs plus the CJK punctuation and fullwidth-forms blocks - a stray fullwidth comma
        # or colon is just as much a leftover as a Han character, and the narrower range misses it.
        cjk = [
            ch
            for ch in text
            if "\u4e00" <= ch <= "\u9fff"  # CJK unified ideographs
            or "\u3000" <= ch <= "\u303f"  # CJK symbols and punctuation
            or "\uff00" <= ch <= "\uffef"  # halfwidth and fullwidth forms
            or "\u3400" <= ch <= "\u4dbf"  # extension A
        ]
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


@pytest.mark.acceptance
def test_every_profile_key_the_engine_reads_is_documented():
    """An undocumented knob is worse than a missing one.

    A measured production run spent ten minutes disassembling ddl_engine.pyc to work out what
    `refund_rate` did, because it was a live profile key with zero mentions in either document.
    """
    import re

    engine = (SKILL_DIR / "scripts" / "ddl_engine.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'self\.profile\.get\(\s*"([a-z_]+)"', engine))
    keys |= set(re.findall(r'self\.profile\[\s*"([a-z_]+)"\s*\]', engine))
    assert keys, "the scrape found nothing - it has drifted from the engine"

    docs = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    docs += (SKILL_DIR / "references" / "profile-spec.md").read_text(encoding="utf-8")

    undocumented = sorted(k for k in keys if f'"{k}"' not in docs and f"`{k}`" not in docs)

    assert not undocumented, (
        f"profile keys the engine reads but neither document mentions: {undocumented}. "
        f"Add them to references/profile-spec.md, or stop reading them."
    )


@pytest.mark.acceptance
def test_readme_template_is_lean():
    """The README duplicated what the database already carries as comments.

    The engine writes COMMENT ON and import_database_file copies it across, so a per-column table
    in the README goes stale the moment the profile changes. A production run produced a README
    with boilerplate Connect and Regenerate sections on top of that.
    """
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    start = text.index("### data/README.md template")
    template = text[start : text.index("> **This is not ETL test data.**")]

    assert "150" in template, "the template has to state its line ceiling"
    assert template.count("\n") < 70, "the template itself must stay small"
    for boilerplate in ("## Connect", "## Regenerate"):
        assert boilerplate not in template, f"{boilerplate} is inferable and must not be templated"
    assert "describe_table" in template, "point the reader at the comments instead of repeating them"


def _engine_ast():
    import ast

    return ast.parse((SKILL_DIR / "scripts" / "ddl_engine.py").read_text(encoding="utf-8"))


def _docs() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8") + (
        SKILL_DIR / "references" / "profile-spec.md"
    ).read_text(encoding="utf-8")


@pytest.mark.acceptance
def test_every_constructor_argument_is_documented():
    """Profile keys are not the only knobs, and the scrape that only checked them missed the rest.

    A measured production run needed to pin the last day of the data, found `end_date` in neither
    document, opened `ddl_engine.py` to look for it - and spent the remaining thirty turns inside
    the engine instead of generating anything. A constructor argument the documents never name is
    exactly as expensive as an undocumented profile key.
    """
    import ast

    engine = next(
        node for node in ast.walk(_engine_ast()) if isinstance(node, ast.ClassDef) and node.name == "DDLEngine"
    )
    init = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    args = [a.arg for a in init.args.args if a.arg != "self"]
    assert args, "the scrape found no arguments - it has drifted from the engine"

    docs = _docs()
    undocumented = sorted(a for a in args if f"`{a}`" not in docs)

    assert not undocumented, (
        f"DDLEngine arguments neither document names: {undocumented}. "
        f"Add them to references/profile-spec.md section 1.2, or drop them from the signature."
    )


@pytest.mark.acceptance
def test_every_dimension_kind_is_documented():
    """`dim_kinds` is a documented profile key whose *values* come from a constant in genlib.

    Naming the key without naming its domain still forces a trip into the source, which is the
    trip these tests exist to prevent.
    """
    import ast

    genlib = ast.parse((SKILL_DIR / "scripts" / "genlib.py").read_text(encoding="utf-8"))
    density = next(
        node.value
        for node in genlib.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DIM_DENSITY" for t in node.targets)
    )
    kinds = [k.value for k in density.keys]
    assert kinds, "the scrape found no dimension kinds - it has drifted from genlib"

    docs = _docs()
    undocumented = sorted(k for k in kinds if f"`{k}`" not in docs)

    assert not undocumented, f"dimension kinds `dim_kinds` accepts but no document names: {undocumented}"


@pytest.mark.acceptance
def test_skill_md_does_not_carry_the_row_allocation_worksheet():
    """The layer-share table is a design aid for inventing a schema, not for judging one.

    It sat in SKILL.md under a heading that said "On Path A you do not compute this", and a measured
    production run computed it anyway - the percentages are a worksheet whatever the sentence above
    them says. It lives in `design-from-scratch.md` now, which a DDL-driven run never opens.
    """
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    design = (SKILL_DIR / "references" / "design-from-scratch.md").read_text(encoding="utf-8")

    for share in ("| Main fact | 10-15% |", "| Event stream |", "| All dimensions |"):
        assert share not in skill, f"{share} is a worksheet; it belongs in design-from-scratch.md"
        assert share in design, f"{share} was dropped rather than moved"


@pytest.mark.acceptance
def test_no_reference_points_at_a_section_that_moved_out():
    """A cross-reference naming the wrong file is worse than no cross-reference.

    Sections 1.2 / 1.3 / 2.4 and phases 0, 1, 2 and 4 moved; anything still calling them SKILL.md
    sections sends the reader to a file that no longer has them.
    """
    import re

    for name in ("SKILL.md", "references/profile-spec.md", "references/pitfalls.md"):
        text = (SKILL_DIR / name).read_text(encoding="utf-8")
        stale = re.findall(r"Phase [0124]\b[^\n]{0,40}", text)
        assert not stale, f"{name} still points at a phase that moved: {stale}"
