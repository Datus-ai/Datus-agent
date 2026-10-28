# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for deterministic SQL lineage, statement fact and comment extraction."""

import pytest

from datus.utils.sql_lineage import (
    SqlFragment,
    extract_comments,
    extract_from_fragments,
    extract_sql_from_python,
    preprocess_template,
    split_statements,
)

DB = "dw"


def run(sql: str, file: str = "etl.sql", dialect: str = "mysql", include_comments: bool = True):
    return extract_from_fragments(
        [SqlFragment(sql, file)], dialect=dialect, default_database=DB, include_comments=include_comments
    )


def edge_map(result):
    return {e.target: e for e in result.lineage}


# ---------------------------------------------------------------- lineage


@pytest.mark.parametrize(
    "sql, target, statement",
    [
        ("INSERT INTO dw.t SELECT a FROM dw.s1 JOIN dw.s2 ON s1.id = s2.id", "dw.t", "INSERT INTO"),
        ("INSERT OVERWRITE TABLE dw.t SELECT a FROM dw.s1, dw.s2", "dw.t", "INSERT OVERWRITE"),
        ("CREATE TABLE dw.t AS SELECT a FROM dw.s1 UNION ALL SELECT a FROM dw.s2", "dw.t", "CREATE TABLE"),
        ("CREATE VIEW dw.t AS SELECT a FROM dw.s1 WHERE a IN (SELECT a FROM dw.s2)", "dw.t", "CREATE VIEW"),
    ],
)
def test_write_statements_produce_edges_with_every_source(sql, target, statement):
    dialect = "hive" if "OVERWRITE" in sql else "mysql"
    result = run(sql, dialect=dialect)
    edges = edge_map(result)
    assert set(edges) == {target}
    assert set(edges[target].sources) == {"dw.s1", "dw.s2"}
    assert edges[target].statement == statement


def test_cte_names_are_not_sources_but_target_reads_are_preserved():
    sql = """
    INSERT INTO dw.t
    WITH base AS (SELECT id FROM dw.orders), agg AS (SELECT id FROM base)
    SELECT agg.id FROM agg JOIN dw.t ON agg.id = t.id
    """
    edge = edge_map(run(sql))["dw.t"]
    assert set(edge.sources) == {"dw.orders", "dw.t"}


def test_unqualified_tables_get_default_database_and_qualified_keep_theirs():
    edge = edge_map(run("INSERT INTO t SELECT a FROM s JOIN other.x ON s.id = x.id"))["dw.t"]
    assert set(edge.sources) == {"dw.s", "other.x"}


def test_plain_select_counts_as_query_without_edge():
    result = run("SELECT a FROM dw.s1 JOIN dw.s2 ON s1.id = s2.id")
    assert result.lineage == []
    assert result.queries == 1
    # Pure queries still record what they read (validated-query corpora feed query nodes).
    [fact] = result.statement_facts
    assert (fact.operation, sorted(fact.read_tables)) == ("QUERY", ["dw.s1", "dw.s2"])


def test_temp_tables_dropped_in_same_file_are_folded():
    sql = """
    CREATE TABLE dw.tmp_a AS SELECT * FROM dw.src1;
    CREATE TABLE dw.tmp_b AS SELECT * FROM dw.tmp_a JOIN dw.src2 ON tmp_a.id = src2.id;
    INSERT INTO dw.final SELECT * FROM dw.tmp_b;
    DROP TABLE dw.tmp_a;
    DROP TABLE dw.tmp_b;
    """
    edges = edge_map(run(sql))
    assert set(edges) == {"dw.final"}
    assert set(edges["dw.final"].sources) == {"dw.src1", "dw.src2"}
    assert set(edges["dw.final"].via_temp) == {"dw.tmp_a", "dw.tmp_b"}


def test_parse_failure_is_reported_and_other_statements_survive():
    sql = "INSERT INTO dw.t SELECT a FROM dw.s;\nSELECT FROM WHERE (((;\nINSERT INTO dw.u SELECT a FROM dw.s"
    result = run(sql)
    assert set(edge_map(result)) == {"dw.t", "dw.u"}
    assert [u.line for u in result.unresolved] == [2]
    assert result.statements == 3


def test_load_mode_detection():
    incremental = edge_map(run("INSERT INTO dw.t SELECT a FROM dw.s WHERE pt_date >= '${pt_date}'"))["dw.t"]
    assert incremental.load_mode == "insert"
    truncated = edge_map(run("TRUNCATE TABLE dw.t; INSERT INTO dw.t SELECT a FROM dw.s"))["dw.t"]
    assert truncated.load_mode == "truncate_reload"
    overwrite = edge_map(run("INSERT OVERWRITE TABLE dw.t SELECT a FROM dw.s", dialect="hive"))["dw.t"]
    assert overwrite.load_mode == "overwrite"
    plain = edge_map(run("INSERT INTO dw.t SELECT a FROM dw.s WHERE a = 1"))["dw.t"]
    assert plain.load_mode == "insert"


# ---------------------------------------------------------------- templating


@pytest.mark.parametrize(
    "raw, expected_fragment",
    [
        ("SELECT * FROM {{ ref('orders') }}", "FROM orders"),
        ("SELECT * FROM {{ source('raw', 'users') }}", "FROM raw.users"),
        ("SELECT * FROM t WHERE dt = '${pt_date}'", "'__tpl_pt_date__'"),
        ("SELECT * FROM t_${suffix}", "t___tpl_suffix__"),
    ],
)
def test_preprocess_template_variants(raw, expected_fragment):
    sql, templated = preprocess_template(raw)
    assert templated is True
    assert expected_fragment in sql


@pytest.mark.parametrize(
    "header",
    [
        "{{ config(materialized='table') }}",
        "{{ config(materialized='incremental', unique_key='id') }}",
        "{{\n  config(\n    materialized='table',\n    tags=['daily']\n  )\n}}",
        "{{- config(materialized='view') -}}",
    ],
)
def test_dbt_config_header_does_not_break_the_model(header):
    sql = f"{header}\nSELECT o.id FROM {{{{ ref('orders') }}}} o\nJOIN {{{{ ref('users') }}}} u\nON o.uid = u.id"
    result = extract_from_fragments([SqlFragment(sql, "m.sql")], dialect="snowflake", default_database=DB)
    assert result.unresolved == []
    [fact] = result.statement_facts
    assert set(fact.read_tables) == {"dw.orders", "dw.users"}
    assert fact.line == header.count("\n") + 2


@pytest.mark.parametrize(
    "ref",
    [
        "ref('orders')",
        "ref('shop', 'orders')",
        'ref("shop", "orders")',
        "ref('orders', v=2)",
        "ref('shop','orders', version=3)",
    ],
)
def test_dbt_ref_variants_resolve_to_the_model_name(ref):
    result = run(f"SELECT * FROM {{{{ {ref} }}}}", dialect="snowflake")
    [fact] = result.statement_facts
    assert fact.read_tables == ["dw.orders"]


def test_preprocess_keeps_line_numbers_across_jinja_blocks():
    raw = "{% if x %}\n\n{% endif %}\nSELECT 1"
    sql, _ = preprocess_template(raw)
    assert sql.count("\n") == raw.count("\n")


def test_split_ignores_semicolons_in_strings_and_reports_lines():
    sql = "-- header\nSELECT ';' AS x;\n\nSELECT 2"
    pieces = split_statements(sql, "mysql")
    assert [line for _, line in pieces] == [1, 3]


def test_python_sources_contribute_only_sql_strings():
    source = 'x = """not sql"""\nq = """\nINSERT INTO dw.t SELECT a FROM dw.s\n"""\n'
    fragments = extract_sql_from_python(source, "dag.py")
    assert len(fragments) == 1
    result = extract_from_fragments(fragments, dialect="mysql", default_database=DB)
    assert edge_map(result)["dw.t"].line == 3


@pytest.mark.parametrize(
    "literal",
    [
        '"""Create the connection."""',
        '"""Select the rows from the cache."""',
        '"""Update the settings for a user."""',
        'f"""SELECT * FROM {table}"""',
        'rf"""INSERT INTO {target} SELECT * FROM s"""',
        'F"""\nCREATE TABLE {name} AS SELECT 1 FROM s\n"""',
    ],
)
def test_python_prose_and_runtime_formatted_strings_are_not_sql(literal):
    source = f"def run():\n    {literal}\n    q = '''INSERT INTO dw.t SELECT a FROM dw.s'''\n"
    assert [f.text for f in extract_sql_from_python(source, "dag.py")] == ["INSERT INTO dw.t SELECT a FROM dw.s"]


@pytest.mark.parametrize(
    "prose", ["Create the connection.", "Update the settings.", "Delete stale files.", "With care, merge the rows."]
)
def test_python_prose_strings_outside_docstrings_are_not_sql(prose):
    source = f'HELP = """{prose}"""\nq = """\nINSERT INTO dw.t SELECT a FROM dw.s\n"""\n'
    assert len(extract_sql_from_python(source, "dag.py")) == 1


def test_python_docstring_detection_tolerates_unparsable_source():
    source = 'print "py2"\nq = """SELECT a FROM dw.s"""\n'
    assert [f.text for f in extract_sql_from_python(source, "dag.py")] == ["SELECT a FROM dw.s"]


@pytest.mark.parametrize(
    "sql",
    [
        "USE dw",
        "ALTER TABLE dw.t1 ADD COLUMN c INT",
        "ALTER TABLE t1 ADD PARTITION (dt = '2024-01-01')",
        "ANALYZE TABLE t2",
        "SET x = 1",
    ],
)
def test_non_data_statements_read_no_tables_and_are_not_queries(sql):
    result = run(sql, dialect="hive" if "PARTITION" in sql else "mysql")
    assert all(f.read_tables == [] and f.operation != "QUERY" for f in result.statement_facts)
    assert result.lineage == []
    assert result.queries == 0


# ---------------------------------------------------------------- comments


def test_comments_attach_to_code_and_ignore_dashes_in_strings():
    sql = "SELECT 'a--b' AS x, -- label for x\n  -- next line note\n  y\nFROM t"
    comments = extract_comments(sql, "f.sql")
    assert [(c.text, c.code, c.line) for c in comments] == [
        ("label for x", "SELECT 'a--b' AS x,", 1),
        ("next line note", "y", 2),
    ]


def test_block_comment_spanning_lines_keeps_following_line_numbers():
    sql = "SELECT 0;\n/* first\n   second */\nSELECT 1 -- after\n"
    comments = extract_comments(sql, "f.sql")
    assert [(c.text, c.line) for c in comments] == [("first second", 2), ("after", 4)]


@pytest.mark.parametrize(
    "text, kind",
    [
        ("-- 一个案件只计算一次", "note"),
        ("-- keep only the latest record", "note"),
        ("-- and status = 1", "code"),
        ("-- case when t in ('手持机') then 1 end", "code"),
        ("-- insert into dw.t", "code"),
    ],
)
def test_comment_kind_classification(text, kind):
    [comment] = extract_comments("SELECT 0\n" + text + "\nSELECT 1", "f.sql")
    assert comment.kind == kind


def test_file_header_is_kept_whole_with_its_line_structure():
    sql = (
        "/* =====\n * [Question]\n * count paid users\n *\n * [Knowledge]\n"
        " * paid: pay_amt > 0 -- in cents\n * ===== */\n-- second header line\nSELECT 1 -- trailing\n"
    )
    header, trailing = extract_comments(sql, "q.sql")
    assert header.kind == "header"
    assert header.line == 1
    assert header.text == (
        "[Question]\ncount paid users\n[Knowledge]\npaid: pay_amt > 0 -- in cents\nsecond header line"
    )
    assert (trailing.kind, trailing.text) == ("note", "trailing")


def test_long_file_header_is_not_clipped_to_the_note_limit():
    body = "\n".join(f"-- rule {i}: status = {i}" for i in range(60))
    [header] = extract_comments(body + "\nSELECT 1", "q.sql")
    assert header.kind == "header"
    assert header.text.count("\n") == 59


def test_decoration_and_punctuation_only_comments_are_dropped():
    assert extract_comments("-- =========\n-- ,\nSELECT 1", "f.sql") == []


def test_comments_carry_the_target_of_their_statement():
    sql = "-- header\nINSERT INTO dw.t SELECT a FROM dw.s;\n-- about u\nINSERT INTO dw.u SELECT a FROM dw.s"
    result = run(sql)
    assert [(c.text, c.statement_target) for c in result.comments] == [
        ("header", "dw.t"),
        ("about u", "dw.u"),
    ]


def test_comments_can_be_skipped_without_changing_lineage():
    sql = "-- note\nINSERT INTO dw.t SELECT a FROM dw.s JOIN dw.u ON s.id = u.id WHERE s.f = 1"
    result = run(sql, include_comments=False)
    assert result.comments == []
    assert [(e.target, sorted(e.sources)) for e in result.lineage] == [("dw.t", ["dw.s", "dw.u"])]


@pytest.mark.parametrize("prefix", ["DROP TABLE IF EXISTS dw.t;", ""])
def test_rebuilt_table_keeps_lineage(prefix):
    """A cleanup before CTAS must not erase the final published table."""
    result = run(prefix + "CREATE TABLE dw.t AS SELECT * FROM dw.src")
    assert edge_map(result)["dw.t"].sources == ["dw.src"]
    assert edge_map(result)["dw.t"].load_mode == "create"


def test_cte_shadow_preserves_outer_physical_source():
    """Nested CTE names must not hide physical tables in outer scopes."""
    result = run(
        "INSERT INTO dw.out SELECT * FROM orders WHERE id IN "
        "(WITH orders AS (SELECT id FROM archive) SELECT id FROM orders)"
    )
    assert set(edge_map(result)["dw.out"].sources) == {"dw.orders", "dw.archive"}


def test_repeated_table_lifecycles_keep_distinct_statement_sources():
    """Reusing a scratch name must not mix sources across its lifetimes."""
    result = run(
        "CREATE TABLE tmp AS SELECT * FROM a; INSERT INTO x SELECT * FROM tmp; "
        "DROP TABLE tmp; CREATE TABLE tmp AS SELECT * FROM b; INSERT INTO y SELECT * FROM tmp;"
    )
    facts = [s for s in result.statement_facts if s.write_tables == ["dw.tmp"] and s.operation.startswith("CREATE")]
    assert [s.read_tables for s in facts] == [["dw.a"], ["dw.b"]]
    assert edge_map(result)["dw.tmp"].sources == ["dw.b"]


def test_truncate_after_insert_does_not_relabel_previous_write():
    """Only a preceding truncate can justify a reload classification."""
    edge = edge_map(run("INSERT INTO t SELECT * FROM s; TRUNCATE TABLE t"))["dw.t"]
    assert edge.load_mode == "insert"


def test_cte_before_insert_resolves_to_physical_table():
    """A WITH clause attached to INSERT remains available to source resolution."""
    result = run("WITH x AS (SELECT * FROM a) INSERT INTO b SELECT * FROM x")
    assert edge_map(result)["dw.b"].sources == ["dw.a"]


def test_temp_folding_preserves_raw_statement_graph():
    """A compact graph must still expose the original intermediate dependencies."""
    result = run("CREATE TABLE tmp AS SELECT * FROM a; INSERT INTO b SELECT * FROM tmp; DROP TABLE tmp")
    assert edge_map(result)["dw.b"].sources == ["dw.a"]
    assert [(e.target, e.sources) for e in result.raw_lineage] == [("dw.tmp", ["dw.a"]), ("dw.b", ["dw.tmp"])]


def test_statement_evidence_starts_at_code_after_block_header():
    """A multiline business header must not move every SQL reference to line one."""
    result = run("/* question\n knowledge */\n-- note\nSELECT * FROM orders")
    assert result.statement_facts[0].line == 4
    assert result.statement_facts[0].read_tables == ["dw.orders"]


def test_comment_only_file_does_not_claim_a_statement():
    """Documentation-only SQL files contribute comments without phantom parsed statements."""
    result = run("/* knowledge */\n-- more knowledge")
    assert result.statements == result.parsed == 0
    assert result.comments[0].text == "knowledge\nmore knowledge"


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE dst SET x=c.x FROM c WHERE dst.id=c.id",
        "DELETE FROM dst USING c WHERE dst.id=c.id",
        "MERGE INTO dst USING c ON dst.id=c.id WHEN MATCHED THEN UPDATE SET x=c.x",
    ],
)
def test_dml_cte_chain_resolves_only_physical_sources(statement):
    """DML CTE names must not become physical tables, even through nested chains."""
    result = run(
        "WITH a AS (SELECT * FROM src), b AS (SELECT * FROM a), c AS (SELECT * FROM b) " + statement,
        dialect="postgres",
    )
    assert result.unresolved == []
    assert result.statement_facts[0].read_tables == ["dw.src"]
    assert edge_map(result)["dw.dst"].sources == ["dw.src"]


def test_dml_cte_does_not_hide_qualified_physical_table():
    """An explicit database-qualified source cannot resolve to a same-named CTE."""
    result = run(
        "WITH c AS (SELECT * FROM src) UPDATE dst SET x=p.x FROM archive.c p WHERE dst.id=p.id",
        dialect="postgres",
    )
    assert set(result.statement_facts[0].read_tables) == {"dw.src", "archive.c"}


@pytest.mark.parametrize(
    "template", ["{{\n ref('a')\n}}", "{{ source(\n'db',\n'a') }}", "{{\n name\n}}", "${\nname\n}"]
)
def test_multiline_template_preserves_subsequent_source_lines(template):
    sql = f"SELECT * FROM {template} a\nJOIN b\nON a.id = b.id;\nSELECT * FROM c"
    processed, templated = preprocess_template(sql)
    assert templated
    assert processed.count("\n") == sql.count("\n")
    second = run(sql).statement_facts[1]
    assert (second.read_tables, second.line) == (["dw.c"], sql.count("\n") + 1)
