# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for deterministic SQL lineage / join / rule / comment extraction."""

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


def run(sql: str, file: str = "etl.sql", dialect: str = "mysql", sections=None):
    return extract_from_fragments([SqlFragment(sql, file)], dialect=dialect, default_database=DB, sections=sections)


def edge_map(result):
    return {e.target: e for e in result.lineage}


def join_set(result):
    return {(j.left_table, j.right_table, tuple(j.keys)) for j in result.joins}


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


def test_cte_names_and_target_are_not_sources():
    sql = """
    INSERT INTO dw.t
    WITH base AS (SELECT id FROM dw.orders), agg AS (SELECT id FROM base)
    SELECT agg.id FROM agg JOIN dw.t ON agg.id = t.id
    """
    edge = edge_map(run(sql))["dw.t"]
    assert edge.sources == ["dw.orders"]


def test_unqualified_tables_get_default_database_and_qualified_keep_theirs():
    edge = edge_map(run("INSERT INTO t SELECT a FROM s JOIN other.x ON s.id = x.id"))["dw.t"]
    assert set(edge.sources) == {"dw.s", "other.x"}


def test_plain_select_counts_as_query_without_edge():
    result = run("SELECT a FROM dw.s1 JOIN dw.s2 ON s1.id = s2.id")
    assert result.lineage == []
    assert result.queries == 1
    # Joins are still collected from pure queries (validated-query corpora feed relationships).
    assert join_set(result) == {("dw.s1", "dw.s2", (("id", "id"),))}


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
    assert incremental.load_mode == "incremental"
    assert incremental.window == ["pt_date >= '${pt_date}'"]
    truncated = edge_map(run("TRUNCATE TABLE dw.t; INSERT INTO dw.t SELECT a FROM dw.s"))["dw.t"]
    assert truncated.load_mode == "truncate_reload"
    overwrite = edge_map(run("INSERT OVERWRITE TABLE dw.t SELECT a FROM dw.s", dialect="hive"))["dw.t"]
    assert overwrite.load_mode == "overwrite"
    plain = edge_map(run("INSERT INTO dw.t SELECT a FROM dw.s WHERE a = 1"))["dw.t"]
    assert plain.load_mode == "insert"


@pytest.mark.parametrize("sections", [[], ["joins"], ["comments"], ["rules"], None])
def test_lineage_does_not_depend_on_requested_sections(sections):
    sql = (
        "INSERT INTO dw.t SELECT s.a FROM (SELECT a, month FROM dw.s) s "
        "LEFT JOIN dw.d d ON s.a = d.a WHERE s.month >= '${month}'"
    )
    edge = edge_map(run(sql, sections=sections))["dw.t"]
    assert edge.load_mode == "incremental"
    assert edge.window == ["month >= '${month}'"]


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


# ---------------------------------------------------------------- joins


def test_join_keys_resolve_through_nested_derived_tables_and_ctes():
    sql = """
    WITH c AS (SELECT user_id AS uid FROM dw.users)
    SELECT * FROM (SELECT o.uid2 FROM (SELECT customer_id AS uid2 FROM dw.orders) o) x
    JOIN c ON x.uid2 = c.uid
    """
    assert join_set(run(sql)) == {("dw.orders", "dw.users", (("customer_id", "user_id"),))}


def test_composite_key_stays_one_edge_and_order_is_canonical():
    a = run("SELECT * FROM dw.b JOIN dw.a ON b.x = a.x AND b.y = a.z")
    b = run("SELECT * FROM dw.a JOIN dw.b ON a.z = b.y AND a.x = b.x")
    assert join_set(a) == join_set(b) == {("dw.a", "dw.b", (("x", "x"), ("z", "y")))}


def test_transform_is_recorded_for_a_derived_key():
    sql = """
    SELECT * FROM (SELECT LEFT(issue_id, 6) AS store_code FROM dw.issues) i
    JOIN dw.stores s ON i.store_code = s.store_code
    """
    [join] = run(sql).joins
    assert join.keys == [("issue_id", "store_code")]
    assert join.transforms == {"dw.issues.issue_id": "LEFT(issue_id, 6)"}


def test_union_branches_become_alternatives_not_a_composite_key():
    sql = """
    SELECT * FROM (SELECT created_at AS d FROM dw.w UNION ALL SELECT finished_at AS d FROM dw.w) u
    JOIN dw.cal c ON u.d = c.d
    """
    [join] = run(sql).joins
    assert join.keys == [("d", "created_at|finished_at")]


def test_or_branch_and_constant_equalities_are_not_join_keys():
    result = run("SELECT * FROM dw.a JOIN dw.b ON a.id = b.id OR a.k = b.k WHERE a.flag = 1")
    assert result.joins == []


def test_unqualified_column_with_several_sources_is_counted_unresolved():
    result = run("SELECT * FROM dw.a JOIN dw.b ON id = b.id")
    assert result.joins == []
    assert result.join_predicates_unresolved == 1


def test_using_clause_is_a_join_key():
    assert join_set(run("SELECT * FROM dw.a JOIN dw.b USING (id)")) == {("dw.a", "dw.b", (("id", "id"),))}


# ---------------------------------------------------------------- rules


def predicates(result):
    return {(p.table, p.column, p.predicate, p.clause) for p in result.predicates}


@pytest.mark.parametrize(
    "condition, predicate",
    [
        ("s.status = 1", "= 1"),
        ("s.status <> 1", "!= 1"),
        ("NOT s.status = 1", "!= 1"),
        ("s.brand NOT IN ('b', 'a')", "NOT IN ('a', 'b')"),
        ("s.name LIKE '%x%'", "LIKE '%x%'"),
        ("s.closed_at IS NOT NULL", "IS NOT NULL"),
        ("s.amount BETWEEN 1 AND 5", "BETWEEN 1 AND 5"),
        ("1 < s.amount", "> 1"),
    ],
)
def test_constant_predicate_shapes(condition, predicate):
    result = run(f"SELECT * FROM dw.s s WHERE {condition}", sections=["rules"])
    assert predicates(result) == {("dw.s", condition.split(".")[1].split()[0], predicate, "WHERE")}


def test_filter_in_subquery_is_attributed_to_the_physical_column():
    sql = "SELECT * FROM (SELECT st AS status_code FROM dw.orders) o WHERE o.status_code = 'PAID'"
    assert predicates(run(sql, sections=["rules"])) == {("dw.orders", "st", "= 'PAID'", "WHERE")}


def test_or_under_where_is_not_a_filter_but_or_under_case_is():
    sql = """
    SELECT CASE WHEN s.type = 'A' OR s.day <= '2024-12-07' THEN 1 END AS f
    FROM dw.s s WHERE s.x = 1 OR s.y = 2
    """
    assert predicates(run(sql, sections=["rules"])) == {
        ("dw.s", "type", "= 'A'", "CASE"),
        ("dw.s", "day", "<= '2024-12-07'", "CASE"),
    }


def test_template_predicates_feed_the_window_not_the_filters():
    result = run("INSERT INTO dw.t SELECT a FROM dw.s WHERE pt_date >= '${pt_date}'", sections=["rules"])
    assert result.predicates == []
    assert edge_map(result)["dw.t"].window == ["pt_date >= '${pt_date}'"]


def test_value_mappings_from_searched_case_simple_case_and_if():
    sql = """
    SELECT
      CASE WHEN o.t = 1 THEN 'MA' WHEN o.t = 2 THEN 'IMAC' END AS type_name,
      CASE o.s WHEN '0' THEN 'open' END AS state_name,
      IF(o.f = 1, 'yes', 'no') AS flag_name
    FROM dw.orders o
    """
    mappings = {(m.column, m.value, m.label) for m in run(sql, sections=["rules"]).mappings}
    assert mappings == {("t", "1", "MA"), ("t", "2", "IMAC"), ("s", "0", "open"), ("f", "1", "yes")}


def test_row_number_partition_resolves_to_physical_columns_with_transform():
    sql = """
    SELECT * FROM (
      SELECT store_code, row_number() over (partition by store_code, date_format(pt_date, '%Y-%m')
                                            order by pt_date desc) AS rn
      FROM dw.store_snapshot
    ) x WHERE rn = 1
    """
    [dedup] = run(sql, sections=["rules"]).dedups
    assert dedup.table == "dw.store_snapshot"
    assert dedup.partition_by == ["store_code", "DATE_FORMAT(pt_date, '%Y-%m')"]
    assert dedup.order_by == ["pt_date DESC"]


def test_sql_comments_do_not_leak_into_predicate_text():
    sql = "SELECT * FROM dw.s WHERE s.flag = 0 /* known bug */"
    assert predicates(run(sql, sections=["rules"])) == {("dw.s", "flag", "= 0", "WHERE")}


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
    result = run(sql, sections=["comments"])
    assert [(c.text, c.statement_target) for c in result.comments] == [
        ("header", "dw.t"),
        ("about u", "dw.u"),
    ]


def test_sections_limit_what_is_collected():
    sql = "-- note\nINSERT INTO dw.t SELECT a FROM dw.s JOIN dw.u ON s.id = u.id WHERE s.f = 1"
    result = run(sql, sections=[])
    assert (result.joins, result.predicates, result.comments) == ([], [], [])
    assert len(result.lineage) == 1
