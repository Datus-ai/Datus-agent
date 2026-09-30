# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for deterministic SQL lineage / join / rule / comment extraction."""

import pytest

from datus.utils.sql_lineage import (
    TABLE_REF_MARK,
    SqlFragment,
    expression_tables,
    extract_comments,
    extract_from_fragments,
    extract_sql_from_python,
    preprocess_template,
    render_expression,
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
    assert incremental.load_mode == "insert"
    assert incremental.parameterized_predicates == ["pt_date >= '${pt_date}'"]
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
    assert edge.load_mode == "insert"
    assert edge.parameterized_predicates == ["month >= '${month}'"]


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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM dw.a JOIN dw.b USING (id) JOIN dw.c ON c.k = b.k",
        "SELECT * FROM dw.a JOIN dw.b USING (id) JOIN dw.c ON c.k = b.k JOIN dw.d ON d.k = c.k",
        "SELECT * FROM dw.a JOIN dw.b USING (id) LEFT JOIN (SELECT k FROM dw.c) c ON c.k = b.k",
    ],
)
def test_using_binds_only_to_relations_written_before_it(sql):
    result = run(sql)
    assert ("dw.a", "dw.b", (("id", "id"),)) in join_set(result)
    assert [c for c in result.conditions if c.status == "unresolved"] == []


@pytest.mark.parametrize(
    "where",
    [
        "a.x IN (SELECT id FROM dw.b)",
        "a.x NOT IN (SELECT id FROM dw.b WHERE b.flag = 1)",
        "EXISTS (SELECT 1 FROM dw.b)",
        "a.x = (SELECT MAX(id) FROM dw.b)",
        "a.x > (SELECT AVG(b.v) FROM dw.b WHERE b.k IN (SELECT k FROM dw.c))",
    ],
)
def test_subquery_predicates_are_not_unresolved_relationships(where):
    result = run(f"SELECT * FROM dw.a WHERE {where}")
    assert result.joins == []
    assert [c for c in result.conditions if c.status == "unresolved"] == []
    assert result.join_predicates_unresolved == 0


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM dw.a WHERE EXISTS (SELECT 1 FROM dw.b WHERE b.id = a.id)",
        "SELECT * FROM dw.a WHERE NOT EXISTS (SELECT 1 FROM dw.b WHERE a.id = b.id AND b.flag = 1)",
        "SELECT * FROM dw.a WHERE a.v > (SELECT AVG(v) FROM dw.b WHERE b.id = a.id)",
        "SELECT * FROM dw.a WHERE EXISTS (SELECT 1 FROM dw.x WHERE EXISTS (SELECT 1 FROM dw.b WHERE b.id = a.id))",
    ],
)
def test_correlated_subquery_equality_is_a_join_key(sql):
    result = run(sql)
    assert ("dw.a", "dw.b", (("id", "id"),)) in join_set(result)
    assert [c for c in result.conditions if c.status == "unresolved"] == []


def test_correlation_does_not_leak_into_derived_tables():
    # A derived table cannot see outer aliases, so ``a`` inside it is not the outer relation.
    result = run("SELECT * FROM dw.a JOIN (SELECT b.id FROM dw.b WHERE b.id = a.id) x ON x.id = a.id")
    assert ("dw.a", "dw.b", (("id", "id"),)) in join_set(result)
    assert len(result.joins) == 1


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM dw.orders o WHERE o.created = updated",
        "SELECT * FROM dw.orders o WHERE created = o.updated",
        "SELECT * FROM dw.orders WHERE created = updated",
        "SELECT * FROM (SELECT * FROM dw.orders) o WHERE o.created = updated",
    ],
)
def test_single_source_row_comparison_is_not_a_self_join(sql):
    result = run(sql)
    assert result.joins == []
    assert result.join_predicates == 0


@pytest.mark.parametrize(
    "sql, dialect, left, right",
    [
        (
            "MERGE INTO dw.t USING dw.s ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.v = s.v",
            "postgres",
            "dw.s",
            "dw.t",
        ),
        (
            "MERGE INTO dw.t tg USING (SELECT id AS sid, v FROM dw.s WHERE k = 1) src ON tg.id = src.sid "
            "WHEN MATCHED THEN UPDATE SET tg.v = src.v",
            "postgres",
            "dw.s",
            "dw.t",
        ),
        (
            "WITH src AS (SELECT id AS sid FROM dw.s) MERGE INTO dw.t USING src ON t.id = src.sid "
            "WHEN MATCHED THEN DELETE",
            "postgres",
            "dw.s",
            "dw.t",
        ),
        ("UPDATE dw.t JOIN dw.s ON t.id = s.id SET t.v = s.v", "mysql", "dw.s", "dw.t"),
        ("UPDATE dw.t SET v = s.v FROM dw.s WHERE t.id = s.id", "postgres", "dw.s", "dw.t"),
        ("DELETE FROM dw.t USING dw.s WHERE t.id = s.id", "postgres", "dw.s", "dw.t"),
        ("DELETE t FROM dw.t JOIN dw.s ON t.id = s.id WHERE s.x = 1", "mysql", "dw.s", "dw.t"),
    ],
)
def test_dml_match_conditions_are_join_keys(sql, dialect, left, right):
    result = run(sql, dialect=dialect)
    [join] = result.joins
    assert (join.left_table, join.right_table) == (left, right)
    assert len(join.keys) == 1
    assert join.statement_target == "dw.t"
    assert set(expression_tables(join.expression)) == {left, right}
    assert [c for c in result.conditions if c.status == "unresolved"] == []


def test_dml_where_filters_become_rules_without_changing_sources():
    sql = "UPDATE dw.t JOIN dw.s ON t.id = s.id SET t.v = s.v WHERE s.k = 1"
    result = run(sql)
    assert edge_map(result)["dw.t"].sources == ["dw.s"]
    assert [(p.table, p.column, p.predicate) for p in result.predicates] == [("dw.s", "k", "= 1")]


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
    assert edge_map(result)["dw.t"].parameterized_predicates == ["pt_date >= '${pt_date}'"]


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


@pytest.mark.parametrize("sections", [[], ["joins"], ["rules"], ["comments"]])
def test_parameterized_predicate_is_not_incremental_evidence(sections):
    """A region parameter says nothing about the loading strategy."""
    edge = edge_map(run("INSERT INTO t SELECT * FROM s WHERE region = '${region}'", sections=sections))["dw.t"]
    assert edge.load_mode == "insert"
    assert edge.parameterized_predicates == ["region = '${region}'"]


def test_truncate_after_insert_does_not_relabel_previous_write():
    """Only a preceding truncate can justify a reload classification."""
    edge = edge_map(run("INSERT INTO t SELECT * FROM s; TRUNCATE TABLE t"))["dw.t"]
    assert edge.load_mode == "insert"


@pytest.mark.parametrize("predicate", ["z.id = a.id", "a.id = z.id"])
def test_canonical_outer_join_preserves_retained_side(predicate):
    """Canonical table order must retain the original preserved table."""
    [join] = run(f"SELECT * FROM z LEFT JOIN a ON {predicate}").joins
    assert (join.left_table, join.right_table, join.join_type) == ("dw.a", "dw.z", "RIGHT")


def test_direct_join_transform_matches_derived_transform():
    """Equivalent direct and derived key expressions resolve consistently."""
    [join] = run("SELECT * FROM issues i JOIN stores s ON LEFT(i.issue_id, 6) = s.store_code").joins
    assert join.keys == [("issue_id", "store_code")]
    assert join.transforms == {"dw.issues.issue_id": "LEFT(i.issue_id, 6)"}


def test_self_join_retains_relationship():
    """Employee-to-manager relationships must survive physical table equality."""
    [join] = run("SELECT * FROM employees e LEFT JOIN employees m ON e.manager_id = m.id").joins
    assert join.keys == [("manager_id", "id")]
    assert join.aliases == ["e", "m"]


def test_unsupported_join_is_visible_in_coverage():
    """An unsupported OR join is reported, rather than silently absent."""
    result = run("SELECT * FROM a JOIN b ON a.id = b.id OR a.x = b.x")
    assert result.join_predicates == result.join_predicates_unresolved == 1
    assert result.conditions[0].status == "unresolved"


def test_negated_conjunction_does_not_invent_mandatory_filters():
    """De Morgan transformations must not turn alternatives into mandatory rules."""
    result = run("SELECT * FROM s WHERE NOT (a = 1 AND b = 2)", sections=["rules"])
    assert result.predicates == []
    assert result.conditions[0].expression == "NOT (a = 1 AND b = 2)"


def test_row_number_without_selection_is_only_a_window():
    """Numbering rows does not change the grain or deduplicate them."""
    result = run("SELECT ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn FROM events")
    assert result.dedups == []
    assert result.window_functions[0].partition_by == ["id"]
    assert result.window_functions[0].selection is None


@pytest.mark.parametrize("selection,kind", [("rn = 1", "dedup"), ("rn <= 10", "top_n")])
def test_window_selection_is_traced_through_projection(selection, kind):
    """Window results require an actual consumer predicate before classification."""
    result = run(
        "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn "
        f"FROM events) x WHERE {selection}"
    )
    assert result.window_functions[0].selection_kind == kind
    assert len(result.dedups) == (1 if kind == "dedup" else 0)


def test_global_row_number_is_observed_without_inventing_partition():
    """Global ranking must be reported even without PARTITION BY."""
    result = run("SELECT ROW_NUMBER() OVER (ORDER BY score DESC) AS rn FROM scores")
    assert len(result.window_functions) == 1
    assert result.window_functions[0].partition_by == []
    assert result.window_functions[0].selection_kind is None
    assert result.dedups == []


def test_window_alias_selection_survives_three_nested_projections():
    """A renamed ROW_NUMBER remains traceable across nested pass-through queries."""
    result = run(
        "SELECT * FROM (SELECT rank1 AS rank2 FROM (SELECT rn AS rank1 FROM "
        "(SELECT ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) AS rn FROM events) a) b) c "
        "WHERE rank2 = 1"
    )
    assert result.dedups[0].selection == "rank2 = 1"


def test_window_alias_on_unrelated_join_source_is_not_dedup():
    """An identically named rank on another input cannot select this window."""
    result = run(
        "SELECT * FROM (SELECT ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) AS rn FROM events) a "
        "JOIN other b ON a.rn=b.rn WHERE b.rn=1"
    )
    assert result.dedups == []


def test_qualify_confirms_window_selection():
    """QUALIFY can select the window expression directly."""
    result = run(
        "SELECT * FROM events QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts) = 1", dialect="snowflake"
    )
    assert result.dedups[0].selection_kind == "dedup"


def test_cte_before_insert_resolves_to_physical_table():
    """A WITH clause attached to INSERT remains available to source resolution."""
    result = run("WITH x AS (SELECT * FROM a) INSERT INTO b SELECT * FROM x")
    assert edge_map(result)["dw.b"].sources == ["dw.a"]


@pytest.mark.parametrize(
    "condition,expected", [("NOT (a=1 OR b=2)", {("a", "!= 1"), ("b", "!= 2")}), ("NOT (a>1)", {("a", "<= 1")})]
)
def test_negation_keeps_sql_boolean_semantics(condition, expected):
    """Safe mandatory atoms reflect negation of both conjunctions and comparisons."""
    result = run(f"SELECT * FROM s WHERE {condition}", sections=["rules"])
    assert {(p.column, p.predicate) for p in result.predicates} == expected


def test_case_conditions_keep_branch_order_and_output_column():
    """Business-rule consumers need complete branch conditions in evaluation order."""
    result = run("SELECT CASE WHEN x=1 AND y=2 THEN 'A' WHEN x=1 THEN 'B' END AS label FROM s")
    assert [(c.expression, c.output_column, c.branch) for c in result.conditions] == [
        ("x = 1 AND y = 2", "label", 1),
        ("x = 1", "label", 2),
    ]


def test_temp_folding_preserves_raw_statement_graph():
    """A compact graph must still expose the original intermediate dependencies."""
    result = run("CREATE TABLE tmp AS SELECT * FROM a; INSERT INTO b SELECT * FROM tmp; DROP TABLE tmp")
    assert edge_map(result)["dw.b"].sources == ["dw.a"]
    assert [(e.target, e.sources) for e in result.raw_lineage] == [("dw.tmp", ["dw.a"]), ("dw.b", ["dw.tmp"])]


def test_rank_threshold_is_a_filter_not_top_n_or_dedup():
    """A percentile-like lower threshold selects ranks without claiming uniqueness."""
    result = run(
        "SELECT * FROM (SELECT ROW_NUMBER() OVER (ORDER BY score) AS rn FROM scores) x "
        "WHERE rn >= 0.8 * (SELECT COUNT(*) FROM scores)"
    )
    assert result.window_functions[0].selection_kind == "filtered"
    assert result.dedups == []


def test_self_join_keeps_both_key_transforms():
    """Two transforms of the same physical column must retain their respective aliases."""
    [join] = run("SELECT * FROM events a JOIN events b ON LEFT(a.id,6) = RIGHT(b.id,6)").joins
    assert join.transforms == {"dw.events@a.id": "LEFT(a.id, 6)", "dw.events@b.id": "RIGHT(b.id, 6)"}


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


@pytest.mark.parametrize("join_type", ["JOIN", "LEFT JOIN", "RIGHT JOIN"])
def test_join_row_condition_does_not_invent_self_relationship(join_type):
    """Comparisons within one row are not additional self-join relationships."""
    result = run(f"SELECT * FROM a {join_type} b ON a.id=b.id AND b.x=b.y")
    assert join_set(result) == {("dw.a", "dw.b", (("id", "id"),))}


def test_condition_spans_preserve_nested_scope_and_fragment_offsets():
    sql = """SELECT * FROM (
 SELECT a.id FROM (
  SELECT a.id FROM a JOIN b
  ON a.id = b.id
 ) a JOIN c
 ON a.id = c.id
) x JOIN d
ON x.id = d.id
AND d.label = 'WHERE fake ON fake
still a literal'
WHERE x.id > d.id"""
    result = extract_from_fragments(
        [SqlFragment(sql, "script.py", line_offset=10)], dialect="mysql", sections={"joins"}
    )
    assert {(j.left_table, j.right_table): j.condition_span for j in result.joins} == {
        ("a", "b"): (14, 14),
        ("a", "c"): (16, 16),
        ("a", "d"): (18, 20),
    }
    assert result.conditions[0].condition_span == (21, 21)
    assert all(j.line == 11 for j in result.joins)


def test_ambiguous_using_evidence_locates_original_clause():
    result = run("SELECT * FROM a JOIN b ON a.id = b.id\nJOIN c USING (\n id\n)")
    [condition] = result.conditions
    assert condition.reason == "Multiple possible left sources"
    assert condition.condition_span == (2, 4)


@pytest.mark.parametrize(
    "template", ["{{\n ref('a')\n}}", "{{ source(\n'db',\n'a') }}", "{{\n name\n}}", "${\nname\n}"]
)
def test_multiline_template_preserves_subsequent_source_lines(template):
    sql = f"SELECT * FROM {template} a\nJOIN b\nON a.id = b.id"
    processed, templated = preprocess_template(sql)
    assert templated
    assert processed.count("\n") == sql.count("\n")
    [join] = run(sql).joins
    assert join.condition_span == (5, 5)


def test_join_expression_templates_mark_full_physical_names_for_the_tool_layer():
    sql = "SELECT * FROM (SELECT id FROM raw.a UNION ALL SELECT id FROM b) x LEFT JOIN dim.d ON x.id = d.id"
    result = extract_from_fragments([SqlFragment(sql, "q.sql")], dialect="mysql", default_database="dw")
    # One physical pair per UNION branch, but a single fragment template for the tool to render.
    assert len(result.joins) == 2
    [template] = {join.expression for join in result.joins}
    assert expression_tables(template) == ["dw.b", "raw.a", "dim.d"]
    rendered = render_expression(template, lambda name: f"<{name}>")
    assert rendered == "x{<dw.b>,<raw.a>} LEFT JOIN <dim.d> ON x.id = d.id"
    assert TABLE_REF_MARK not in rendered
