# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""What one analyzed source contributes: write edges, query nodes, labels and temp builders."""

from datus.storage.lineage.analysis import analyze_source, content_digest


def labels(text, python=False):
    result = analyze_source("f", text, kind="file", python=python, dialect="mysql")
    by_line = {}
    for _, node, evidence in result.edges:
        if node.startswith("query:"):
            by_line[evidence.line] = result.query_labels.get(node)
    return by_line


def test_record_describes_the_analysis():
    text = "INSERT INTO t SELECT * FROM s;\nSELECT FROM (;"
    result = analyze_source("a.sql", text, kind="file", dialect="hive", default_database="dw", size=3, mtime=1.5)
    record = result.record
    assert (record.sha256, record.dialect, record.default_database) == (content_digest(text), "hive", "dw")
    assert (record.size, record.mtime, record.statements, record.parsed, record.complete) == (3, 1.5, 2, 1, False)
    assert [(issue.line, issue.reason.split(":")[0]) for issue in record.issues] == [(2, "parse error")]
    assert [(up, down, ev.line, ev.operation) for up, down, ev in result.edges] == [("dw.s", "dw.t", 1, "insert")]


def test_header_describes_only_the_first_statement_and_notes_only_the_line_they_annotate():
    text = (
        "-- Q: revenue by region\n"
        "SELECT * FROM a;\n"
        "SELECT * FROM b; -- trailing note\n"
        "-- belongs to c\n"
        "\n"
        "SELECT *\n"
        "  -- inside the query, annotating its second line\n"
        "  FROM c;\n"
        "SELECT * FROM d;\n"
    )
    assert labels(text) == {2: "Q: revenue by region", 3: "trailing note", 6: "belongs to c", 9: None}


def test_python_comments_stay_with_their_own_literal():
    text = 'A = """\n-- first query\nSELECT * FROM a\n"""\nB = """\nSELECT * FROM b\n"""\n'
    assert labels(text, python=True) == {3: "first query", 6: None}


def test_identical_queries_share_one_node():
    result = analyze_source("f", "SELECT * FROM a;\nSELECT  *  FROM  a;", kind="file", dialect="mysql")
    assert len({node for _, node, _ in result.edges}) == 1
    assert [ev.line for _, _, ev in result.edges] == [1, 2]


def test_temp_builders_are_located_within_the_same_python_literal():
    text = (
        'X = """\nCREATE TABLE tmp AS SELECT * FROM other;\nDROP TABLE tmp;\n"""\n'
        'Y = """\nCREATE TABLE tmp AS SELECT * FROM src;\nINSERT INTO final SELECT * FROM tmp;\nDROP TABLE tmp;\n"""\n'
    )
    result = analyze_source("f.py", text, kind="file", python=True, dialect="mysql")
    [(up, down, evidence)] = [edge for edge in result.edges if edge[1] == "final"]
    assert (up, evidence.line) == ("src", 7)
    assert [(hop.table, hop.lines) for hop in evidence.via_temp] == [("tmp", [6])]
