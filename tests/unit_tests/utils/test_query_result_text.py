from datus.utils.query_result_text import omitted_rows, omitted_rows_line, parse_query_result_text


def test_reads_the_id_row_count_and_rows():
    parsed = parse_query_result_text(
        "result_id: query_metrics:abc\nrows: 30\nwarning where_unread: not read; hint: x: y\n\n"
        "a,b\n1,2\n... 28 rows omitted\n3,4"
    )

    assert parsed.result_id == "query_metrics:abc"
    assert parsed.row_count == 30
    assert parsed.lines[2] == "warning where_unread: not read; hint: x: y"
    assert parsed.columns == ["a", "b"]
    assert parsed.shown_rows == 2


def test_a_row_count_line_with_a_note_still_reads():
    parsed = parse_query_result_text("rows: 10000 (reached the 10000-row cap; there may be more)\n\nid\n1")

    assert parsed.result_id is None
    assert parsed.row_count == 10000


def test_an_empty_result_keeps_its_header():
    parsed = parse_query_result_text("result_id: q\nrows: 0\n\nregion,value")

    assert parsed.columns == ["region", "value"]
    assert parsed.shown_rows == 0


def test_text_that_is_not_a_query_result_reads_as_none():
    assert parse_query_result_text("description: x\npath: a / b") is None
    assert parse_query_result_text({"result_id": "q"}) is None
    assert parse_query_result_text(None) is None


def test_the_omitted_rows_line_reads_back():
    assert omitted_rows_line(51) == "... 51 rows omitted"
    assert omitted_rows_line(1) == "... 1 row omitted"
    assert omitted_rows(omitted_rows_line(51)) == 51
    assert omitted_rows(omitted_rows_line(1)) == 1
    assert omitted_rows("...,...") is None
