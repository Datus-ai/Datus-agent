from datus.storage.lancedb_conditions import And, Or, build_where, eq, ge, gte, in_, lt, ne


def test_lancedb_conditions_basic():
    node = And(
        [
            eq("status", "active"),
            ge("score", 80),
            in_("type", ["A", "B", "C"]),  # expanded into OR chain
            ne("deleted_at", None),  # -> IS NOT NULL
        ]
    )

    where_clause = build_where(node)

    assert (
        where_clause
        == "(status = 'active' AND score >= 80 AND (type = 'A' OR type = 'B' OR type = 'C') AND deleted_at IS NOT NULL)"
    )


def test_lancedb_conditions_gte_and_digit_column():
    expr = Or(
        [
            And([eq("status", "active"), gte("score", 80)]),
            And([eq("country", "US"), lt("age", 30)]),
        ],
    )
    where_clause = build_where(expr)

    assert where_clause == "((status = 'active' AND score >= 80) OR (country = 'US' AND age < 30))"

    quoted = build_where(eq("123column", "value"))
    assert quoted == "\"123column\" = 'value'"


def test_in_handles_null_values():
    clause = build_where(in_("status", ["open", None, "closed"]))
    assert clause == "(status = 'open' OR status = 'closed' OR status IS NULL)"

    null_only_clause = build_where(in_("status", [None]))
    assert null_only_clause == "(status IS NULL)"


def test_base_compile_where_accepts_node_and_string():
    assert build_where(None) is None
    assert build_where("status = 'active'") == "status = 'active'"

    node = eq("status", "active")
    assert build_where(node) == "status = 'active'"

    assert build_where("  ") is None
