import pytest

from datus.cli.autocomplete import AtReferenceParser


@pytest.fixture
def parser():
    return AtReferenceParser()


def test_parse(parser: AtReferenceParser):
    user_input = (
        "/What is the comment's rating score of the post which was created on 7/19/2010 7:19:56 PM? "
        "@Table postHistory @Table users"
    )
    parse_result = parser.parse_input(user_input)
    assert parse_result["tables"]

    assert len(parse_result["tables"]) == 2
    assert parse_result["tables"][0] == "postHistory"

    user_input = (
        "/What is the comment's rating score of the post which was created on 7/19/2010 7:19:56 PM? "
        "@Table postHistory @Table "
    )
    parse_result = parser.parse_input(user_input)
    assert parse_result["tables"]

    assert len(parse_result["tables"]) == 1

    user_input = (
        "/What is the comment's rating score of the post which was created on 7/19/2010 7:19:56 PM? "
        "Use @Table db.schema.table and @Metrics domain1.layer_1.layer_2 "
    )
    parse_result = parser.parse_input(user_input)
    assert parse_result["tables"]
    assert parse_result["metrics"]

    assert len(parse_result["tables"]) == 1
    assert len(parse_result["metrics"]) == 1


def test_parse_agent_basic(parser: AtReferenceParser):
    """``@Agent <name>`` returns the bare agent name regardless of the
    rest of the prompt — the consumer uses this as a routing hint into
    the chat agent's ``task`` tool, not as a path."""
    result = parser.parse_input("Please @Agent gen_sql write a query for me")
    assert result["agent"] == "gen_sql"


def test_parse_agent_first_match_wins(parser: AtReferenceParser):
    """Subsequent ``@Agent`` mentions are dropped on purpose so the
    dispatch hint stays deterministic — we don't want one prompt to
    fan out to two subagents."""
    result = parser.parse_input("@Agent gen_sql first @Agent gen_report second")
    assert result["agent"] == "gen_sql"


def test_parse_agent_case_insensitive(parser: AtReferenceParser):
    result = parser.parse_input("hello @agent gen_sql world")
    assert result["agent"] == "gen_sql"


def test_parse_agent_coexists_with_other_references(parser: AtReferenceParser):
    """``@Agent`` lives in its own slot — the table / metric extractors
    must remain unaffected when an agent mention is present."""
    result = parser.parse_input("@Agent gen_sql analyse @Table sales.orders for @Metrics core.revenue")
    assert result["agent"] == "gen_sql"
    assert result["tables"] == ["sales.orders"]
    assert result["metrics"] == ["core.revenue"]


def test_parse_agent_absent_returns_none(parser: AtReferenceParser):
    result = parser.parse_input("Just @Table foo.bar nothing else")
    assert result["agent"] is None


def test_parse_agent_name_does_not_swallow_following_words(parser: AtReferenceParser):
    """Identifier-only regex prevents the following sentence from being
    pulled into the agent name — the consumer would otherwise treat
    'gen_sql write me a query' as the agent name and fail visibility."""
    result = parser.parse_input("@Agent gen_sql write me a query")
    assert result["agent"] == "gen_sql"
