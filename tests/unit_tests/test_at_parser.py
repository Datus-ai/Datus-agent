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
    print("$$$", parse_result)
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
