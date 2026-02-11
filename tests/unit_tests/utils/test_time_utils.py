from datus.utils.time_utils import format_duration_human


def test_human_time():
    assert format_duration_human(23 * 60 + 36) == "23m36s"

    assert format_duration_human(1 * 3600 + 24 * 60 + 30) == "1h24m30s"

    assert format_duration_human(2 * 3600 + 3 * 60) == "2h3m"
