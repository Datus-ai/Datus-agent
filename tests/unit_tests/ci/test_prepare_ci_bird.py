import sqlite3

import pytest

from ci.prepare_ci_bird import prepare_ci_bird
from tests.conftest import load_acceptance_config


@pytest.mark.acceptance
def test_ci_bird_fixture_loads_without_home_benchmark(tmp_path, monkeypatch):
    root = tmp_path / "dev_databases"
    prepare_ci_bird(root)
    monkeypatch.setenv("DATUS_TEST_BIRD_ROOT", str(root))

    config = load_acceptance_config(datasource="bird_sqlite", home=str(tmp_path))
    assert set(config.list_databases("bird_sqlite")) == {
        "california_schools",
        "card_games",
        "financial",
        "toxicology",
    }
    assert str(root) in config.services.datasources["bird_school"].uri

    with sqlite3.connect(root / "financial" / "financial.sqlite") as connection:
        count = connection.execute(
            "SELECT COUNT(card.card_id) FROM card JOIN disp ON card.disp_id = disp.disp_id "
            "WHERE card.type = 'gold' AND disp.type = 'OWNER'"
        ).fetchone()[0]
    assert count == 1
    assert (
        sum((root / name / f"{name}.sqlite").stat().st_size for name in ("card_games", "financial", "toxicology"))
        < 1_000_000
    )
