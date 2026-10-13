#!/usr/bin/env python3

"""Build the small SQLite fixtures used by untrusted PR and merge-queue jobs."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "tests" / "data" / "ci_bird"
CALIFORNIA_SOURCE = REPO_ROOT / "datus" / "sample_data" / "california_schools" / "california_schools.sqlite"
GENERATED_DATABASES = ("card_games", "financial", "toxicology")


def prepare_ci_bird(root: Path) -> None:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    california_target = root / "california_schools" / "california_schools.sqlite"
    california_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CALIFORNIA_SOURCE, california_target)

    for name in GENERATED_DATABASES:
        target = root / name / f"{name}.sqlite"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        schema = (SOURCE_DIR / f"{name}.sql").read_text(encoding="utf-8")
        with sqlite3.connect(target) as connection:
            connection.executescript(schema)

    financial = root / "financial" / "financial.sqlite"
    with sqlite3.connect(financial) as connection:
        matching_cards = connection.execute(
            "SELECT COUNT(card.card_id) FROM card JOIN disp ON card.disp_id = disp.disp_id "
            "WHERE card.type = 'gold' AND disp.type = 'OWNER'"
        ).fetchone()[0]
    if matching_cards != 1:
        raise RuntimeError(f"CI financial fixture must contain one matching card, got {matching_cards}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Absolute dev_databases directory in runner temp")
    args = parser.parse_args()
    if not args.root.is_absolute():
        parser.error("root must be an absolute path")
    prepare_ci_bird(args.root)
    print(f"Prepared CI BIRD SQLite fixtures under {args.root}")


if __name__ == "__main__":
    main()
