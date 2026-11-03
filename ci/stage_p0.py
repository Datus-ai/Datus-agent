"""Copy the test harness outside the checkout without copying the datus package."""

import argparse
import shutil
from pathlib import Path


def stage(source: Path, destination: Path, environment: Path | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("tests", "ci", "conf"):
        shutil.copytree(
            source / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "p0-execution.*", "product-execution.*"),
        )
    for name in ("pyproject.toml", "pytest.ini"):
        shutil.copy2(source / name, destination / name)
    (destination / ".venv").symlink_to(environment or source / ".venv", target_is_directory=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()
    stage(args.source.resolve(), args.destination.resolve(), args.environment.resolve() if args.environment else None)
