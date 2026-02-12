#!/usr/bin/env bash
set -euo pipefail

# Run unit tests with coverage
python -m pytest tests/unit_tests/ \
  --cov=datus \
  --cov-report=xml:coverage.xml \
  --cov-report=term-missing \
  -s -vv --tb=short --showlocals \
  | tee pytest-coverage.txt
