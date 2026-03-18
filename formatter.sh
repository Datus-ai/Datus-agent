#!/bin/bash
set -euo pipefail

echo "Starting code formatting..."

# 1. Format with ruff (replaces black + isort)
echo "1. Formatting code..."
ruff format datus/ tests/

# 2. Lint and auto-fix (replaces flake8 + isort)
echo "2. Linting and auto-fixing..."
ruff check --fix datus/ tests/

# 3. Check remaining issues
echo "3. Checking remaining issues..."
ruff check datus/ tests/

echo "Formatting completed!"
