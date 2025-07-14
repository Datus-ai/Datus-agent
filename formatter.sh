#!/bin/bash

echo "Starting code formatting..."

# 1. Remove unused imports and variables (aggressive cleanup)
echo "1. Removing unused imports and variables..."
autoflake --in-place --remove-unused-variables --remove-all-unused-imports --remove-duplicate-keys --recursive datus/

# 2. Format with Black (must run before autopep8 as Black is more strict)
echo "2. Formatting with Black..."
black --line-length=120 datus/

# 3. Sort imports
echo "3. Sorting imports..."
isort --profile=black --line-length=120 datus/

# 4. Fix remaining PEP8 issues
echo "4. Fixing PEP8 issues..."
autopep8 --in-place --max-line-length=120 --aggressive --aggressive --recursive datus/

# 5. Check remaining issues
echo "5. Checking remaining issues..."
flake8 --max-line-length=120 --extend-ignore=E203,W503 datus/

echo "Formatting completed!"
