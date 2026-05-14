#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache/fastembed"

# Ensure the fastembed model cache is populated on the host before building;
# the Dockerfile COPY-step pulls from this directory.
if [ ! -d "$CACHE_DIR" ] || [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "[build] fastembed cache missing, running prefetch..."
    "$SCRIPT_DIR/prefetch_model.sh"
fi

docker build -t datus-agent:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
