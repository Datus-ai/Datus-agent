#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache/fastembed"

# Ensure the fastembed model cache is populated on the host before building;
# the Dockerfile COPY-step pulls from this directory. Validate sentinel files
# rather than mere non-emptiness so a half-finished prefetch never bakes a
# broken cache into the image.
if [ ! -f "$CACHE_DIR/models--qdrant--all-MiniLM-L6-v2-onnx/refs/main" ] || \
   [ ! -f "$CACHE_DIR/fast-all-MiniLM-L6-v2/model.onnx" ]; then
    echo "[build] fastembed cache missing or incomplete, running prefetch..."
    "$SCRIPT_DIR/prefetch_model.sh"
fi

docker build -t datus-agent:latest -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"
