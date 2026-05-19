#!/usr/bin/env bash
# Pre-download fastembed's all-MiniLM-L6-v2 model on the host into a local
# cache directory so the Docker build can COPY it into the image. Use this
# when the Docker build network can't reach huggingface.co / hf-mirror.com
# directly.
#
# Override the HF endpoint via the environment if needed:
#   HF_ENDPOINT=https://huggingface.co ./build_scripts/prefetch_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$SCRIPT_DIR/cache/fastembed"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export FASTEMBED_CACHE_PATH="$CACHE_DIR"

mkdir -p "$CACHE_DIR"

if command -v uv >/dev/null 2>&1 && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    RUNNER=(uv run --project "$PROJECT_ROOT" python -)
else
    RUNNER=(python3 -)
fi

echo "[prefetch] HF_ENDPOINT=$HF_ENDPOINT"
echo "[prefetch] cache_dir=$CACHE_DIR"

# Step 1: download the model via fastembed. It will try HF first and fall
# back to its native source if HF is unreachable. The resulting layout:
#   $CACHE_DIR/fast-all-MiniLM-L6-v2/{model.onnx,tokenizer.json,...}
#   $CACHE_DIR/models--qdrant--all-MiniLM-L6-v2-onnx/refs/main  (commit hash)
"${RUNNER[@]}" <<'PY'
import os
from fastembed import TextEmbedding

cache_dir = os.environ["FASTEMBED_CACHE_PATH"]
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=cache_dir)
list(model.embed(["warmup"]))
print("[prefetch] fastembed download done")
PY

# Step 2: datus's check_snapshot expects an HF-snapshot layout with the model
# files materialised under snapshots/<commit>/. Mirror the native files there
# so the runtime can load fully offline.
HF_DIR="$CACHE_DIR/models--qdrant--all-MiniLM-L6-v2-onnx"
NATIVE_DIR="$CACHE_DIR/fast-all-MiniLM-L6-v2"
REF_FILE="$HF_DIR/refs/main"

if [ ! -f "$REF_FILE" ]; then
    echo "[prefetch] ERROR: expected ref file $REF_FILE not found." >&2
    exit 1
fi
if [ ! -d "$NATIVE_DIR" ]; then
    echo "[prefetch] ERROR: native cache $NATIVE_DIR not found." >&2
    exit 1
fi

COMMIT="$(tr -d '[:space:]' < "$REF_FILE")"
if [ -z "$COMMIT" ]; then
    echo "[prefetch] ERROR: empty commit hash in $REF_FILE" >&2
    exit 1
fi
SNAP_DIR="$HF_DIR/snapshots/$COMMIT"
mkdir -p "$SNAP_DIR"

# Required artifacts: if any are missing the runtime check_snapshot() will fail,
# so fail fast here instead of producing a silently broken cache.
required_files=(model.onnx config.json tokenizer.json tokenizer_config.json)
for f in "${required_files[@]}"; do
    if [ ! -f "$NATIVE_DIR/$f" ]; then
        echo "[prefetch] ERROR: required artifact missing: $NATIVE_DIR/$f (snapshot $SNAP_DIR)" >&2
        exit 1
    fi
    cp -f "$NATIVE_DIR/$f" "$SNAP_DIR/$f"
done

# Optional artifacts: copy when present; absence is non-fatal.
for f in special_tokens_map.json vocab.txt; do
    [ -f "$NATIVE_DIR/$f" ] && cp -f "$NATIVE_DIR/$f" "$SNAP_DIR/$f"
done

# Strip macOS AppleDouble sidecar files that confuse Linux readers.
find "$CACHE_DIR" -name '._*' -delete 2>/dev/null || true

echo "[prefetch] snapshot populated: $SNAP_DIR"
echo "[prefetch] model cached at $CACHE_DIR"
