#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 /path/to/autogluon-fev [output-dir]" >&2
  echo "environment: FEV_DEVICE=mps|cpu|cuda (default: mps)" >&2
  echo "             FEV_BATCH_SIZE=512" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 2
fi

FEV_REPO=$1
OUTPUT_DIR=${2:-output/fev_bench}
DEVICE=${FEV_DEVICE:-mps}
BATCH_SIZE=${FEV_BATCH_SIZE:-512}
PINNED_COMMIT=81cf1255bb0c88dc039ae9bca23f73db6d9dfa61
ARCHIVE_ROOT=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$ARCHIVE_ROOT/../.." && pwd)

if [ ! -d "$FEV_REPO/.git" ]; then
  echo "FEV repository not found: $FEV_REPO" >&2
  usage
  exit 2
fi

actual_commit=$(git -C "$FEV_REPO" rev-parse HEAD)
if [ "$actual_commit" != "$PINNED_COMMIT" ]; then
  echo "FEV repository is not at the pinned commit:" >&2
  echo "  expected: $PINNED_COMMIT" >&2
  echo "  actual:   $actual_commit" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
adapter_root="$ARCHIVE_ROOT/models/fracast-0"
staged_root="$FEV_REPO/models/fracast-0"
cp "$adapter_root/model.py" "$staged_root/model.py"
cp "$adapter_root/requirements.txt" "$staged_root/requirements.txt"

MODEL_KWARGS=$(printf '{"model_id":"ztxtech/fracast-0","weights":"w8","batch_size":%s,"device":"%s"}' "$BATCH_SIZE" "$DEVICE")
cd "$OUTPUT_DIR"
python "$FEV_REPO/models/evaluate.py" \
  --model fracast-0 \
  --name fracast-0 \
  --benchmark "$FEV_REPO/benchmarks/fev_bench/tasks.yaml" \
  --model-kwargs "$MODEL_KWARGS" \
  --deps-installed

echo "FEV result written to $OUTPUT_DIR/fracast-0.csv"
echo "Archived run settings: device=$DEVICE, batch_size=$BATCH_SIZE"
