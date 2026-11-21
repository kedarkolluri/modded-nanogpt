#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-100}
PACK_BACKEND=${PACK_BACKEND:-triton-tiled}
RUN_DIR=${RUN_DIR:-results/fused-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
for batch in 4 16 32; do
  for mode in bf16 fp4-native-sr; do
    "$PYTHON" spark/train.py --precision "$mode" --pack-backend "$PACK_BACKEND" \
      --batch "$batch" --steps "$STEPS" --eval-every 50 --eval-batches "$((32/batch))" \
      --output "$RUN_DIR/b${batch}-${mode}.jsonl"
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"b*.jsonl | tee "$RUN_DIR/summary.md"
