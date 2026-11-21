#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
FP4_EPILOGUE=${FP4_EPILOGUE:-eager}
STEPS=${STEPS:-100}
PACK_BACKEND=${PACK_BACKEND:-triton-tiled}
RUN_DIR=${RUN_DIR:-results/precision-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
"$PYTHON" spark/check_mxfp8.py --pack-backend "$PACK_BACKEND" | tee "$RUN_DIR/check-mxfp8.json"
for batch in 4 16 32; do
  for mode in bf16 mxfp8-native-sr fp4-native-sr; do
    "$PYTHON" spark/train.py --precision "$mode" --pack-backend "$PACK_BACKEND" \
      --tail-precision mxfp8 --fp4-epilogue "$FP4_EPILOGUE" --batch "$batch" --steps "$STEPS" \
      --eval-every 50 --eval-batches "$((32/batch))" \
      --output "$RUN_DIR/b${batch}-${mode}.jsonl"
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"b*.jsonl | tee "$RUN_DIR/summary.md"
