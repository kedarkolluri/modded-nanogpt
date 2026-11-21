#!/usr/bin/env bash
set -euo pipefail
# Set PYTHON to the CUDA-enabled aarch64 interpreter on the target Spark.
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-200}
RUN_DIR=${RUN_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
"$PYTHON" spark/check_precision.py | tee "$RUN_DIR/check-precision.json"
"$PYTHON" spark/check_native.py | tee "$RUN_DIR/check-native.json"
for mode in bf16 fp4-native-sr fp4-native-nearest; do
  "$PYTHON" spark/train.py --precision "$mode" --steps "$STEPS" \
    --eval-every 100 --output "$RUN_DIR/$mode.jsonl"
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"*.jsonl | tee "$RUN_DIR/summary.md"
