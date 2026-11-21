#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/profiles-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
for mode in bf16 mxfp8-native-sr fp4-native-sr; do
  "$PYTHON" spark/train.py --precision "$mode" --pack-backend triton-tiled \
    --tail-precision mxfp8 --batch 32 --steps 25 --eval-every 100 --eval-batches 1 \
    --profile-start 20 --profile-steps 3 --profile-dir "$RUN_DIR/$mode" \
    --output "$RUN_DIR/$mode.jsonl"
done
