#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/reuse-comparison-$(date -u +%Y%m%dT%H%M%SZ)}
STEPS=${STEPS:-100}
mkdir -p "$RUN_DIR"
for mode in bf16 current reuse; do
  precision=mxfp8-native-sr
  extra=()
  if [[ "$mode" == bf16 ]]; then precision=bf16; fi
  if [[ "$mode" == reuse ]]; then extra=(--pipeline-mxfp8 --reuse-mxfp8-weights); fi
  "$PYTHON" spark/train.py --precision "$precision" --pack-backend triton-tiled \
    --tail-precision mxfp8 --compile --fuse-activation --batch 32 --sequence 256 \
    --steps "$STEPS" --eval-every 50 --eval-batches 1 "${extra[@]}" \
    --output "$RUN_DIR/$mode.jsonl" > "$RUN_DIR/$mode.log" 2>&1
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"*.jsonl > "$RUN_DIR/summary.md"
