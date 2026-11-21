#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-100}
RUN_DIR=${RUN_DIR:-results/activation-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
for round in 1 2; do
  modes=(bf16 mxfp8 fused)
  if [[ "$round" == 2 ]]; then modes=(fused mxfp8 bf16); fi
  for mode in "${modes[@]}"; do
    precision=mxfp8-native-sr
    extra=()
    if [[ "$mode" == bf16 ]]; then precision=bf16; fi
    if [[ "$mode" == fused ]]; then extra=(--fuse-activation); fi
    "$PYTHON" spark/train.py --precision "$precision" --pack-backend triton-tiled \
      --tail-precision mxfp8 --batch 32 --sequence 256 --steps "$STEPS" \
      --eval-every 50 --eval-batches 1 "${extra[@]}" \
      --output "$RUN_DIR/r${round}-${mode}.jsonl"
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"r*.jsonl | tee "$RUN_DIR/summary.md"
