#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/compile-$(date -u +%Y%m%dT%H%M%SZ)}
STEPS=${STEPS:-100}
mkdir -p "$RUN_DIR"
for round in 1 2; do
  modes=(bf16-compiled bf16-fused-compiled mxfp8-compiled mxfp8-fused-compiled bf16-fused-eager mxfp8-fused-eager)
  if [[ "$round" == 2 ]]; then modes=(mxfp8-fused-eager bf16-fused-eager mxfp8-fused-compiled mxfp8-compiled bf16-fused-compiled bf16-compiled); fi
  for mode in "${modes[@]}"; do
    precision=bf16
    if [[ "$mode" == mxfp8* ]]; then precision=mxfp8-native-sr; fi
    extra=()
    if [[ "$mode" == *fused* ]]; then extra+=(--fuse-activation); fi
    if [[ "$mode" == *compiled ]]; then extra+=(--compile); fi
    TORCH_LOGS=graph_breaks,recompiles "$PYTHON" spark/train.py \
      --precision "$precision" --pack-backend triton-tiled --tail-precision mxfp8 \
      --batch 32 --sequence 256 --steps "$STEPS" --eval-every 50 --eval-batches 1 \
      "${extra[@]}" --output "$RUN_DIR/r${round}-$mode.jsonl" \
      > "$RUN_DIR/r${round}-$mode.log" 2>&1
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"r*.jsonl > "$RUN_DIR/summary.md"
