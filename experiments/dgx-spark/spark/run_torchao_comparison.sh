#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/torchao-$(date -u +%Y%m%dT%H%M%SZ)}
STEPS=${STEPS:-100}
mkdir -p "$RUN_DIR"
for round in 1 2; do
  modes=(bf16 ours torchao torchao-fused)
  if [[ "$round" == 2 ]]; then modes=(torchao-fused torchao ours bf16); fi
  for mode in "${modes[@]}"; do
    precision=torchao-mxfp8
    extra=()
    if [[ "$mode" == bf16 ]]; then precision=bf16; extra=(--fuse-activation); fi
    if [[ "$mode" == ours ]]; then precision=mxfp8-native-sr; extra=(--fuse-activation); fi
    if [[ "$mode" == torchao-fused ]]; then extra=(--fuse-activation); fi
    TORCH_LOGS=graph_breaks,recompiles "$PYTHON" spark/train.py \
      --precision "$precision" --pack-backend triton-tiled --tail-precision mxfp8 \
      --compile --batch 32 --sequence 256 --steps "$STEPS" --eval-every 50 --eval-batches 1 \
      "${extra[@]}" --output "$RUN_DIR/r${round}-$mode.jsonl" \
      > "$RUN_DIR/r${round}-$mode.log" 2>&1
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"r*.jsonl > "$RUN_DIR/summary.md"
