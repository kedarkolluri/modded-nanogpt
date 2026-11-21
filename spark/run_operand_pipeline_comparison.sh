#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/pipelined-mxfp8-comparison}
STEPS=${STEPS:-100}
mkdir -p "$RUN_DIR"
for round in 1 2; do
  modes=(bf16 current pipeline)
  if [[ "$round" == 2 ]]; then modes=(pipeline current bf16); fi
  for mode in "${modes[@]}"; do
    precision=mxfp8-native-sr
    extra=()
    if [[ "$mode" == bf16 ]]; then precision=bf16; fi
    if [[ "$mode" == pipeline ]]; then extra=(--pipeline-mxfp8); fi
    TORCH_LOGS=graph_breaks,recompiles "$PYTHON" spark/train.py \
      --precision "$precision" --pack-backend triton-tiled --tail-precision mxfp8 \
      --compile --fuse-activation --batch 32 --sequence 256 --steps "$STEPS" \
      --eval-every 50 --eval-batches 1 "${extra[@]}" \
      --output "$RUN_DIR/r${round}-$mode.jsonl" > "$RUN_DIR/r${round}-$mode.log" 2>&1
  done
done
"$PYTHON" spark/summarize.py "$RUN_DIR/"r*.jsonl > "$RUN_DIR/summary.md"
