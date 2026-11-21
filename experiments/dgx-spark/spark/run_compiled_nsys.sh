#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-python3}
RUN_DIR=${RUN_DIR:-results/compiled-nsys-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
for mode in bf16 mxfp8; do
  precision=bf16
  if [[ "$mode" == mxfp8 ]]; then precision=mxfp8-native-sr; fi
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$RUN_DIR/$mode-fused" "$PYTHON" spark/train.py \
    --precision "$precision" --tail-precision mxfp8 --pack-backend triton-tiled \
    --fuse-activation --compile --batch 32 --sequence 256 --steps 24 --eval-every 50 \
    --eval-batches 1 --nsight --profile-start 20 --profile-steps 3 \
    --output "$RUN_DIR/$mode.jsonl" > "$RUN_DIR/$mode.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum,cuda_kern_exec_sum,nvtx_gpu_proj_sum,cuda_api_sum \
    --format csv --output "$RUN_DIR/$mode" --force-export=true \
    "$RUN_DIR/$mode-fused.nsys-rep" > "$RUN_DIR/stats-$mode.log" 2>&1
  "$PYTHON" spark/analyze_nsys.py "$RUN_DIR/$mode-fused.sqlite" > "$RUN_DIR/$mode-analysis.json"
done
