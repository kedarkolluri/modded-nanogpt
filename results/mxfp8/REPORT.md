# Native MXFP8 baseline, profiles and FP4 epilogue optimization

2026-11-20. A native MXFP8 MLP baseline now runs on GB10. The comparison still favors BF16; profile-guided epilogue fusion improves FP4 and brings it close to MXFP8.

## Controlled training comparison

All runs use the same 123,568,128-parameter ModernArch adaptation, seed 42, sequence length 256, 100 steps, fixed optimizer settings and 8,192 validation tokens. Compare precision modes within each batch; larger batches see more training tokens.

- BF16: ordinary autograd and BF16 autocast throughout.
- MXFP8: native MXFP8 forward and backward in all twelve MLPs. Forward rounding is nearest; backward stochastically re-quantizes the saved quantized forward values.
- FP4/MXFP8: NVFP4 forward in the first ten MLPs, MXFP8 forward in the final two; stochastic MXFP8 backward throughout.

Attention and tied embedding/head retain BF16 autocast in all modes. FP32 master weights and the same optimizers are retained. The two low-precision modes share the backward estimator and fused packing backend. This isolates the forward-precision recipe more closely than the earlier BF16-only comparison; it is not an exact reproduction of the article’s proprietary model or kernels.

| Batch | BF16 tokens/s | MXFP8 tokens/s | FP4/MXFP8 tokens/s |
|---|---:|---:|---:|
| 4 | 9,117 | 8,721 | 8,479 |
| 16 | 16,472 | 15,809 | 14,920 |
| 32 | 18,064 | 17,627 | 16,627 |

The initial FP4 path is 2.8%, 5.6% and 5.7% below MXFP8 throughput at batches 4, 16 and 32 respectively. These figures do not reproduce the article’s reported 8% pretraining gain. Timings exclude the first five steps, data preparation and validation, and include the optimizer.

## Profile findings

Each batch-32 profile captures three steps after twenty warmup/training steps. The table sums actual CUDA kernel durations per step; it is not full job wall time. GPU user annotation ranges are excluded, and kernels are attributed to CPU ranges through launch correlation IDs. All kernels matched, and phase totals reconcile with kernel totals in all four profiles. Nested phase/quantization ranges overlap.

| Mode | Forward kernel ms | Backward kernel ms | Optimizer kernel ms | Total kernel ms |
|---|---:|---:|---:|---:|
| BF16 | 137.38 | 236.86 | 58.99 | 433.24 |
| MXFP8 | 142.22 | 249.57 | 59.23 | 451.02 |
| FP4 before epilogue fusion | 172.35 | 254.85 | 61.60 | 488.80 |
| FP4 after epilogue fusion | 144.46 | 249.21 | 60.98 | 454.65 |

The initial FP4 forward pass costs about 30 ms more than MXFP8. Its separate outer-scale multiplies and FP32-to-BF16 conversion account for about 31.8 ms across the FP4 MLPs. Backward FP8 packing is almost identical: 22.46 ms for FP4/MXFP8 and 22.60 ms for MXFP8. The common optimizer costs about 59–62 ms, and the large-vocabulary softmax forward/backward is another substantial shared cost.

## Profile-guided fix

`--fp4-epilogue triton` fuses both outer-scale multiplies and the final BF16 conversion. Exact output and nearest-backward gradient parity passed against the eager implementation. At an 8192x3072 output shape, the isolated epilogue dropped from 2.395 ms to 0.680 ms.

A fresh 100-step batch-32 run improved FP4 throughput from **16,627 to 17,458 tokens/s**, about **5.0% faster**. Its validation loss was 7.1901, versus 7.2176 before fusion, 7.1827 for MXFP8 and 7.1723 for BF16. One seed and this short token budget cannot establish accuracy differences.

The optimized FP4 run is still **1.0% below MXFP8** and **3.4% below BF16** throughput. There is no demonstrated win over either baseline. The remaining investigation should target activation/packing memory traffic and native GEMM output handling. Optimizing shared work only in FP4 would require careful interpretation of any later precision claim.

## Validation and limits

Native MXFP8 forward and both backward GEMMs passed reference checks under eager, row-Triton and tiled-Triton packing. Relative errors were below 0.004. Saved dequantized values, strided inputs, zero rows, outliers and MLP precision coverage passed. Existing FP4 and packing regression checks also passed. All ten 100-step runs completed with finite losses.

Peak memory remains roughly unchanged; saved activations are reconstructed BF16, and optimizer/master state is not compressed. Runs are single-seed diagnostics with fixed learning rates and GPU algorithms not forced deterministic. They do not establish convergence to the race target. Hardware is GB10 SM 12.1, PyTorch 2.14.0+cu130, Triton 3.8.0, driver 580.178.04. No installed packages were changed.

## Reproduce and recover

```sh
PY=/home/kedar/projects/lawzero-reproduction/.venv/bin/python
cd /home/kedar/projects/inklings2-mini
$PY spark/check_mxfp8.py --pack-backend triton-tiled
$PY spark/check_epilogue.py
PYTHON=$PY RUN_DIR=results/my-three-way spark/run_precision_comparison.sh
PYTHON=$PY FP4_EPILOGUE=triton RUN_DIR=results/my-optimized-comparison \
  spark/run_precision_comparison.sh
PYTHON=$PY FP4_EPILOGUE=triton RUN_DIR=results/my-profiles spark/run_profiles.sh
```

Core model and quantization code is the same across the original three-way benchmarks and their profiles; profiling-helper changes add attribution and validation. Source hashes are recorded in every run. Compressed Chrome traces are available under `profiles/` locally and on Spark but excluded from Git; JSON summaries, raw logs and reports are checked in.

Before MXFP8: commit `67290e4`, tag `spark-fused-v1`. MXFP8 implementation: commit `d7eef99`, tag `spark-mxfp8-implementation-v1`. Recovery bundles are stored on Spark in `snapshots/`.
