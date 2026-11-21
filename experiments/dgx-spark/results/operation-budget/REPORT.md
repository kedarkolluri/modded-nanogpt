# Complete Spark operation and training budget

The comparison target is compiled BF16 and the established native MXFP8 path.
A reduction relative to a slower experimental kernel is recovery from a
regression; it is not a saving relative to the training baseline.

## Latest complete GPU kernel accounting

Fresh Nsight Systems captures: three warmed steps beginning at step 20, same
123.6M model, batch32/sequence256, compiled model, fused activation, same source
and environment. All three captures have zero unmatched kernels. Kernel sums
reconcile exactly before rounding. These durations include profiler overhead;
unprofiled training times below come from separate 100-step runs.

Fused operators share one measured kernel; their individual costs cannot be
separated by this capture. Names below group every kernel exhaustively. All
individual names, phase attribution, call counts and durations are in the three
`nsys/*-all-kernels.csv` files. No operator category is silently dropped.

| GPU kernel category, ms/step | BF16 | Current MXFP8 | Weight reuse |
|---|---:|---:|---:|
| Fused normalization / residual / rotary / related ops | 30.03 | 30.30 | 30.14 |
| Other fused pointwise / casts / copies / embedding / RNG | 7.20 | 5.26 | 5.33 |
| Other GEMMs (BF16 MLP, attention projections, head) | 75.10 | 41.59 | 42.14 |
| Flash attention | 14.68 | 14.71 | 14.57 |
| MLP activation forward | 5.26 | — | — |
| Cross entropy / softmax | 10.48 | 10.61 | 10.64 |
| MLP activation derivative | 7.95 | 7.70 | 7.65 |
| Optimizer (Muon + AdamW) | 58.05 | 58.70 | 57.83 |
| MXFP8 row packing (includes fused activation packing) | — | 17.24 | 14.79 |
| MXFP8 transpose packing | — | 14.70 | 14.55 |
| Native MXFP8 GEMMs | — | 19.61 | 16.17 |
| Activation quantization + first-projection GEMM | — | — | 13.78 |
| **Total GPU kernel time** | **208.76** | **220.43** | **227.59** |

A dash means no separate kernel in that category, not free computation. BF16
MLP GEMMs are included in the other-GEMMs row. The reuse kernel includes both
its first-projection GEMM and activation quantization. The optimizer row includes
all Muon and AdamW kernels. A category with several fused operations is not
counted again under its constituent operations.

## Phase totals from the same captures

| Phase, ms/step | BF16 | Current MXFP8 | Weight reuse |
|---|---:|---:|---:|
| forward | 55.53 | 62.17 | 70.29 |
| backward | 95.18 | 99.55 | 99.47 |
| optimizer | 58.05 | 58.70 | 57.83 |

GPU kernel sums differ slightly from busy-time unions if kernels overlap.
The captured first-to-last-kernel spans are 213.90/226.18/233.72 ms per step;
no-kernel intervals within those spans are 5.36/5.91/6.25 ms. These intervals
include boundaries/synchronization/memory operations and are not all CPU stalls.

## Actual full-training step time

Separate unprofiled 100-step runs, first five excluded. Each step processes
8,192 tokens. The times include forward, backward and optimizer; input preparation
and validation are excluded. One run per path; no statistical significance claim.

| Path | Mean measured step ms | Tokens/s |
|---|---:|---:|
| Compiled BF16 | 214.55 | 38,182 |
| Current native MXFP8 | 228.27 | 35,888 |
| Weight reuse MXFP8 | 233.28 | 35,117 |

All runs completed with finite losses and zero graph breaks. Weight reuse adds
5.01 ms to the measured mean step versus current MXFP8 in this comparison.

## Where the apparent 50% improvement went

Fresh-seed isolated production-shape timings, including required packing:

| First projection region | ms per call | Reference |
|---|---:|---|
| Separate native packing + native GEMM | 0.638 | Established MXFP8 path |
| Both operands quantized inside loop | 2.442 | Experimental regression |
| Weight reuse + wider output tile | 1.171 | Improved experimental path |

Twelve first projections run per training step. Against the native 0.638 ms
baseline, the reuse version is still 0.533 ms slower per isolated call. The
2.442→1.171 ms reduction did not beat that baseline. The native baseline already
quantizes each complete operand once; our all-in-loop experiment introduced
repeated conversion that subsequent experiments partially removed.

The full-model Nsight budget makes this explicit:

| Current→reuse change | GPU ms/step |
|---|---:|
| Native MXFP8 GEMM time removed | -3.44 |
| Row-packing time removed | -2.45 |
| New fused first-projection kernels added | +13.78 |
| **Subtotal for these replaced operations** | **+7.89** |

Other kernel groups vary slightly between captures; the complete GPU kernel
sum increases 7.16 ms. This profile delta and the unprofiled wall-step delta
come from separate runs and must not be forced into an exact additive identity.

Against BF16, current MXFP8 saves 13.90 ms in non-optimizer GEMM kernels. It
adds 31.94 ms of row/transpose packing and removes 5.26 ms of separate forward
activation kernels. The remaining categories reconcile the net +11.67 ms in
total GPU kernel time. Packing still exceeds the GEMM benefit.

## Historical implementation comparison

Compiled model unless marked eager. These are different experiment sessions,
not one interleaved run. Step ms is 8192 / reported tokens-per-second * 1000;
when throughput is a two-run mean, this is a representative derived time, not
the arithmetic mean of all individual steps. Each experiment retains its matched
BF16 control in its own report. No full-training run exists for explicit buffering.

| Implementation | Representative step ms | Tokens/s | Evidence |
|---|---:|---:|---|
| BF16, eager fused | 365.20 | 22,432 | 2 × 100 steps; compile report |
| MXFP8, eager fused | 375.45 | 21,819 | 2 × 100 steps; compile report |
| BF16, ordinary compiled | 216.86 | 37,776 | 2 × 100 steps; compile report |
| MXFP8, ordinary compiled | 246.60 | 33,220 | 2 × 100 steps; compile report |
| BF16, fused compiled | 216.44 | 37,848 | 2 × 100 steps; compile report |
| MXFP8, fused compiled | 230.49 | 35,542 | 2 × 100 steps; compile report |
| TorchAO MXFP8 ordinary | 233.28 | 35,116 | 2 × 100 steps; TorchAO report |
| TorchAO MXFP8 fused | 228.65 | 35,827 | 2 × 100 steps; TorchAO report |
| MXFP8 epilogue fusion | 227.49 | 36,011 | 2 × 100 steps; epilogue report |
| Both operands inside GEMM loop | 292.15 | 28,041 | 1 × 20 steps; short performance check |
| Explicit lookahead buffering | Not measured | Not measured | Kernel tests only |
| Weight reuse | 233.28 | 35,117 | 1 × 100 steps; latest comparison |
| Dedicated CUDA producer/consumer | 258.31 | 31,714 | 1 × 100 steps; separate producer/consumer report |

Original eager FP4/NVFP4 and earlier batch-size sweeps are separate historical
recipes; see `results/mxfp8/three-way/summary.md` and `results/REPORT.md`. They
must not be ranked directly against the compiled runs here.

## Reproduce

```sh
PYTHON=$PY RUN_DIR=results/new-operation-budget spark/run_operation_budget_nsys.sh
$PY spark/analyze_operation_budget.py results/new-operation-budget
```

The analyzer joins CUDA kernels to runtime launch correlation IDs, then assigns
serial training phase windows across threads. This handles PyTorch backward
worker-thread launches that default NVTX GPU projection misses. Raw binary
Nsight captures and SQLite databases remain on Spark and are ignored by Git.

A subsequent [explicit producer/consumer experiment](../producer-consumer/REPORT.md)
verified concurrent worker roles with shared buffers and passing sanitizers.
Its 100-step throughput is 31,714 tokens/s. That variant is not included in the
three matched Nsight operation budgets above; its evidence and timings are
reported separately.
