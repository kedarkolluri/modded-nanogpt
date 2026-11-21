# Activation fusion on DGX Spark

ReLU-square is now fused into MXFP8 input packing for the second MLP projection.
The kernel preserves the original BF16 square rounding, computes block scales,
and writes quantized values and backward reconstructions in one pass. A second
kernel fuses the activation derivative. Gradient packing remains separate.
A BF16 activation-fusion control uses the same derivative optimization.

## Unprofiled training results

124M-parameter ModernArch modded-nanogpt model; GB10 SM 12.1; PyTorch
2.14.0+cu130 and Triton 3.8.0. FineWeb, batch 32, sequence 256, seed 42,
100 steps per run. Throughput excludes the first five steps, input preparation,
and validation, and includes optimizer work. Each run trains on 819,200 tokens.

| Mode | Repeat 1 tokens/s | Repeat 2 tokens/s | Mean tokens/s | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Original BF16 | 18,403 | 18,396 | 18,399 | 10.54 |
| Original MXFP8 | 17,736 | 17,751 | 17,744 | 10.54 |
| Fused MXFP8 | 21,680 | 21,653 | 21,666 | 9.42 |
| Fused BF16 control | 22,506 | 22,455 | 22,481 | 9.41 |

Fused MXFP8 improves throughput **22.1% over original MXFP8** and **17.8%
over original BF16**. However, it remains **3.6% below fused BF16**. The result
shows a substantial fusion benefit, but does not establish a precision-specific
advantage over equally optimized BF16. Peak allocation falls by approximately
1.12 GiB for both fused modes.

Actual run order was BF16, MXFP8, fused MXFP8, fused MXFP8, MXFP8, BF16,
then two fused BF16 controls. The control was added after the initial comparison;
it was not interleaved. The script now includes all four modes in reversed rounds
for future experiments. Raw logs and final losses are in
[comparison/summary.md](comparison/summary.md). Two short runs with one seed and
8,192 evaluation tokens per evaluation do not establish convergence parity.

## Correctness and isolated measurement

`check_activation.py` passed exact packed-byte, scale, reconstruction, output,
input-gradient, and weight-gradient comparisons, including stochastic-rounding
RNG replay. BF16 activation and complete MLP gradient comparisons also passed.
The existing MXFP8 regression passed. Results are recorded in the three check
JSON files in this directory.

For an [8192, 3072] BF16 activation, separate ReLU-square plus packing took
1.485 ms; the fused path took 0.551 ms, a 2.69x isolated speedup.

## Profiles

Separate 25-step runs captured three steps beginning at step 20. These are actual
CUDA kernel durations attributed through launch correlation IDs, not wall-clock
training measurements. Nested child ranges must not be added to phase totals.

| Mode | Forward ms/step | Backward ms/step | Optimizer ms/step | Total kernel ms/step |
|---|---:|---:|---:|---:|
| Fused MXFP8 | 106.83 | 206.71 | 59.33 | 372.88 |
| Fused BF16 | 109.94 | 190.96 | 59.26 | 360.15 |

Both traces have zero unmatched kernels and phase sums reconcile to kernel totals.
MXFP8 forward is slightly faster here, while backward costs about 15.8 ms more.
MXFP8 backward packing alone accounts for 23.58 ms/step; its separate activation
derivative costs 7.72 ms/step. Combining that derivative with gradient packing
is a next candidate, although its benefit must be measured. Other backward
packing and conversion work remains.

Profile summaries and operator tables are checked in under `profile-*-fused/`.
Compressed Chrome traces are retained locally and on Spark but excluded from Git.

## Reproduction

Run `spark/check_activation.py`, `spark/check_bf16_activation.py`, and
`spark/check_mxfp8.py` with the CUDA Python environment. Then run
`PYTHON=$PY RUN_DIR=results/new-activation-comparison spark/run_activation_comparison.sh`.
The initial MXFP8 fusion implementation is commit `a6f3a56`; individual logs
record source hashes. The final snapshot also includes the BF16 control.
