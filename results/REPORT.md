# First DGX Spark experiment — 2026-11-20

Native NVFP4 forward and MXFP8 backward training executed successfully on the
user's NVIDIA GB10 (SM 12.1). The first implementation is slower than BF16.

The experiment adapts the historical modded-nanogpt ModernArch model to one GPU:
123,568,128 parameters, batch 4, sequence length 256, 200 optimizer steps,
204,800 training tokens, seed 42, and 8,192 fixed validation tokens. MLPs in the
first ten of twelve layers use the selected precision recipe. All runs use the
same initial parameters, token order, optimizer settings and validation input.
GPU algorithms are not forced deterministic; same-seed repetitions can differ.

| Precision | Final validation loss | Steady training tokens/s | Peak allocated GiB |
|---|---:|---:|---:|
| BF16 | 8.5368 | 9,124 | 2.53 |
| Native NVFP4 / MXFP8, stochastic rounding | 8.5441 | 3,909 | 2.54 |
| Native NVFP4 / MXFP8, nearest rounding | 8.5582 | 5,495 | 2.54 |

The stochastic variant runs at 43% of baseline throughput (about 2.33x the step
time). This diagnostic demonstrates execution and finite gradients, not
convergence to the race's 3.28 loss target. Validation improves sharply initially
but rises between steps 100 and 200 in these small-batch, fixed-learning-rate
runs. Longer training needs an appropriate batch size and learning-rate schedule.
One seed and this token budget cannot establish an accuracy advantage for either
rounding method.

An isolated GEMM at M=1024, N=3072, K=768 measured:

| Operation | Milliseconds |
|---|---:|
| BF16 GEMM | 0.0541 |
| Native NVFP4 GEMM, FP32 output | 0.0514 |
| Eager FP4 packing for both operands | 1.8984 |
| Native MXFP8 GEMM, BF16 output | 0.0334 |
| Eager stochastic MXFP8 packing for both operands | 1.6376 |

GEMM timings exclude packing and outer-scale application. The packing routines
also produce saved dequantized values where needed. This result points to eager
packing overhead as a major optimization target. Native FP4's GEMM advantage is
small at this particular shape/output dtype, so fusion alone does not guarantee
an end-to-end win. Other shapes and batches need measurement.

Correctness checks passed on Spark: zero preservation, stochastic-rounding
expectation, exact representable values, gradient shapes/finiteness, nonuniform
scale layouts, and both native backward GEMMs against reference computations.
Native GEMM relative errors against dequantized references were below 0.004.
The constant-input block probes returned exactly the expected result.

The implementation retains FP32 master weights and optimizer state, and saves
BF16 reconstructed activations for backward. It therefore has no demonstrated
memory saving. Four Over Six, fused quantization/packing and packed saved
activations remain unimplemented. This dense-model experiment does not test
MoE communications or Thinking Machines' proprietary GB300 kernel stack.

The next useful experiments are fused Triton packing, separate measurements of
weight/activation quantization, and larger batch sizes. A longer matched-seed
convergence comparison should follow once the implementation overhead is reduced.

Raw evidence: `native-comparison-*.jsonl`, `layer-benchmark.json`,
`check-native.json`, `check-precision.json`, and `block-report.json`.
The earlier `diagnostic-*` and `gpt124m-*` runs use the BF16-GEMM numerical
reference backend; their timings are not native low-precision performance.
