# First-projection MXFP8 GEMM with a fused activation-packing epilogue

The prototype uses native SM12x block-scaled `mma.sync` to compute the first MLP
projection, rounds the accumulator to BF16, applies ReLU-square with the original
BF16 rounding boundary, computes 1x32 scales, and writes packed MXFP8 data and
scales for the second projection. It uses no TMEM and no BF16 GEMM emulation.

Training still saves the BF16 preactivation for the derivative and reconstructed
quantized activation for the weight gradient. The saved preactivation write is
therefore retained; its separate read and the activation-pack launch are removed.
First-projection input/weight packing and second-projection GEMM remain separate.
The backward GEMMs, stochastic rounding and RNG consumption order are unchanged.
This is epilogue fusion, not conversion of both operands inside the GEMM mainloop.

## Validation

Checks cover M=1,129,256,8192; N=128,512,1024,3072; K=96,256,768, including
partial rows and scale padding. Observed BF16 preactivations matched native
`torch._scaled_mm` numerically exactly for all four cases. Packed bytes, scale
bytes (including padding), and backward reconstructions match a separate pack
of the same preactivation exactly. Whole-MLP outputs and input/weight gradients
also matched numerically exactly in the stochastic-rounding replay check, with
identical RNG state consumption.

The GEMM check allows relative L2 error below 0.003 and the whole-MLP comparison
below 0.015 to accommodate accumulation-order differences; observed errors were
all zero. Exact agreement on these inputs is not a universal bitwise contract
with the cuBLAS/CUTLASS implementation for every shape and input.

Emitted PTX contains native `mma.sync.aligned.m16n8k32...kind::mxf8f6f4.block_scale`
with E4M3 inputs/E8M0 scales, 30 static `cp.async` instruction occurrences, no
`tcgen05` instructions, 168 registers per thread and zero spills for the checked
production specialization. Async copies establish load pipelining; this does not
claim that the epilogue's conversion overlaps its own tile's preceding MMA.

## Isolated tile sweep

For M=8192, N=3072, K=768, quantized first-projection operands are prepared before
timing. The reference times native GEMM plus our existing fused activation pack;
the prototype times the fused GEMM/epilogue. Both produce the four training
outputs: preactivation, MXFP8 activation, scales, and reconstructed activation.

Ten tile/stage/warp configurations were tested. Best: BM=32, BN=128, BK=128,
4 warps, 2 stages, also the integration default.

| Path | GPU time |
|---|---:|
| Native GEMM plus separate activation pack | 0.8830 ms |
| Fused GEMM epilogue | 0.8231 ms |

This is a **1.073x isolated speedup**, saving about **0.060 ms per MLP**.
Twelve MLPs save only about 0.72 ms per training step if that isolated difference
transfers directly. It does not predict a large end-to-end speedup.
`tiles.jsonl` records every candidate, including slower configurations.

## Full-training method

The same 123,568,128-parameter model, batch 32, sequence 256, seed 42, FineWeb,
100 steps, Muon and fused AdamW. Compare compiled BF16, compiled current MXFP8,
and compiled epilogue MXFP8 twice in reversed order. Every path uses the existing
fused activation setting; epilogue fusion is an additional opt-in flag. Both MXFP8
paths use stochastic backward rounding and all 24 MLP projections in MXFP8.

Throughput includes optimizer work and excludes input preparation, evaluation,
and the first five steps. Compilation startup and validation are excluded.
Individual step durations and compiler counters remain in the logs. Short runs
cannot establish convergence equivalence or time to target loss.

## Reproduce

```sh
$PY spark/check_gemm_epilogue.py
$PY spark/bench_gemm_epilogue.py
PYTHON=$PY RUN_DIR=results/new-epilogue spark/run_gemm_epilogue_comparison.sh
PYTHON=$PY RUN_DIR=results/new-epilogue-nsys spark/run_gemm_epilogue_nsys.sh
```

Use `--fuse-mlp-epilogue` with native MXFP8, Triton packing and `--compile` on
`spark/train.py`. It remains off by default. Default shapes require K divisible
by 32 and first-projection output width divisible by 128.

## Measured training and profiles

Two 100-step runs per path gave mean tokens/s: compiled BF16 38,195;
current compiled MXFP8 36,160; epilogue MXFP8 36,011. The epilogue change
has no established training benefit (mean -0.41% versus current MXFP8).
The isolated 0.883→0.823 ms saving does not establish a full-training gain.

Nsight Systems attributed every captured kernel across three steps. Current vs
epilogue forward GPU kernel time was 62.68 vs 61.88 ms; backward 99.16 vs
100.25 ms; optimizer 57.98 vs 58.33 ms. Forward launches fell 318→306 per step.
Nsight Compute measured the isolated fused kernel at 924 us under replay;
SM throughput 19.91%, L2 throughput 50.20%. Replay timing differs from the
unprofiled microbenchmark. This is output epilogue fusion; it does not implement
operand quantization inside the GEMM loop.
