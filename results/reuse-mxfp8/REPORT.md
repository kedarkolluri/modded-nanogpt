# Reusing packed weights across GEMM output tiles on Spark

## Change

The first MLP projection now has an optional path that packs its weight once per
forward call, then runs an activation-quantizing native MXFP8 GEMM. Every CTA
reads the same packed weight and E8M0 scales. No packed state survives across
optimizer updates. Backward uses the same saved reconstructed BF16 operands,
stochastic rounding and existing native GEMMs as before.

Wider output tiles also reuse each converted activation tile across more output
columns. The selected production configuration is BM16/BN512/BK64, eight warps,
one compiler stage, without explicit lookahead. It converts each activation
block six times across N=3072, versus 24 times in the previous BN128 experiment.
Weight conversion happens once, versus once for every M output tile previously.
The improvement combines weight reuse and tile tuning; it does not isolate the
benefit of weight reuse alone.

`--pipeline-mxfp8 --reuse-mxfp8-weights` enables this path. Adding
`--buffered-mxfp8` uses a smaller BM32/BN128/BK64/four-warp lookahead variant.
The buffered variants did not win the sweep and can spill at larger tile sizes.
Nothing in this result establishes a measured conversion/MMA overlap percentage.

## Correctness

Successful configurations match separate native packing plus native GEMM exactly
for output and both saved reconstructed operands. Tests include masked M/N
boundaries, K=96 tails, the production (8192,3072,768) shape, and the wider
(1024,1024,4096) diagnostic. Native block-scaled MMA is checked in PTX.

Whole-MLP output, input gradient, both weight gradients and CUDA RNG state match
exactly with nonzero random weights. Fullgraph compiled nearest-rounding forward
and backward match eager exactly for the default tuned path and the smaller
buffered path. Logs retain all resource failures in the tile sweep.

## Fresh-seed confirmation

Tile selection uses seed 42. Confirmation uses seed 43, four alternating
forward/reverse timing orders, 100 ms warmup and 300 ms timing per path per order.
Every path produces the same output and saved reconstructed operands. Timings
include fresh weight packing on every call; the separate path includes both
operand packs. Median milliseconds:

| Shape M,N,K | Separate pack + native GEMM | Both operands quantized inside loop | Weight reuse + wider tile |
|---|---:|---:|---:|
| 8192,3072,768 | 0.638 | 2.442 | 1.171 |
| 1024,1024,4096 | 0.507 | 0.498 | 0.404 |

The production-shape fused path improves 2.09x relative to the previous fused
path, but still takes 1.83x the separate native path's time. The K4096 diagnostic
uses BM32/BN256/BK64/four-warps/one-stage and takes 20.3% less time than separate
packing, a 1.25x throughput increase for that operation. This is not evidence of
a BF16 or whole-training speedup.

All three timing windows per sweep configuration and all four confirmation
rounds are retained in the JSONL files. `kernel_only_ms` in the sweep excludes
weight packing and is diagnostic only; it is not the comparison above.

## Full training method

Compiled 123,568,128-parameter model, batch 32, sequence 256, seed 42, FineWeb,
Muon plus fused AdamW. BF16, current MXFP8 and weight-reuse MXFP8 use identical
settings, including the existing fused activation path. MXFP8 uses stochastic
backward rounding and both projections in all 12 MLPs use MXFP8.

One 100-step run per path, first five steps excluded, validation at steps 0/50/100.
Input preparation, evaluation and compilation startup are excluded from steady
throughput. This is a performance comparison, not convergence validation or a
multi-run statistical estimate.

An earlier 20-step BF16 check had several slow early steps. Other startup
attempts failed CUDA context allocation before training; their logs are retained.
The runner now forces a tiny CUDA allocation before constructing the CPU model.
The comparison uses that same startup sequence for all paths. This is not a
confirmed diagnosis of the earlier driver/context allocation failure.

## Full training result

| Path | Tokens/s | Peak allocated GPU memory |
|---|---:|---:|
| Compiled BF16 | 38,182 | 5.08 GiB |
| Current compiled MXFP8 | 35,888 | 4.69 GiB |
| Weight-reuse MXFP8 | 35,117 | 4.69 GiB |

All three 100-step runs completed with finite losses and zero graph breaks.
Weight reuse is 2.1% below current MXFP8 throughput and 8.0% below BF16 in this
single comparison. It improves over the earlier all-in-loop prototype but does
not improve over the established native MXFP8 baseline. The wider diagnostic's
isolated benefit does not apply to this model's first-projection shape.

The startup sequence with a tiny CUDA allocation completed these three runs;
this does not establish why earlier context allocations failed. No driver or
system memory settings were changed.

## Reproduce

```sh
$PY spark/bench_reuse_mxfp8.py > results/reuse-mxfp8/sweep.jsonl
$PY spark/bench_reuse_mxfp8.py --wide > results/reuse-mxfp8/wide.jsonl
$PY spark/bench_reuse_paired.py > results/reuse-mxfp8/paired.jsonl
$PY spark/check_pipelined_mxfp8.py --reuse-weights
$PY spark/check_pipelined_mxfp8.py --reuse-weights --buffered
PYTHON=$PY RUN_DIR=results/new-reuse spark/run_weight_reuse_comparison.sh
```

Create the result directory before redirecting benchmark stdout. Use a fresh run
directory for training; JSONL output is append-only.
