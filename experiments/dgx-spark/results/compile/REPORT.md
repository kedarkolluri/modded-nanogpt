# Compiled BF16 and native MXFP8 on DGX Spark

Both model paths now support `torch.compile` with Inductor, `fullgraph=True`,
`dynamic=False`, and the default compile mode. The optimizer loop remains as
before; Muon's Newton–Schulz operation was already compiled in the eager baseline.
CUDA Graph replay and max-autotune mode are not enabled in this comparison.

Two compilation compatibility fixes preserve the packing/GEMM mathematics:
MXFP8 now contains its own backward body instead of calling another autograd
class's backward, and the tiled Triton transpose pack receives its tile constants
explicitly so Dynamo can analyze its arguments and mutations.

## Method

Same 123,568,128-parameter ModernArch model, batch 32, sequence 256, seed 42,
100 training steps per run, FineWeb data, Muon plus fused AdamW, and native MXFP8
with stochastic backward rounding. Evaluation every 50 steps on 8,192 tokens.
Each of six modes runs twice, reversing the complete order in the second round.
All GPU runs are serial. No TorchAO or package changes are involved.

Steady throughput includes forward, backward and optimizer, excludes input
preparation and validation, and excludes the first five training steps. All
individual step durations are recorded, along with run wall time, peak allocation,
source hashes and Dynamo counters. Run wall time includes compilation/startup;
first-step timing includes training/backward compilation and optimizer warmup.
The compiler cache was reused across runs, so startup times are not clean cold
compilation benchmarks.

## Results

| Mode | Repeat 1 tokens/s | Repeat 2 tokens/s | Mean tokens/s | Peak allocated GiB |
|---|---:|---:|---:|---:|
| BF16, compiled ordinary activation | 38,049 | 37,503 | 37,776 | 5.08 |
| BF16, compiled hand-fused activation | 37,925 | 37,771 | 37,848 | 5.08 |
| MXFP8, compiled ordinary activation | 33,245 | 33,196 | 33,220 | 4.69 |
| MXFP8, compiled hand-fused activation | 36,116 | 34,968 | 35,542 | 4.69 |
| BF16, eager hand-fused control | 22,493 | 22,370 | 22,432 | 9.41 |
| MXFP8, eager hand-fused control | 21,779 | 21,859 | 21,819 | 9.42 |

Comparing hand-fused modes, compilation improves BF16 throughput **68.7%** and
MXFP8 throughput **62.9%**. Compiled MXFP8 remains **6.1% below compiled BF16**,
but uses about **7.7% less peak allocated GPU memory**. Both modes benefit
substantially from compiling the whole model. Hand fusion still helps compiled
MXFP8; compiled BF16 performs similarly with either activation implementation.

All twelve runs completed with finite training losses, and all eight compiled
runs report zero graph breaks. Final validation losses are retained in
[comparison/summary.md](comparison/summary.md). The second fused compiled MXFP8
run was 3.2% slower than its first repeat; its steady median was 0.2332 s and
maximum 0.2515 s, compared with 0.2267 s and 0.2325 s in the first run. There is
no large compilation stall in these measured steps. Two repeats do not identify
the cause of that variability; both are included in the mean.

## Correctness and compiler checks

`check_compile.py` passed exact output, input-gradient and both weight-gradient
comparisons for eager versus compiled fused MLPs in BF16 and nearest-rounding
MXFP8. The MXFP8 packing/backward regression also passed. Real-model MXFP8 runs
retain stochastic rounding. Compiled random-number generation may use a different
sequence from eager, so training trajectories are not expected to be bitwise
identical. These short runs do not establish convergence parity.

`fullgraph=True` rejects graph breaks. Compiler diagnostics are recorded with
`TORCH_LOGS=graph_breaks,recompiles`. Separate initial evaluation/training graphs,
rotary-cache state and Muon matrix shapes cause startup specializations. Steady
step durations must be checked for later compilation stalls.

## Reproduction

On Spark, set `PYTHON` to the CUDA interpreter and use a fresh output directory:

```sh
$PYTHON spark/check_compile.py
PYTHON=$PYTHON RUN_DIR=results/new-compile-comparison spark/run_compile_comparison.sh
```

Use `--compile` on `spark/train.py` to select this model compilation path.
The hand-fused activation is separately controlled by `--fuse-activation`.

## Compiled Nsight Systems captures

Separate 24-step runs captured three warmed-up steps beginning at step 20, using
both hand-fused compiled paths. As before, CUDA launch correlation and serial CPU
phase windows account for autograd worker-thread launches. Both captures have
zero unmatched kernels, and phase sums reconcile with all captured kernel time.

| Kernel measurement per step | Compiled BF16 | Compiled MXFP8 |
|---|---:|---:|
| Forward | 56.30 ms | 63.55 ms |
| Backward | 97.35 ms | 100.03 ms |
| Optimizer | 58.86 ms | 58.14 ms |
| Total kernel sum | 212.51 ms | 221.72 ms |
| Forward kernel count | 209 | 318 |
| Backward kernel count | 307 | 404 |
| Optimizer kernel count | 2,318 | 2,318 |

For reference, the eager fused captures had 536 forward kernels in both modes,
and 770/962 backward kernels for BF16/MXFP8. The net extra backward launches for
MXFP8 therefore fall from 192 to 97 after compilation. The remaining compiled
time gap in these captures is primarily forward (7.25 ms), with a smaller
backward difference (2.68 ms). Earlier eager-only attribution should not be
carried forward unchanged.

The compiled kernel tables contain fused RMS normalization, casting, and
cross-entropy forward/backward kernels. Custom MXFP8 packing kernels remain;
across forward and backward they total 32.12 ms per step in this capture.
The optimizer remains a significant common cost. Profile durations diagnose
execution; throughput claims use the twelve unprofiled runs above.

CSV tables, analysis JSON and binary reports are under `nsys/`. The binaries and
SQLite databases are retained locally and on Spark and excluded from Git.
Reproduce with `PYTHON=$PYTHON RUN_DIR=results/new-compiled-nsys spark/run_compiled_nsys.sh`.
