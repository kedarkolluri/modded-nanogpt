# Nsight Systems and Nsight Compute on Spark

Captured both fused MXFP8 and fused BF16, serially on the GB10. Systems version
2025.3.2; Compute version 2025.3.1. These are separate from the earlier PyTorch
Profiler captures. Original `.nsys-rep` and `.ncu-rep` files are retained here and
on Spark under `results/nsight/`; binary reports and SQLite exports are ignored
by Git. CSV exports, detailed counter reports, and analysis JSON are checked in.

## Systems: complete training

Same 124M model, batch 32, sequence 256, seed 42. Three training steps captured
beginning at step 20 through CUDA profiler start/stop APIs. Includes attention,
head, MLPs and optimizers. Sampling and CPU context switches disabled; CUDA and
NVTX enabled. No PyTorch Profiler was active.

| Kernel measurement per step | Fused MXFP8 | Fused BF16 |
|---|---:|---:|
| Forward | 108.02 ms | 109.59 ms |
| Backward | 202.88 ms | 190.51 ms |
| Optimizer | 58.39 ms | 58.90 ms |
| Kernel sum | 369.29 ms | 359.00 ms |
| Backward kernel count | 962 | 770 |
| No kernel executing, within captured span | 7.28 ms | 6.93 ms |

Backward contributes the gap. The no-kernel intervals include step boundaries,
memory operations and synchronization; they are not all CPU launch stalls.
Profiler timings are diagnostic; use the separate unprofiled comparison for
throughput claims.

**Attribution caveat:** the default Nsight NVTX GPU projection incorrectly shows
almost no backward work because PyTorch launches backward kernels on an autograd
worker thread. `spark/analyze_nsys.py` correlates each kernel to its CUDA runtime
launch, then uses the serial phase CPU time windows across threads. Both captures
have zero unmatched kernels, and phase sums reconcile with all captured kernels.
Do not use the default `nvtx_gpu_proj_sum` backward total as a training measurement.

## Compute: production-shape MLP

One MLP with input `[32,256,768]` and intermediate width 3072, ten warmup
forward/backward iterations, then one captured iteration. This isolates packing,
activation, GEMMs and casts; it excludes attention and optimizer. We collected
SpeedOfLight, MemoryWorkloadAnalysis, Occupancy and basic launch statistics.
Kernel replay used ten passes and default cache control. The probe uses random
weights and gradients, not a training checkpoint. Replay durations are not
end-to-end training timings.

| Sum of replayed kernel durations | Fused MXFP8 | Fused BF16 |
|---|---:|---:|
| Six GEMMs | 2.144 ms | 3.369 ms |
| MXFP8 packing (including fused forward activation) | 3.056 ms | — |
| Separate activation kernels | 0.663 ms | 1.134 ms |
| Other kernels, including casts and RNG | 0.353 ms | 0.525 ms |
| Total | 6.216 ms | 5.028 ms |

The lower-precision GEMMs save about 1.225 ms here, but packing/conversion costs
consume the gain. MXFP8 GEMMs report tensor-pipe active throughput of 44.6–56.9%
of sustained elapsed peak; BF16 GEMMs report 58.6–76.4%. This counter is not the
fraction of advertised FLOPs achieved.

The fused forward activation pack takes 0.589 ms, with 7.69% SM throughput and
16.57% reported memory throughput. Large stochastic transpose packs take about
0.546–0.550 ms each and report about 64–65% SM throughput. These observations
suggest different bottlenecks across packing variants; a blanket claim that all
packing is bandwidth-saturated is unsupported. This GB10 report does not expose
DRAM bandwidth counters, so the memory percentage must not be interpreted as
LPDDR bandwidth utilization.

Next candidates are avoiding repeated operand repacking and fusing the activation
derivative with gradient packing. Scheduler/warp-stall counter collection would
help explain the low-throughput forward pack before tuning its block layout.

Normal-user counter collection returned ERR_NVGPUCTRPERM. Collection succeeded
using existing passwordless sudo for the profiler process; no driver security
settings were changed. Both probe runs finished with finite input gradients.

## Reproduce

From the project root on Spark, set `PYTHON` to the CUDA environment interpreter
and run `spark/run_nsight.sh` with a fresh `RUN_DIR`. The script runs all GPU
captures serially. Raw CSV exports and `*-ncu-selected.csv` retain counter values;
`*-ncu-details.txt` includes tool diagnostics and launch/occupancy details.
