# Explicit quantized lookahead buffers on GB10

## What was implemented

A second GEMM kernel has a prologue that quantizes tile 0, a loop that keeps
current and next quantized operands distinct, and a drain that multiplies the
last tile. Both operands and both sets of scales are loop-carried values in
Triton's lowered IR. The source expresses quantization of the next tile before
MMA consumes the current tile. This is explicit register-value buffering, not
a manually allocated shared-memory producer/consumer ring or warp specialization.

The sweep separately varies the kernel `num_stages` and the loop
`tl.range(..., num_stages=...)` setting. The installed Triton documentation says
the former pipelines loads feeding dot operations, whereas the latter attempts
to pipeline most loads. These settings are not proof of conversion/MMA overlap.

The training opt-in is `--pipeline-mxfp8 --buffered-mxfp8`, currently using
BM32/BN64/BK32, four warps, one compiler stage. Existing defaults are unchanged.

## Schedule evidence and its limits

`analyze_buffered_schedule.py` disassembles the compiled production-shape kernels
with CUDA cuobjdump and saves cubin hashes, lowered loop signatures and SASS
excerpts in `schedule-evidence.json`. The buffered IR retains five loop-carried
values: accumulator, current A, A scales, current W, and W scales. The serial
loop retains only its accumulator.

In the buffered SASS, native `QMMA.SF.16832.F32.E4M3.E4M3.E8` instructions are
interleaved with scale encoding arithmetic, e.g. FADD at 0xcaf0/0xcb00 and F2I
at 0xcb30/0xcb60 between QMMA instructions. This is stronger evidence of
instruction scheduling than the earlier observation of `cp.async` alone.
The excerpt does not establish a measured simultaneous execution duration or
show that all next-tile quantization is hidden behind current-tile MMA. Much
conversion work still appears in large groups before MMA. We do not report a
hardware overlap percentage or claim a fully effective conversion/MMA pipeline.

## Timing

Each sweep configuration uses 100 ms warmup and three separate 300 ms timing
windows. Besides the model's six BK128 iterations, we test 32 BK128 iterations
with K4096. Smaller BK32/BK64 tests reduce live buffers and add more iterations.
All successful configurations match the separate-packing reference exactly,
including saved reconstructed operands; resource failures are retained in logs.

A confirmation uses identical BM32/BN128/BK32/four-warps/one-stage configurations
for serial and buffered kernels, four alternating forward/reverse measurement
orders, and 100 ms warmup plus 300 ms timing for each path in each order.
Median milliseconds:

| Shape M,N,K | K iterations | Separate pack + GEMM | Serial conversion loop | Buffered lookahead |
|---|---:|---:|---:|---:|
| 8192,3072,768 | 24 | 0.641 | 2.357 | 2.842 |
| 1024,1024,4096 | 128 | 0.503 | 0.512 | 0.511 |

The production shape regresses about 21% versus the same serial tile. The long
loop is effectively tied. More iterations do not reveal a substantial benefit
in this implementation. This is a result for these kernels and shapes, not an
upper bound on what a different pipeline design could achieve.

The full sweep also found some serial small-tile configurations competitive
with the separate path on the K4096 diagnostic. That is a different shape from
the model's first projection. No full-training speedup is inferred.

## Nsight Compute

NCU captures serial and buffered BM16/BN64/BK128/four-warps/one-stage kernels.
These are the original large-buffer diagnostic, not the small-tile confirmation.

| Metric | Serial | Buffered |
|---|---:|---:|
| Profiled duration | 4.24 ms | 5.53 ms |
| Registers/thread | 179 | 255 |
| Achieved occupancy | 16.29% | 14.95% |
| Eligible warps/scheduler | 0.46 | 0.42 |
| L1TEX dependency stalls, average cycles | 2.2 | 2.9 |

Compiler metadata reports 20 spills for that buffered configuration. NCU's
rule describes L1TEX dependency stalls; this does not establish DRAM bandwidth
saturation. The smaller confirmation tile is spill-free (serial 122 registers,
buffered 185). Its regression shows that eliminating spills alone is insufficient.

Whole-MLP output, input/weight gradients, CUDA RNG consumption and fullgraph
compiled forward/backward parity all pass exactly with the buffered training
configuration. No additional full-training comparison was run: the kernel tests
show no gain to validate end to end.

## Reproduce

```sh
$PY spark/bench_buffered_mxfp8.py
$PY spark/bench_buffered_mxfp8.py --small-bk
$PY spark/bench_buffered_paired.py
$PY spark/analyze_buffered_schedule.py
$PY spark/check_pipelined_mxfp8.py --buffered
sudo -n /opt/nvidia/nsight-compute/2025.3.1/ncu --profile-from-start off \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --section SchedulerStats --section WarpStateStats \
  -o results/buffered-mxfp8/serial-buffered \
  "$PY" spark/probe_buffered_mxfp8_ncu.py
```

Redirect each benchmark's stdout to its corresponding JSONL file. Assembly and
binary profiler artifacts remain on Spark and are ignored by Git; text evidence,
measurements, NCU details and test output are checked in.
