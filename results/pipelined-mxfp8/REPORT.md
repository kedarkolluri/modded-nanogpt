# Operand quantization inside MXFP8 GEMM on Spark

Implemented `--pipeline-mxfp8` as an opt-in first-projection MLP experiment.
The kernel reads the original FP32/BF16 operands, computes nearest E4M3 values
and E8M0 scales in groups of 32 inside each K-loop iteration, then issues
native SM121 block-scaled MMA. It saves BF16 reconstructed operands once
per element for the unchanged stochastic backward. No TMEM is required.
The second projection and backward retain the existing native MXFP8 path.

## Correctness

Forward GEMM and reconstructed operands match separate packing plus native
GEMM exactly on (M,N,K)=(1,128,96), (129,512,256), and (8192,3072,768).
Whole-MLP output, input gradient, both weight gradients and CUDA RNG state
match the existing implementation exactly. Fullgraph compiled nearest-rounding
forward/backward match eager exactly. Tests use nonzero random weights.

## Kernel measurements

Production-shape separate packing plus native GEMM: 0.639 ms. Representative
fused operand-loop configurations (BM,BN,BK,warps,stages):

| Configuration | ms | Registers/thread | Spills | PTX cp.async occurrences |
|---|---:|---:|---:|---:|
| 32,64,128,4,2 | 6.047 | 211 | 0 | 54 |
| 32,128,128,8,1 | 2.808 | 196 | 0 | 0 |
| 16,128,128,8,2 | 7.998 | 170 | 0 | 42 |

The default uses the first configuration to exercise compiler staging without
spilling. The fastest tested configuration uses one stage and is still 4.4x
slower than separate packing. Several larger staged tiles exceed GB10's
101,376-byte per-block shared-memory limit. Raw results include these failures.

This implementation repeats activation quantization for each N output tile and
weight quantization for each M output tile. At the default production grid,
activation conversion repeats 48 times and weight conversion 256 times.
FP32 source reads and reduction/conversion work replace globally reusable
packed operands. Register pressure and shared-memory staging also limit tiling.
These are structural costs visible in the implementation and compiler metadata;
we have not attributed precise runtime fractions to them with hardware counters.

PTX confirms native MXFP8 MMA and asynchronous copies in staged variants.
That establishes an operand-loop implementation with compiler staging, **not
proof that quantization instructions overlap MMA execution**. No instruction
schedule or measured overlap percentage is claimed. More stages did not produce
a speedup. Shared-memory reuse or prepacking the reusable weight operand are
possible next experiments; neither is implemented here.

## Training

The compiled 123.6M-parameter model, batch 32, sequence 256, seed 42, 20 steps,
with the first five excluded, reached 28,041 tokens/s using this experiment.
There were zero graph breaks and losses remained finite. This is a short
performance check, not convergence validation. An initial process failed CUDA
context allocation before running the model; a fresh process completed.
A subsequent BF16 comparison also failed context allocation before training,
so no fresh paired baseline is claimed. Earlier two-run 100-step means were
38,195 tokens/s for BF16 and 36,160 for current MXFP8; these are historical
comparisons, not matched runs from this session.

## Reproduce

```sh
$PY spark/check_pipelined_mxfp8.py
$PY spark/bench_pipelined_mxfp8.py
PYTHON=$PY RUN_DIR=results/new-operand-loop spark/run_operand_pipeline_comparison.sh
```

The comparison script defaults to two reversed-order 100-step runs per path.
The measurements in this report use the separately recorded short runs and
microbenchmark sweep, not an executed 100-step comparison script.
