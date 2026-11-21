# Modded-nanogpt experiments on DGX Spark

Single-GB10 experiments with native NVFP4 forward matrix multiplication and
MXFP8 backward matrix multiplication, inspired by Thinking Machines Lab's
[End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/).

## Source and scope

`upstream/` is KellerJordan/modded-nanogpt, checked out at
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`.
`spark/record_model.py` contains definitions extracted from the
`2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt` record
under its MIT license (`LICENSE.upstream`). The historical model uses RoPE,
QK RMS normalization, ReLU-squared MLPs, tied embeddings and Muon.

The runner adapts that model to one GPU and small batches. It uses fixed learning
rates (AdamW .003 for the tied embedding/head, Muon .02 for blocks). These are
short diagnostic runs, not the current eight-H100 leaderboard recipe or a claim
of reaching its validation target. The default model has 123,568,128 parameters.

## Precision modes

- `bf16`: BF16 autocast, FP32 master parameters and ordinary autograd.
- `fp4-reference-sr`: simulated block quantization followed by BF16 GEMMs.
- `fp4-reference-nearest`: deterministic backward re-quantization ablation.
- `fp4-native-sr`: packed E2M1 forward GEMMs with E4M3 block scales, and
  E4M3 MXFP8 backward GEMMs with E8M0 block scales and stochastic rounding.
- `fp4-native-nearest`: the native deterministic re-quantization ablation.

Only the first five-sixths of MLP layers are replaced. Attention, embedding/head,
and the final one-sixth of layers retain the BF16 baseline. NVFP4 uses 1x16
blocks, per-token activation outer scales and per-weight-tensor outer scales.
Backward re-quantizes the saved **quantized forward values** after transposition,
using 1x32 MXFP8 blocks. Native GEMMs call `torch._scaled_mm` with actual packed
FP4 or FP8 operands; unsupported execution raises an error, with no BF16 fallback.

The native implementation supports eager PyTorch packing and fused Triton packing
(`--pack-backend triton` or `triton-tiled`). The fused backend performs block reductions, quantization,
FP4 bit packing and cuBLAS scale-layout writes in one kernel per operand, with
a two-stage global maximum reduction for FP4 weights. MXFP8 supports strided
transpose inputs directly; `triton-tiled` uses coalesced 16x128 tiles for those
transposes to avoid the row kernel's large-batch regression. It draws rounding seeds from PyTorch's CUDA RNG. Both
backends save dequantized BF16 forward inputs. It does not yet provide
Four Over Six activation scaling, packed saved activations,
or optimized weight caches. It is an experimental port, not an exact reproduction
of Thinking Machines' kernels. This dense model does not exercise MoE dispatch,
S2F6 combine, or distributed communication.

Scale packing follows the [cuBLAS block-scale layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout).
Numerical checks compare nonuniform native results and both gradients with
reference computations. Constant-input probes alone do not validate scale layout.

## Run on Spark

The development machine is `kedar@spark-a681` over Tailscale. Files live in
`/home/kedar/projects/inklings2-mini`. The validated existing interpreter is:

```sh
PY=/home/kedar/projects/lawzero-reproduction/.venv/bin/python
cd /home/kedar/projects/inklings2-mini
$PY spark/probe.py --output spark-report.json
$PY spark/probe_blocks.py
$PY spark/check_precision.py
$PY spark/check_native.py
$PY spark/train.py --precision bf16 --steps 200 --output results/baseline.jsonl
$PY spark/train.py --precision fp4-native-sr --pack-backend triton-tiled --steps 200 --output results/native-sr.jsonl
$PY spark/train.py --precision fp4-native-nearest --pack-backend triton-tiled --steps 200 --output results/native-nearest.jsonl
$PY spark/summarize.py results/baseline.jsonl results/native-sr.jsonl results/native-nearest.jsonl
```

No packages in that existing environment were changed. It contains PyTorch
2.14.0+cu130 and Triton 3.8.0. A fresh environment needs CUDA-enabled **aarch64**
PyTorch with GB10 support; a CPU wheel or a generic incompatible CUDA build will
not work. The host driver is 580.178.04; compute capability is 12.1.

The input files are the GPT-2-tokenized `kjj0/fineweb10B-gpt2` dataset's
`fineweb_train_000001.bin` and `fineweb_val_000000.bin`, downloaded into `data/`
on Spark. The runner verifies shard headers and lengths. Supply the same data,
seed, model dimensions, batch and sequence length across precision modes.

Logs contain configuration, parameter count, training losses, validation losses,
training duration and peak PyTorch allocated GPU memory. Steady throughput
excludes the first five optimizer steps and excludes input loading and validation;
training duration includes optimizer compilation. Initial validation runs before
training. Thus this is training-step throughput, not full job wall time. Validation
uses a fixed RNG fork so it does not change the training quantization RNG stream.

Start with short runs; assess convergence with longer runs, larger batches and
multiple seeds before treating a small loss difference as meaningful. No result
here establishes the post's 8%/20% speedups on Spark.

## Fused packing and larger batches

```sh
$PY spark/check_pack.py --tiled
$PY spark/check_native.py --pack-backend triton-tiled
$PY spark/bench_layer.py --pack-backend triton-tiled
PYTHON=$PY RUN_DIR=results/my-fused-comparison spark/run_fused_comparison.sh
```

The comparison script defaults to `triton-tiled` (override `PACK_BACKEND`) and
runs batches 4, 16 and 32 for 100 steps per precision,
using a constant 8,192-token validation set. Compare precision modes **within**
a batch: different batch sizes see different training-token counts at 100 steps.
Runtime logs include hashes of the model, runner and precision implementations.

`check_pack.py` checks exact eager/fused deterministic parity across BF16 and
FP32 inputs, zero rows, outliers, non-tile-aligned dimensions, transpose strides,
stochastic-rounding expectation, fresh noise and RNG-state replay. The native
GEMM checks additionally validate both backward matrix multiplications.
FP4 midpoint rounding follows the eager implementation's toward-zero tie rule;
this is not a bitwise reproduction of a hardware nearest-even converter.

Latest measured results: [fused packing and batch scaling](results/fused/REPORT.md).

## MXFP8 baseline and profiling

`mxfp8-native-sr` uses native MXFP8 forward GEMMs and stochastic MXFP8
backward re-quantization. Forward rounding is nearest. The fused packer writes
reconstructed forward inputs for backward in the same kernel; the backward
estimator uses those quantized forward values, as in the FP4 path.

Use `--tail-precision mxfp8` for the three-way comparison: the MXFP8 mode then
converts every MLP, while the FP4 mode uses NVFP4 in the first five-sixths of MLPs
and MXFP8 in the final sixth. BF16 uses ordinary autograd everywhere. Attention
and the tied embedding/head stay in BF16 autocast for all modes. Older runs used
`--tail-precision bf16` (still the default) and remain reproducible.

```sh
$PY spark/check_mxfp8.py --pack-backend triton-tiled
PYTHON=$PY RUN_DIR=results/my-three-way spark/run_precision_comparison.sh
$PY spark/train.py --precision mxfp8-native-sr --tail-precision mxfp8   --pack-backend triton-tiled --batch 32 --eval-batches 1 --steps 25   --profile-start 20 --profile-steps 3 --profile-dir results/profile-mxfp8   --output results/profile-mxfp8.jsonl
```

Profiles contain a compressed Chrome trace, raw operator statistics and a
kernel-attribution summary. The analysis excludes GPU annotation ranges and
attributes actual CUDA kernels to CPU ranges using launch correlation IDs.
Nested ranges overlap, so parent and child totals must not be summed. Profiling
introduces overhead; use the separate unprofiled runs for throughput claims.

Code/results snapshot before MXFP8: commit `67290e4`, tag `spark-fused-v1`.
A Git recovery bundle is stored on Spark in `snapshots/`.
