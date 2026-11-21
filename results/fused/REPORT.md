# Fused packing and batch scaling on DGX Spark

2026-11-20. Fused Triton quantization/packing works on GB10 and substantially
improves the initial native-FP4 implementation. BF16 remains the faster training
mode for the tested 124M model.

## Training measurements

All runs use the historical modded-nanogpt ModernArch adaptation, 123,568,128
parameters, sequence length 256, seed 42 and 100 optimizer steps. The final two
of twelve layers, attention and tied embedding/head retain BF16 autocast. Other
MLPs use native NVFP4 forward and stochastic MXFP8 backward GEMMs. Precision
modes within each batch see identical token sequences and initial parameters.
Each validation uses the same first 8,192 validation tokens. Different batches
see different training-token counts at step 100, so compare precision modes
within a batch when assessing loss.

| Batch | Training tokens | BF16 tokens/s | Fused row tokens/s | Fused tiled tokens/s | Tiled / BF16 throughput |
|---|---:|---:|---:|---:|---:|
| 4 | 102,400 | 9,146 | 8,530 | 8,623 | 94.3% |
| 16 | 409,600 | 16,529 | 14,790 | 15,311 | 92.6% |
| 32 | 819,200 | 18,239 | 12,173 | 16,844 | 92.3% |

| Batch | BF16 final validation loss | Row final loss | Tiled final loss |
|---|---:|---:|---:|
| 4 | 8.0037 | 8.0134 | 8.0551 |
| 16 | 7.4715 | 7.4474 | 7.4601 |
| 32 | 7.1871 | 7.1904 | 7.2039 |

Training-step throughput excludes the first five steps, data preparation and
validation, and includes the optimizer. These are single-seed diagnostic runs,
with fixed learning rates and small batches relative to the race recipe. GPU
algorithms are not forced deterministic. Loss differences do not establish an
accuracy advantage or convergence to the 3.28 race target. No memory reduction
was established: peak allocated memory remains about 2.53/5.97/10.54 GiB for
batches 4/16/32, because master weights and reconstructed saved activations are
retained.

The previous eager native implementation measured about 3,909 tokens/s at batch
4 over 200 steps. The new tiled implementation measures about 8,623 over 100
steps, roughly 2.2x that earlier throughput. The fresh within-batch BF16
comparisons above are the primary evidence; the earlier run has a different
step count and implementation version.

## Kernel changes and measurements

`spark/triton_pack.py` fuses block maxima, quantization, FP4 nibble packing,
dequantized values for backward, and cuBLAS scale swizzling. Weight outer scales
use a separate two-stage absolute-maximum reduction. Activation outer scales are
computed per token in the packing kernel. MXFP8 rounding seeds come from
PyTorch's CUDA RNG so RNG-state forks/replay remain supported.

At M=1024, N=3072, K=768, the paired operand pack benchmark measured:

| Packing operation | Eager ms | Fused ms | Reduction in time |
|---|---:|---:|---:|
| NVFP4 | 1.8516 | 0.0582 | 31.8x |
| Stochastic MXFP8 | 1.6013 | 0.0651 | 24.6x |

These are pack-only timings. The native GEMM does not include packing or outer
scale application; a kernel speedup does not imply the same training speedup.
The layer benchmark uses BF16 synthetic operands; actual training also packs
FP32 master weights.

The initial row kernel regressed at batch 32 during transpose packing.
`spark/triton_pack_tiled.py` uses 16x128 tiles for strided transpose inputs,
coalescing loads and reducing each program's working set. It retains the row
kernel for contiguous inputs. For a transposed BF16 input of logical shape
768x8192, stochastic MXFP8 packing dropped from 1.9995 ms to 0.1085 ms (18.4x).
This improved batch-32 training from 12,173 to 16,844 tokens/s, while still
falling short of BF16's 18,239. The remaining gap needs a full-step profile;
these measurements do not assign it to one unmeasured cause.

## Validation

Passed on the Spark:

- Exact eager/fused deterministic parity for ten BF16/FP32 shape cases,
  including zero rows, outliers, padded rows and padded scale columns.
- Direct transpose packing, stochastic expectation and RNG-state replay.
- Bitwise row/tiled parity in nearest and stochastic modes at the three
  tested transpose sizes, with the same RNG state.
- Native forward and both native backward GEMMs against dequantized references;
  relative errors remain below 0.004.
- Finite losses through all nine 100-step training comparisons.

FP4 midpoint rounding follows the eager implementation's toward-zero tie rule.
It does not claim bitwise identity with a nearest-even hardware converter.
The port still lacks Four Over Six, packed saved activations and fused GEMM
outer-scale epilogues. This dense model does not exercise MoE communications.

## Reproduce

On Spark, from `/home/kedar/projects/inklings2-mini`:

```sh
PY=/home/kedar/projects/lawzero-reproduction/.venv/bin/python
$PY spark/check_pack.py --tiled
$PY spark/check_native.py --pack-backend triton-tiled
PYTHON=$PY PACK_BACKEND=triton-tiled RUN_DIR=results/my-comparison \
  spark/run_fused_comparison.sh
```

PyTorch 2.14.0+cu130, Triton 3.8.0, driver 580.178.04, GB10 SM 12.1.
No installed packages were changed. Core source hashes are embedded in each
run log. `source-hashes.json` captures the final local source files.

Raw evidence is in this directory: `b*.jsonl`, `layer-*.json`,
`transpose-layouts.json` and `check-*.json`. Unqualified `fp4-native-sr` filenames
use the first fused row backend; filenames ending in `-tiled` use the final
recommended fused backend. The row kernels remain selectable for comparison.
