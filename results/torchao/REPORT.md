# TorchAO MXFP8 versus our compiled implementation on DGX Spark

## Integration

TorchAO **0.18.0**, with PyTorch **2.14.0+cu130**, installed in an isolated
`.venv-torchao` environment. A `.pth` file exposes the existing CUDA environment's
packages, so PyTorch, Triton and NumPy are shared without changing their versions
or modifying the original environment. All comparison runs use this interpreter.

The PyPI ARM wheel is Python-only. We built the official MXFP8 extension from
matching release tag `v0.18.0`, commit recorded in [build.json](build.json), with
compute_120 PTX for GB10. The four official extension sources are unmodified;
no TorchAO Python code is patched. The build installs only `_C_mxfp8.so` into the
isolated package. The build script, source hashes and compiler flags are saved.

Our thin adapter uses TorchAO's own `_to_mxfp8_then_scaled_mm` primitive, the same
one used by its `MXFP8Linear` module. It selects native AUTO kernels, RCEIL scales,
and MXFP8 for both backward GEMMs. TorchAO requires BF16 operands, so the adapter
casts inputs and weights while retaining the same FP32 master parameters and
optimizer state as our model. These casts are part of measured execution.
Attention and the output head retain their previous BF16-autocast path.

**Numerical recipes differ.** TorchAO quantizes original saved BF16 operands with
deterministic rounding. Our implementation requantizes reconstructed forward
values and uses stochastic rounding in backward. The benchmark compares these
implementations, not two bitwise-equivalent realizations of the same estimator.
Training convergence equivalence remains untested.

## Method

The same 123,568,128-parameter ModernArch model, FineWeb shard, seed 42,
batch 32, sequence 256, 100 training steps, and Muon plus fused AdamW. All 24 MLP
projections use MXFP8 in either MXFP8 implementation. Validation uses 8,192 tokens
every 50 steps. Optimizer work is included in throughput; input preparation,
evaluation and the first five warmup steps are excluded.

All paths use `torch.compile(fullgraph=True, dynamic=False)` in default Inductor
mode. No CUDA Graph replay or max-autotune mode is enabled. The optimizer loop
is unchanged; its Muon Newton–Schulz function was already compiled.

Order: BF16, ours, TorchAO ordinary activation, TorchAO generic fused activation;
then the complete reverse order. Each run has fresh model and optimizer state.
All GPU runs are serial. Compiler cache reuse is permitted; startup is not a
cold-compilation benchmark. Both TorchAO activation variants use TorchAO's own
packing; generic fusion only combines ReLU-square and its derivative.

## Results

| Compiled path | Repeat 1 tokens/s | Repeat 2 tokens/s | Mean tokens/s | Peak allocated GiB |
|---|---:|---:|---:|---:|
| BF16, generic fused activation | 37,724 | 37,577 | **37,651** | 5.08 |
| Our MXFP8, fused activation packing | 35,897 | 35,942 | **35,919** | 4.69 |
| TorchAO MXFP8, ordinary activation | 35,087 | 35,146 | **35,116** | 4.69 |
| TorchAO MXFP8, generic fused activation | 35,867 | 35,788 | **35,827** | 4.69 |

TorchAO with generic activation fusion is within **0.3%** of our implementation,
which is too small a difference to establish a meaningful performance advantage
from two repeats. It remains **4.8% below BF16** on this Spark workload.
Ordinary TorchAO is **2.2% below our implementation**; adding generic activation
fusion improves TorchAO throughput **2.0%**. MXFP8 modes use about **7.7% less peak
allocated GPU memory** than BF16.

All eight 100-step runs completed with finite training losses and zero graph
breaks. Raw logs, per-step times and final validation losses are in
[comparison/summary.md](comparison/summary.md). These short single-seed runs do
not establish equal convergence or time to a target loss. This is a benchmark
of the specified FP32-master adapter and default native TorchAO recipe on GB10,
not a maximum-performance claim for all TorchAO configurations or model sizes.

## Correctness and native execution checks

`check_torchao.py` compares the adapter and compiled adapter against TorchAO's
official BF16 `MXFP8Linear`. Outputs, input gradients and weight gradients matched
exactly. The adapter's master parameters remain FP32. The native CUDA quantization
operator reports a registered CUDA implementation. The full-model smoke run also
completed with finite losses and zero graph breaks.

## Reproduction

Create an isolated environment and install the pinned wheel without upgrading
PyTorch. Expose the original environment's site-packages with a `.pth` file.
Check out `v0.18.0` in `upstream/torchao`, then run with the isolated interpreter:

```sh
$PY spark/build_torchao_mxfp8.py upstream/torchao --metadata results/torchao/build.json
$PY spark/check_torchao.py
PYTHON=$PY RUN_DIR=results/new-torchao-comparison spark/run_torchao_comparison.sh
PYTHON=$PY RUN_DIR=results/new-torchao-nsys spark/run_torchao_nsys.sh
```

Sources: [TorchAO 0.18.0 linear primitive](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/moe_training/mxfp8_linear.py)
and [official extension build configuration](https://github.com/pytorch/ao/blob/v0.18.0/setup.py).

## Nsight Systems verification

Separate ordinary/fused TorchAO runs captured three warmed-up steps beginning at
step 20. Both traces contain **72 native block-scaled E4M3/E8M0 GEMMs per step**,
which accounts for all three GEMMs in all 24 MLP projections. The actual kernel
is `cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_ue8m0xe4m3_ue8m0xe4m3...`,
also used by our native implementation. There is no emulated-MXFP8 GEMM fallback
in these captures. CUDA column quantization and Triton row quantization kernels
are both present.

| Kernel measurement per step | TorchAO ordinary | TorchAO generic fused |
|---|---:|---:|
| Forward | 70.57 ms | 66.01 ms |
| Backward | 97.83 ms | 98.66 ms |
| Optimizer | 57.90 ms | 58.58 ms |
| Forward kernel count | 486 | 475 |
| Backward kernel count | 429 | 429 |
| Optimizer kernel count | 2,318 | 2,318 |

Both captures have zero unmatched kernels and phase sums reconcile with all
captured kernel time. Attribution correlates CUDA runtime launches with serial
CPU phase windows across autograd threads. Profile durations diagnose work;
reported throughput comes from the eight separate unprofiled benchmark runs.

The traces confirm that TorchAO supplies optimized conversion kernels, while
using the same native block-scaled GEMM engine here. Conversion, scale layout,
casting and activation work remain substantial around those GEMMs. Neither
implementation eliminates the conversion cost on this model shape and hardware.

CSV exports, analysis JSON and binary reports are under `nsys/`. Binary reports
and SQLite databases are retained locally and on Spark but excluded from Git.
