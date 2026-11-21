# Native within-layer row pipeline

Reuses the existing Triton `_fp8` packing kernel and cuBLAS `_scaled_mm` GEMM. No new MMA implementation. One layer's activation rows are partitioned; a producer stream packs the next chunk while the main stream multiplies a ready chunk. Each activation element is packed once; weights are packed once per call. Saved BF16 operands are included. Output and saved activations are written directly into contiguous slices, without assembly copies.

Production shape M=8192, N=3072, K=768; seed 45. Four alternating forward/reverse benchmark orders, 100 ms warmup and 300 ms measurement each. CUDA graph replay for all paths removes Python launch gaps. Stream events establish readiness; record_stream protects producer allocations until consumers finish; producer rejoins the caller. Every eager and captured output and both saved operands match the full native call exactly.

| Rows per chunk | Serial ms | Overlap ms |
|---|---:|---:|
| Full native | 0.6231 | — |
| 512 | 0.6640 | 0.6599 |
| 1024 | 0.6185 | 0.6318 |
| 2048 | 0.6051 | 0.6201 |
| 4096 | 0.6299 | 0.6260 |

Medians across four measurements. Small differences are not a demonstrated training gain. The best serial chunking is about 2.9% faster than full native in this microbenchmark; overlap is slower than that serial control. No training integration or BF16 advantage is claimed.

## Measured overlap

Nsight Systems captured five graph replays per path, tracing graph nodes. Analysis sums intersections of packing and native GEMM kernel intervals on different streams. For 2048-row chunks, average per call:

| Quantity | Serial ms | Overlap ms |
|---|---:|---:|
| Packing kernel duration sum | 0.2843 | 0.3002 |
| Native GEMM duration sum | 0.3049 | 0.3439 |
| Cross-stream pack/GEMM overlap | 0 | 0.0561 |

The trace proves within-layer kernel concurrency. Packing and GEMM durations grow by about 0.055 ms combined, approximately offsetting the 0.056 ms of hidden time. This is consistent with resource contention; the trace does not isolate which hardware resource causes it. There are no output concatenation kernels in the final trace. These profiled durations are separate from unprofiled benchmark medians.

The first implementation concatenated chunk outputs and saved activations, costing about 0.55 ms per call. Its results are retained as `assembly-initial*`; final code writes directly into destination slices using the unchanged packing kernel and the native GEMM out overload.

This experiment pipelines separate native kernels within one layer. It does not insert a conversion stage into cuBLAS's internal K-tile loop; that interface is not exposed by `_scaled_mm`. It also does not overlap layers. There is no reason to replace the training path with this version given these results.

## Reproduce on Spark

```sh
/home/kedar/projects/lawzero-reproduction/.venv/bin/python spark/bench_native_chunk_pipeline.py
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o results/native-chunk-pipeline/trace /home/kedar/projects/lawzero-reproduction/.venv/bin/python spark/bench_native_chunk_pipeline.py --profile
nsys export --type sqlite --force-overwrite=true --output results/native-chunk-pipeline/trace.sqlite results/native-chunk-pipeline/trace.nsys-rep
python3 spark/analyze_native_chunk_pipeline.py
```

Binary traces remain on Spark and are ignored by Git. Raw timings, summaries, benchmark, and analyzer are committed.
