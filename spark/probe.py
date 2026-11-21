"""Run on Spark: python spark/probe.py --output spark-report.json.

Tests real BF16 and tensor-scaled FP8 GEMMs. Block-scaled formats are
reported as unverified until a compatible backend is exercised.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path


def probe():
    report = {'platform': platform.platform(), 'machine': platform.machine()}
    for package in ('torch', 'torchao', 'triton', 'transformer-engine'):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    try:
        report['nvidia_smi'] = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['nvidia_smi_error'] = str(exc)
    try:
        import torch
    except ImportError:
        report['error'] = 'PyTorch is not installed'
        return report
    report['cuda_runtime'] = torch.version.cuda
    if not torch.cuda.is_available():
        report['error'] = 'CUDA is unavailable'
        return report
    report['gpu'] = torch.cuda.get_device_name()
    report['compute_capability'] = torch.cuda.get_device_capability()
    report['memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    torch.manual_seed(42)
    a = torch.randn(2048, 1024, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(1024, 2048, device='cuda', dtype=torch.bfloat16)
    reference = a @ b

    def measure(name, fn):
        try:
            for _ in range(5):
                out = fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                out = fn()
            torch.cuda.synchronize()
            report[name] = {
                'status': 'executed',
                'ms': (time.perf_counter() - start) * 1000 / 20,
                'relative_l2_error': ((out.float() - reference.float()).norm()
                                      / reference.float().norm()).item(),
                'finite': bool(torch.isfinite(out).all().item()),
            }
        except Exception as exc:
            report[name] = {'status': 'failed', 'error': str(exc)}

    measure('bf16', lambda: a @ b)
    # These fixed scales are only for the synthetic unit-normal inputs above.
    # Quantization is outside timing; this is a GEMM probe, not training speed.
    try:
        af = a.to(torch.float8_e4m3fn)
        bf = b.T.contiguous().to(torch.float8_e4m3fn).T
        scale = torch.ones((), device='cuda', dtype=torch.float32)
        measure('fp8_tensor_scaled', lambda: torch._scaled_mm(
            af, bf, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16))
    except Exception as exc:
        report['fp8_tensor_scaled'] = {'status': 'failed', 'error': str(exc)}
    report['nvfp4'] = {'status': 'unverified', 'reason': 'Needs a tested GB10 block-scaled backend'}
    report['mxfp8'] = {'status': 'unverified', 'reason': 'Tensor-scaled FP8 does not establish MXFP8 support'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('spark-report.json'))
    args = parser.parse_args()
    result = probe()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
