"""Build TorchAO's unmodified MXFP8 CUDA extension for Spark's ARM wheel.

Run in the isolated TorchAO environment against an exact matching release tree.
Only the official MXFP8 extension is built; no package Python code is patched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch
import torchao
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('--metadata', type=Path, required=True)
args = p.parse_args()
root = args.source.resolve()
if (root / 'version.txt').read_text().strip() != torchao.__version__.split('+')[0]:
    raise ValueError('Build sources must match the installed TorchAO release')
csrc = root / 'torchao/csrc/cuda/mx_kernels'
sources = [csrc / name for name in ('mxfp8_extension.cpp', 'mxfp8_cuda.cu',
                                  'mx_block_rearrange_2d_M_groups.cu', 'fused_pad_token_groups.cu')]
flags = ['-std=c++20', '-O3', '-DUSE_CUDA', '-DTORCH_TARGET_VERSION=0x020b000000000000']
cuda_flags = flags + ['-gencode=arch=compute_120,code=compute_120', '--expt-relaxed-constexpr']
path = load(name='_C_mxfp8_spark', sources=[str(x) for x in sources],
            extra_include_paths=[str(csrc)], extra_cflags=flags,
            extra_cuda_cflags=cuda_flags, is_python_module=False, verbose=True)
target = Path(torchao.__file__).parent / '_C_mxfp8.so'
shutil.copy2(path, target)
args.metadata.parent.mkdir(parents=True, exist_ok=True)
args.metadata.write_text(json.dumps({'torch': torch.__version__, 'torchao': torchao.__version__,
    'source_commit': subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
    'source_status': subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True),
    'sources_sha256': {str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
    'cflags': flags, 'cuda_flags': cuda_flags, 'installed_extension': str(target),
    'extension_sha256': hashlib.sha256(target.read_bytes()).hexdigest()},indent=2)+'\n')
