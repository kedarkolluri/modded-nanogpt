"""Single-Spark diagnostic derived from modded-nanogpt's ModernArch record.

This is an experimental adaptation, not a leaderboard-comparable run.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from record_model import GPT, GPTConfig, Muon
from precision import replace_mlps
from profile_tools import StepProfiler


def tokens(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f'Invalid FineWeb shard: {path}')
    data = np.memmap(path, mode='r', dtype=np.uint16, offset=1024)
    if len(data) != header[2]:
        raise ValueError(f'Truncated FineWeb shard: {path}')
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--precision', choices=['bf16', 'fp4-reference-sr', 'fp4-reference-nearest', 'fp4-native-sr', 'fp4-native-nearest', 'mxfp8-native-sr', 'mxfp8-native-nearest'], default='bf16')
    p.add_argument('--fuse-activation', action='store_true')
    p.add_argument('--fp4-epilogue', choices=['eager', 'triton'], default='eager')
    p.add_argument('--tail-precision', choices=['bf16', 'mxfp8'], default='bf16')
    p.add_argument('--pack-backend', choices=['eager', 'triton', 'triton-tiled'], default='eager')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--sequence', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--profile-dir', type=Path)
    p.add_argument('--profile-start', type=int, default=20)
    p.add_argument('--profile-steps', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.batch * args.sequence % 32 or args.width % 32:
        p.error('batch*sequence and width must be divisible by 32')
    if args.profile_dir and (args.profile_start < 5 or args.profile_steps < 1 or args.profile_start + args.profile_steps > args.steps):
        p.error('Profile must begin after warmup and finish within the run')
    if args.fuse_activation and (not args.precision.startswith('mxfp8-native') or args.pack_backend == 'eager'):
        p.error('Activation fusion requires native MXFP8 and a Triton packing backend')
    profiler = StepProfiler(args.profile_dir, args.profile_start, args.profile_steps)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = tokens(args.data / 'fineweb_train_000001.bin')
    val = tokens(args.data / 'fineweb_val_000000.bin')
    count = args.batch * args.sequence
    if len(train) < count * args.steps + 1 or len(val) < count * args.eval_batches + 1:
        p.error('Not enough tokens for this run')
    model = GPT(GPTConfig(n_layer=args.layers, n_embd=args.width, n_head=args.heads)).cuda()
    if args.precision.startswith(('fp4-native', 'mxfp8-native')):
        from native import replace_mlps as replace_native, set_pack_backend, set_epilogue_backend
        set_pack_backend(args.pack_backend)
        set_epilogue_backend(args.fp4_epilogue)
        replace_native(model, stochastic=args.precision.endswith('-sr'),
                       forward_format='mxfp8' if args.precision.startswith('mxfp8') else 'nvfp4',
                       tail_precision=args.tail_precision)
    elif args.precision != 'bf16':
        replace_mlps(model, stochastic=args.precision.endswith('-sr'))
    if args.fuse_activation:
        from activation import replace_activation_mlps
        replace_activation_mlps(model)
    # Match initialization/data across modes. Separate RNG for quantization.
    torch.cuda.manual_seed_all(args.seed + 1000)
    optimizers = [
        torch.optim.AdamW([model.transformer.wte.weight], lr=.003, betas=(.9, .95), fused=True),
        Muon(model.transformer.h.parameters(), lr=.02),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def emit(row):
        print(json.dumps(row), flush=True)
        with args.output.open('a') as f:
            f.write(json.dumps(row) + '\n')
    if args.output.exists():
        raise FileExistsError(args.output)
    emit({'config': vars(args) | {'data': str(args.data), 'output': str(args.output),
                                    'profile_dir': str(args.profile_dir) if args.profile_dir else None},
          'parameters': sum(x.numel() for x in model.parameters()),
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
          'native_low_precision_gemm': args.precision.startswith(('fp4-native', 'mxfp8-native')),
          'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('train.py', 'native.py', 'triton_pack.py', 'triton_pack_tiled.py', 'precision.py', 'record_model.py', 'profile_tools.py', 'trace_analysis.py', 'epilogue.py', 'activation.py')}})
    def batch(data, offset):
        z = torch.from_numpy(np.array(data[offset:offset+count+1], dtype=np.int64)).cuda()
        return z[:-1].view(args.batch, args.sequence), z[1:].view(args.batch, args.sequence)
    train_seconds = 0.
    measured_seconds = 0.
    measured_steps = 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            # Validation must not consume the training quantization random stream.
            model.eval()
            with torch.random.fork_rng(devices=[0]), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                torch.cuda.manual_seed_all(args.seed + 2000)
                losses = [model(*batch(val, i*count), return_logits=False)[1]
                          for i in range(args.eval_batches)]
                loss = torch.stack(losses).mean().item()
            emit({'step': step, 'val_loss': loss, 'train_seconds': train_seconds})
            model.train()
        if step == args.steps:
            break
        x, y = batch(train, step * count)
        torch.cuda.synchronize()
        profiler.before_step(step, model)
        start = time.perf_counter()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with profiler.scope('phase/forward'):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y, return_logits=False)
        with profiler.scope('phase/backward'):
            loss.backward()
        with profiler.scope('phase/optimizer'):
            for index, opt in enumerate(optimizers):
                with profiler.scope('optimizer/adamw' if index == 0 else 'optimizer/muon'):
                    opt.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - start
        profiler.after_step(step)
        train_seconds += duration
        if step >= 5:
            measured_seconds += duration
            measured_steps += 1
        value = loss.item()
        if not np.isfinite(value):
            raise RuntimeError(f'Non-finite loss at step {step}')
        if step % 10 == 0:
            emit({'step': step+1, 'train_loss': value, 'step_seconds': duration})
    profiler.finish()
    emit({'finished': True, 'tokens': args.steps * count,
          'train_seconds': train_seconds, 'warmup_excluded_steps': min(5, args.steps),
          'steady_tokens_per_second': measured_steps * count / measured_seconds if measured_steps else None,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
