"""Bounded warmed-up CPU/CUDA profiles with phase and quantization ranges.

Profiler timings diagnose attribution; use separate unprofiled runs for speed.
"""
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import shutil
import torch


class StepProfiler:
    def __init__(self, directory, start=20, steps=3):
        self.directory = directory
        self.start, self.steps = start, steps
        self.active = False
        self.prof = None
        self.hooks = []
        self.originals = {}

    def scope(self, name):
        return torch.profiler.record_function(name) if self.active else nullcontext()

    def before_step(self, step, model):
        if self.directory is None or step != self.start:
            return
        self.directory.mkdir(parents=True, exist_ok=False)
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_stack=False, profile_memory=False)
        self.prof.start()
        self.active = True
        for name, module in model.named_modules():
            if name == 'lm_head' or name.endswith(('.attn', '.mlp')):
                stack = []
                def pre(mod, inputs, label=name, stack=stack):
                    ctx = torch.profiler.record_function('module/' + label)
                    stack.append(ctx)
                    ctx.__enter__()
                def post(mod, inputs, output, stack=stack):
                    stack.pop().__exit__(None, None, None)
                self.hooks.extend([module.register_forward_pre_hook(pre),
                                   module.register_forward_hook(post, always_call=True)])
        # Patch only during profiling, keeping normal benchmark paths untouched.
        import native
        for name in ('fp4_pack', 'fp8_pack', 'mx_mm'):
            original = getattr(native, name)
            self.originals[name] = original
            def wrapped(*args, _fn=original, _name=name, **kwargs):
                with torch.profiler.record_function('native/' + _name):
                    return _fn(*args, **kwargs)
            setattr(native, name, wrapped)

    def after_step(self, step):
        if not self.active:
            return
        self.prof.step()
        if step + 1 == self.start + self.steps:
            self.finish()

    def finish(self):
        if not self.active:
            return
        import native
        for hook in self.hooks:
            hook.remove()
        for name, fn in self.originals.items():
            setattr(native, name, fn)
        self.prof.stop()
        self.active = False
        rows = []
        for event in self.prof.key_averages(group_by_input_shape=True):
            rows.append({'name': event.key, 'count': event.count,
                         'self_cpu_us': event.self_cpu_time_total,
                         'cpu_us': event.cpu_time_total,
                         'self_cuda_us': event.self_device_time_total,
                         'cuda_us': event.device_time_total,
                         'input_shapes': event.input_shapes})
        summary = {'profiled_steps': self.steps, 'start_step_zero_based': self.start,
                   'note': 'Profiler overhead is included; compare speed with separate unprofiled runs.',
                   'operators': sorted(rows, key=lambda r: r['self_cuda_us'], reverse=True)}
        (self.directory / 'operators.txt').write_text(self.prof.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=50))
        trace = self.directory / 'trace.pt.trace.json'
        self.prof.export_chrome_trace(str(trace))
        with trace.open('rb') as src, gzip.open(str(trace)+'.gz', 'wb') as dst:
            shutil.copyfileobj(src, dst)
        from trace_analysis import analyze_trace
        summary['trace_analysis'] = analyze_trace(str(trace)+'.gz', self.steps)
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        trace.unlink()
