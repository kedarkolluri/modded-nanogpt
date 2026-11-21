"""Compare fullgraph compiled fused MLP outputs and gradients with eager.

Nearest MXFP8 isolates compiler correctness from stochastic RNG sequence changes.
Stochastic training is checked separately in the real-model smoke/benchmark runs.
"""
import copy
import json
import torch
import native
from activation import FusedActivationMLP, FusedBF16ActivationMLP
from record_model import MLP, GPTConfig

native.set_pack_backend('triton-tiled')
results = []
for mode in ('bf16', 'mxfp8-nearest'):
    torch.manual_seed(42)
    model = MLP(GPTConfig(n_embd=256, n_head=2)).cuda()
    for param in model.parameters():
        param.data.normal_(0, .02)
    if mode != 'bf16':
        for name in ('c_fc', 'c_proj'):
            setattr(model, name, native.NativeLinear(getattr(model, name), False, 'mxfp8'))
        model = FusedActivationMLP(model)
    else:
        model = FusedBF16ActivationMLP(model)
    compiled = torch.compile(copy.deepcopy(model), fullgraph=True, dynamic=False)
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    grad = torch.randn_like(x, dtype=torch.bfloat16)
    values = []
    for fn in (model, compiled):
        x.grad = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = fn(x)
        y.backward(grad)
        values.append((y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in fn.parameters()]))
    diffs = []
    for a,b in zip(*values):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        diffs.append(float((a.float()-b.float()).abs().max()))
    results.append({'mode': mode, 'max_abs_differences_output_input_weight_grads': diffs})
print(json.dumps({'passed': True, 'results': results,
                  'dynamo_counters': {k:dict(v) for k,v in torch._dynamo.utils.counters.items()}}))
