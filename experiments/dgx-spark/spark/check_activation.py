"""Check fused activation packing and BF16 derivatives against the old path."""
import copy
import json
import torch
from torch.nn import functional as F
import native
from activation import pack_activation, activation_backward, ReluSquareMXFP8Fn, FusedActivationMLP
from record_model import MLP, GPTConfig
from bench_layer import measure


def main():
    native.set_pack_backend('triton-tiled')
    torch.manual_seed(91)
    cases = 0
    for rows, cols in [(1, 32), (17, 96), (129, 256), (256, 3072)]:
        z = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16)
        z[0, :8] = torch.tensor([0., -0., .5, -.5, 2**-30, -2**-30, 20., -20.], device='cuda', dtype=torch.bfloat16)
        ref = native.fp8_pack(F.relu(z).square(), False, return_dequant=True)
        fused = pack_activation(z)
        for i, (a,b) in enumerate(zip(ref, fused)):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (rows, cols, i)
        zz = z.detach().clone().requires_grad_()
        g = torch.randn_like(z)
        F.relu(zz).square().backward(g)
        assert torch.equal(zz.grad, activation_backward(z, g)), (rows, cols, 'derivative')
        cases += 1
    # Native projection parity, including stochastic backward RNG consumption.
    z = torch.randn(128, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(384, 256, device='cuda', requires_grad=True)
    g = torch.randn(128, 384, device='cuda', dtype=torch.bfloat16)
    for stochastic in (False, True):
        values = []
        state = torch.cuda.get_rng_state()
        for fused in (False, True):
            torch.cuda.set_rng_state(state)
            z.grad = w.grad = None
            y = ReluSquareMXFP8Fn.apply(z, w, stochastic) if fused else native.MXFP8LinearFn.apply(F.relu(z).square(), w, stochastic)
            y.backward(g)
            values.append((y.detach().clone(), z.grad.clone(), w.grad.clone(), torch.cuda.get_rng_state()))
        for i, (a,b) in enumerate(zip(*values)):
            assert torch.equal(a, b), ('projection', stochastic, i)
    # Whole MLP integration preserves parameter names and both projection gradients.
    mlp = MLP(GPTConfig(n_embd=128, n_head=2)).cuda()
    for name in ('c_fc', 'c_proj'):
        linear = getattr(mlp, name)
        linear.weight.data.normal_(0, .03)
        setattr(mlp, name, native.NativeLinear(linear, True, 'mxfp8'))
    fused = FusedActivationMLP(copy.deepcopy(mlp))
    assert list(dict(mlp.named_parameters())) == list(dict(fused.named_parameters()))
    x = torch.randn(2, 64, 128, device='cuda', requires_grad=True)
    g = torch.randn(2, 64, 128, device='cuda', dtype=torch.bfloat16)
    values = []
    state = torch.cuda.get_rng_state()
    for model in (mlp, fused):
        torch.cuda.set_rng_state(state)
        x.grad = None
        y = model(x)
        y.backward(g)
        values.append((y.detach().clone(), x.grad.clone(), *[p.grad.clone() for p in model.parameters()]))
    for i,(a,b) in enumerate(zip(*values)):
        assert torch.equal(a,b), ('mlp', i)
    z = torch.randn(8192, 3072, device='cuda', dtype=torch.bfloat16)
    separate_ms = measure(lambda: native.fp8_pack(F.relu(z).square(), False, return_dequant=True))
    fused_ms = measure(lambda: pack_activation(z))
    print(json.dumps({'passed': True, 'packed_bitwise_parity_cases': cases,
                      'exact_projection_and_mlp_gradients': True, 'rng_consumption_parity': True,
                      'shape': [8192,3072], 'separate_activation_pack_ms': separate_ms,
                      'fused_activation_pack_ms': fused_ms}))


if __name__ == '__main__':
    main()
