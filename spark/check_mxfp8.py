"""Validate the full native MXFP8 forward/backward path and saved values."""
import argparse
import json
import torch
from torch import nn
import native
from check_pack import decoded_mx
from precision import mxfp8_reference
from record_model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack-backend', choices=['eager', 'triton', 'triton-tiled'], default='triton-tiled')
    args = p.parse_args()
    native.set_pack_backend(args.pack_backend)
    torch.manual_seed(51)
    for transpose in (False, True):
        x = torch.randn(129, 256, device='cuda')
        x[0] = 0
        x[1, 0] = 1000
        if transpose:
            x = x.T.contiguous().T  # Noncontiguous logical rows, still 256 columns.
        for stochastic in (False, True):
            q, s, d = native.fp8_pack(x, stochastic, return_dequant=True)
            assert torch.equal(d, decoded_mx(q, s).bfloat16())
            assert torch.isfinite(d).all()
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    w = torch.randn(384, 256, device='cuda', requires_grad=True)
    errors = {}
    for stochastic in (False, True):
        x.grad = w.grad = None
        y = native.MXFP8LinearFn.apply(x, w, stochastic)
        ax = mxfp8_reference(x.detach(), False).bfloat16()
        bw = mxfp8_reference(w.detach(), False).bfloat16()
        expected = ax.float() @ bw.float().T
        forward_error = ((y.float()-expected).norm()/expected.norm()).item()
        assert forward_error < .004
        g = torch.randn_like(y)
        y.backward(g)
        assert x.grad.dtype == w.grad.dtype == torch.float32
        assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        errors[f'forward_sr_{stochastic}'] = forward_error
        if not stochastic:
            gg = g.reshape(-1, 384).float()
            dx = mxfp8_reference(gg, False) @ mxfp8_reference(bw.T.float().contiguous(), False).T
            dw = mxfp8_reference(gg.T.contiguous(), False) @ mxfp8_reference(ax.reshape(-1,256).T.float().contiguous(), False).T
            for name, actual, ref in [('dx', x.grad.reshape_as(dx), dx), ('dw', w.grad, dw)]:
                error = ((actual-ref).norm()/ref.norm()).item()
                assert error < .004, (name, error)
                errors[name] = error
    # Precision coverage: protect the last sixth with MXFP8 in both recipes.
    model = GPT(GPTConfig(n_layer=6, n_embd=128, n_head=2))
    native.replace_mlps(model, forward_format='nvfp4', tail_precision='mxfp8')
    formats = [b.mlp.c_fc.forward_format for b in model.transformer.h]
    assert formats == ['nvfp4']*5 + ['mxfp8']
    model = GPT(GPTConfig(n_layer=6, n_embd=128, n_head=2))
    native.replace_mlps(model, forward_format='mxfp8', tail_precision='mxfp8')
    assert all(b.mlp.c_fc.forward_format == 'mxfp8' for b in model.transformer.h)
    assert isinstance(model.lm_head, nn.Linear)
    print(json.dumps({'passed': True, 'pack_backend': args.pack_backend,
                      'dequant_and_strides': True, 'coverage': True, 'relative_errors': errors}))


if __name__ == '__main__':
    main()
