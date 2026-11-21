"""Numerical invariants for the experimental reference backend."""
import json
import torch
from precision import nvfp4_reference, mxfp8_reference, ReferenceLinearFn


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)
    zeros = torch.zeros(32, 32, device=device)
    assert torch.equal(nvfp4_reference(zeros), zeros)
    assert torch.equal(mxfp8_reference(zeros), zeros)
    # Fixed block maximum makes the scale constant across independent draws.
    # Test expectation on nonrepresentable values, including negative/subnormal.
    row = torch.tensor([448., 1.07, -1.07, .0007] * 8, device=device)
    values = row.repeat(32768, 1)
    mean = mxfp8_reference(values).mean(0)
    error = (mean-row).abs().max().item()
    assert error < .002, error
    # An exactly representable input is unchanged by stochastic requantization.
    exact = torch.tensor([448., 2., -4., 0.] * 8, device=device).repeat(4, 1)
    assert torch.equal(mxfp8_reference(exact), exact)
    x = torch.randn(2, 32, 64, device=device, requires_grad=True)
    w = torch.randn(128, 64, device=device, requires_grad=True)
    y = ReferenceLinearFn.apply(x, w, True)
    y.float().square().mean().backward()
    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    # Nearest-rounding ablation: check the backward's transpose orientation.
    xx = x.detach().clone().requires_grad_()
    ww = w.detach().clone().requires_grad_()
    yy = ReferenceLinearFn.apply(xx, ww, False)
    g = torch.randn_like(yy)
    yy.backward(g)
    xq = nvfp4_reference(xx, per_token=True).to(torch.bfloat16).reshape(-1, 64)
    wq = nvfp4_reference(ww).to(torch.bfloat16)
    gg = g.reshape(-1, 128)
    dx = mxfp8_reference(gg, False) @ mxfp8_reference(wq.T.contiguous(), False).T
    dw = mxfp8_reference(gg.T.contiguous(), False) @ mxfp8_reference(xq.T.contiguous(), False).T
    assert torch.equal(xx.grad, dx.reshape_as(xx).float())
    assert torch.equal(ww.grad, dw.float())
    print(json.dumps({'passed': True, 'device': device, 'sr_max_mean_error': error}))


if __name__ == '__main__':
    main()
