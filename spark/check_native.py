"""Validate nonuniform packed layouts and both native gradient GEMMs."""
import argparse
import json
import torch
import native
from native import NativeLinearFn
from precision import nvfp4_reference, mxfp8_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack-backend', choices=['eager', 'triton', 'triton-tiled'], default='eager')
    args = parser.parse_args()
    native.set_pack_backend(args.pack_backend)
    fp4_pack, fp8_pack = native.fp4_pack, native.fp8_pack
    torch.manual_seed(19)
    x = torch.randn(256, 512, device='cuda')
    w = torch.randn(384, 512, device='cuda')
    aq, sa, ao, ax = fp4_pack(x, True)
    bq, sb, bo, bw = fp4_pack(w)
    # Packing and reference must reconstruct identical BF16 forward inputs.
    assert torch.equal(ax, nvfp4_reference(x, True).bfloat16())
    assert torch.equal(bw, nvfp4_reference(w).bfloat16())
    native_out = (torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.float32)*ao*bo).bfloat16()
    expected = ax.float() @ bw.float().T
    error = ((native_out.float()-expected).norm()/expected.norm()).item()
    assert error < .01, error
    fa, fsa = fp8_pack(x, False)
    fb, fsb = fp8_pack(w, False)
    mx = torch._scaled_mm(fa, fb.T, fsa, fsb, out_dtype=torch.bfloat16)
    ref = mxfp8_reference(x, False) @ mxfp8_reference(w, False).T
    mx_error = ((mx.float()-ref).norm()/ref.norm()).item()
    assert mx_error < .01, mx_error
    x.requires_grad_(); w.requires_grad_()
    y = NativeLinearFn.apply(x, w, False)
    g = torch.randn_like(y)
    y.backward(g)
    dx = mxfp8_reference(g.float(), False) @ mxfp8_reference(bw.T.float().contiguous(), False).T
    dw = mxfp8_reference(g.T.float().contiguous(), False) @ mxfp8_reference(ax.T.float().contiguous(), False).T
    dx_error = ((x.grad-dx).norm()/dx.norm()).item()
    dw_error = ((w.grad-dw).norm()/dw.norm()).item()
    assert dx_error < .01 and dw_error < .01, (dx_error, dw_error)
    print(json.dumps({'passed': True, 'pack_backend': args.pack_backend, 'nvfp4_relative_error': error, 'mxfp8_relative_error': mx_error,
                      'dx_relative_error': dx_error, 'dw_relative_error': dw_error}))


if __name__ == '__main__':
    main()
