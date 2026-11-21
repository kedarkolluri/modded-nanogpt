"""Exercise real cuBLAS block-scaled GEMMs; never substitute BF16."""
import json
import torch


def main():
    results = {}
    for name, dtype, scale_dtype, block in [
        ('nvfp4', torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16),
        ('mxfp8', torch.float8_e4m3fn, torch.float8_e8m0fnu, 32),
    ]:
        try:
            m, n, k = 256, 256, 512
            if name == 'nvfp4':
                # E2M1 code 2 represents 1; pack two 1s into each byte.
                a = torch.full((m, k//2), 0x22, dtype=torch.uint8, device='cuda').view(dtype)
                b = torch.full((n, k//2), 0x22, dtype=torch.uint8, device='cuda').view(dtype).T
            else:
                a = torch.ones((m, k), device='cuda').to(dtype)
                b = torch.ones((n, k), device='cuda').to(dtype).T
            sa = torch.ones(m*k//block, device='cuda').to(scale_dtype)
            sb = torch.ones(n*k//block, device='cuda').to(scale_dtype)
            c = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            assert torch.equal(c, torch.full_like(c, k))
            results[name] = {'status': 'executed', 'max_error': (c.float()-k).abs().max().item()}
        except Exception as exc:
            results[name] = {'status': 'failed', 'error': str(exc)}
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
