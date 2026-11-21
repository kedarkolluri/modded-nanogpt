"""Compare direct row and coalesced tile MXFP8 transpose packers."""
import json
import torch
import triton_pack as row
import triton_pack_tiled as tiled
from bench_layer import measure


def main():
    torch.manual_seed(190)
    rows = []
    for tokens in (1024, 4096, 8192):
        x = torch.randn(tokens, 768, device='cuda', dtype=torch.bfloat16).T
        for stochastic in (False, True):
            state = torch.cuda.get_rng_state()
            rq, rs = row.fp8_pack(x, stochastic)
            torch.cuda.set_rng_state(state)
            tq, ts = tiled.fp8_pack(x, stochastic)
            assert torch.equal(rq.view(torch.uint8), tq.view(torch.uint8)), (tokens, stochastic, 'codes')
            assert torch.equal(rs.view(torch.uint8), ts.view(torch.uint8)), (tokens, stochastic, 'scales')
            rows.append({'tokens': tokens, 'stochastic': stochastic,
                         'row_ms': measure(lambda: row.fp8_pack(x, stochastic)),
                         'tiled_ms': measure(lambda: tiled.fp8_pack(x, stochastic))})
    print(json.dumps({'passed_bitwise_parity': True, 'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
