"""SM12x native MXFP8 GEMM with ReLU-square/packing in its epilogue.

Preactivation and reconstructed activation are still saved for training backward.
The epilogue removes the separate preactivation read and activation-pack launch.
No TMEM and no BF16 GEMM emulation are used; verify emitted PTX in the check.
"""
import torch
from torch import nn
import triton
import triton.language as tl
import native
from activation import activation_backward
from triton_pack import _scale_offset


@triton.jit
def _gemm_pack(A, AS, W, WS, Z, H, HS, HD,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               KSC: tl.constexpr, NSC: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0)*BM + tl.arange(0, BM)
    rn = tl.program_id(1)*BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, BK//32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block*BK + rk
        sk = block*(BK//32) + rs
        a = tl.load(A + rm[:, None]*K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), 0.)
        b = tl.load(W + rn[None, :]*K + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K), 0.)
        sa = tl.load(AS + _scale_offset(rm[:, None], sk[None, :], KSC),
                     (rm[:, None] < M) & (sk[None, :] < K//32), 127)
        sb = tl.load(WS + _scale_offset(rn[:, None], sk[None, :], KSC),
                     (rn[:, None] < N) & (sk[None, :] < K//32), 127)
        acc = tl.dot_scaled(a, sa, 'e4m3', b, sb, 'e4m3', acc)
    # Preserve the same two BF16 rounding boundaries as the existing pipeline.
    z = acc.to(tl.bfloat16)
    positive = tl.maximum(z.to(tl.float32), 0.)
    h = (positive*positive).to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(h, (BM, BN//32, 32))
    maximum = tl.max(tl.abs(blocks), 2)
    exponent = tl.ceil(tl.log2(tl.maximum(tl.div_rn(maximum, 448.), 1.1754943508222875e-38)))
    scale = tl.exp2(exponent)
    q = (blocks/scale[:, :, None]).to(tl.float8e4nv)
    offset = rm[:, None]*N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Z + offset, z, mask)
    tl.store(H + offset, tl.reshape(q, (BM, BN)), mask)
    dequant = tl.reshape(q.to(tl.float32)*scale[:, :, None], (BM, BN))
    tl.store(HD + offset, dequant, mask)
    sn = tl.program_id(1)*(BN//32) + tl.arange(0, BN//32)
    encoded = tl.where(rm[:, None] < M, exponent + 127., 0.).to(tl.uint8)
    tl.store(HS + _scale_offset(rm[:, None], sn[None, :], NSC), encoded,
             (rm[:, None] < tl.cdiv(M,128)*128) & (sn[None, :] < N//32))


def gemm_pack(a, sa, w, sw, bm=32, bn=128, bk=128, warps=4, stages=2, return_kernel=False):
    m, k = a.shape
    n = w.shape[0]
    if k % 32 or n % 128:
        raise ValueError('Fused epilogue requires K divisible by 32 and N by 128')
    z = torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    h = torch.empty((m,n),device=a.device,dtype=torch.float8_e4m3fn)
    hd = torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    hs = torch.empty(triton.cdiv(m,128)*128*(n//32),device=a.device,dtype=torch.uint8)
    kernel = _gemm_pack[(triton.cdiv(triton.cdiv(m,128)*128,bm),triton.cdiv(n,bn))](
        a,sa.view(torch.uint8),w,sw.view(torch.uint8),z,h,hs,hd,
        m,n,k,triton.cdiv(k//32,4)*4,n//32,bm,bn,bk,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    result = (z,h,hs.view(torch.float8_e8m0fnu),hd)
    return (*result,kernel) if return_kernel else result


class FusedMLPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, stochastic):
        shape = x.shape
        aq,sa,ax = native.fp8_pack(x.reshape(-1,shape[-1]),False,return_dequant=True)
        bq,sb,bw = native.fp8_pack(w1,False,return_dequant=True)
        z,hq,hs,hx = gemm_pack(aq,sa,bq,sb)
        cq,sc,cw = native.fp8_pack(w2,False,return_dequant=True)
        y = torch._scaled_mm(hq,cq.T,hs,sc,out_dtype=torch.bfloat16)
        ctx.save_for_backward(ax,bw,z,hx,cw)
        ctx.shape,ctx.stochastic = shape,stochastic
        return y.reshape(*shape[:-1],w2.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x,w,z,h,v = ctx.saved_tensors
        g = grad.reshape(-1,grad.shape[-1])
        dh = native.mx_mm(g,v.T,ctx.stochastic)
        dv = native.mx_mm(g.T,h.T,ctx.stochastic)
        dz = activation_backward(z,dh)
        dx = native.mx_mm(dz,w.T,ctx.stochastic)
        dw = native.mx_mm(dz.T,x.T,ctx.stochastic)
        return dx.reshape(ctx.shape),dw,dv,None


class FusedEpilogueMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.c_fc,self.c_proj = mlp.c_fc,mlp.c_proj
        self.stochastic = mlp.c_fc.stochastic

    def forward(self, x):
        return FusedMLPFn.apply(x,self.c_fc.weight,self.c_proj.weight,self.stochastic)


def replace_mlps(model):
    count = 0
    for block in model.transformer.h:
        if getattr(block.mlp.c_fc,'forward_format',None) == 'mxfp8':
            block.mlp = FusedEpilogueMLP(block.mlp)
            count += 1
    return count
