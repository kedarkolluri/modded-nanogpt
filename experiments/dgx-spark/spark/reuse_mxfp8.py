"""Prepack reusable weights once; quantize only activations inside GEMM.

Weight packing is redone on every call, so optimizer updates cannot make a
cached quantized weight stale. Activation conversion still repeats across N tiles.
"""
import torch
import triton
import triton.language as tl
import native
from pipelined_mxfp8 import _quant
from triton_pack import _scale_offset


@triton.jit
def _tile(A,W,WS,AD,rm,rn,block,jn,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          NC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    kk=block*BK+tl.arange(0,BK)
    sk=block*(BK//32)+tl.arange(0,BK//32)
    av=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0.)
    aq,sa,ad=_quant(av,BM,BK)
    wq=tl.load(W+rn[:,None]*K+kk[None,:],(rn[:,None]<N)&(kk[None,:]<K),0.)
    sw=tl.load(WS+_scale_offset(rn[:,None],sk[None,:],NC),
               (rn[:,None]<N)&(sk[None,:]<K//32),127)
    if jn == 0:
        tl.store(AD+rm[:,None]*K+kk[None,:],ad,(rm[:,None]<M)&(kk[None,:]<K))
    return aq,sa,wq,sw


@triton.jit
def _activation_kernel(A,W,WS,Y,AD,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                       NC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                       BUFFERED:tl.constexpr):
    im=tl.program_id(0)
    jn=tl.program_id(1)
    rm=im*BM+tl.arange(0,BM)
    rn=jn*BN+tl.arange(0,BN)
    acc=tl.full((BM,BN),0,tl.float32)
    if BUFFERED:
        aq,sa,wq,sw=_tile(A,W,WS,AD,rm,rn,0,jn,M,N,K,NC,BM,BN,BK)
        for block in range(1,tl.cdiv(K,BK)):
            nq,ns,nw,nws=_tile(A,W,WS,AD,rm,rn,block,jn,M,N,K,NC,BM,BN,BK)
            acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
            aq,sa,wq,sw=nq,ns,nw,nws
        acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
    else:
        for block in range(tl.cdiv(K,BK)):
            aq,sa,wq,sw=_tile(A,W,WS,AD,rm,rn,block,jn,M,N,K,NC,BM,BN,BK)
            acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
    tl.store(Y+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))


def activation_mm(a,wq,sw,bm=16,bn=512,bk=64,warps=8,stages=1,buffered=False,return_kernel=False):
    m,k=a.shape
    n=wq.shape[0]
    if k%32 or not a.is_contiguous() or not wq.is_contiguous():
        raise ValueError('Requires contiguous matrices and K divisible by 32')
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    ad=torch.empty_like(a,dtype=torch.bfloat16)
    kernel=_activation_kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
        a,wq,sw.view(torch.uint8),y,ad,m,n,k,triton.cdiv(k//32,4)*4,
        bm,bn,bk,buffered,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (y,ad,kernel) if return_kernel else (y,ad)


def weight_reuse_mm(a,w,**kwargs):
    wq,sw,wd=native.fp8_pack(w,False,return_dequant=True)
    y,ad=activation_mm(a,wq,sw,**kwargs)
    return y,ad,wd
