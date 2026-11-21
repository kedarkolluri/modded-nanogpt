"""Experimental operand quantization in the native MXFP8 GEMM K loop.

The compiler schedules the loop; num_stages is an experiment, not evidence that
conversion and MMA overlap. Each output tile currently repeats operand conversion.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quant(x, ROWS: tl.constexpr, BK: tl.constexpr):
    blocks = tl.reshape(x.to(tl.float32), (ROWS, BK//32, 32))
    maximum = tl.max(tl.abs(blocks), 2)
    exponent = tl.ceil(tl.log2(tl.maximum(tl.div_rn(maximum,448.),1.1754943508222875e-38)))
    scale = tl.exp2(exponent)
    q = (blocks/scale[:,:,None]).to(tl.float8e4nv)
    d = q.to(tl.float32)*scale[:,:,None]
    return tl.reshape(q,(ROWS,BK)), (exponent+127.).to(tl.uint8), tl.reshape(d,(ROWS,BK))


@triton.jit
def _kernel(A,W,Y,AD,WD,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    im=tl.program_id(0)
    jn=tl.program_id(1)
    rm=im*BM+tl.arange(0,BM)
    rn=jn*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+rk
        av=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0.)
        wv=tl.load(W+rn[:,None]*K+kk[None,:],(rn[:,None]<N)&(kk[None,:]<K),0.)
        aq,sa,ad=_quant(av,BM,BK)
        wq,sw,wd=_quant(wv,BN,BK)
        acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
        if jn == 0:
            tl.store(AD+rm[:,None]*K+kk[None,:],ad,(rm[:,None]<M)&(kk[None,:]<K))
        if im == 0:
            tl.store(WD+rn[:,None]*K+kk[None,:],wd,(rn[:,None]<N)&(kk[None,:]<K))
    tl.store(Y+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))


@triton.jit
def _load_quant(A,W,AD,WD,rm,rn,block,im,jn,
                M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    kk=block*BK+tl.arange(0,BK)
    av=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0.)
    wv=tl.load(W+rn[:,None]*K+kk[None,:],(rn[:,None]<N)&(kk[None,:]<K),0.)
    aq,sa,ad=_quant(av,BM,BK)
    wq,sw,wd=_quant(wv,BN,BK)
    if jn == 0:
        tl.store(AD+rm[:,None]*K+kk[None,:],ad,(rm[:,None]<M)&(kk[None,:]<K))
    if im == 0:
        tl.store(WD+rn[:,None]*K+kk[None,:],wd,(rn[:,None]<N)&(kk[None,:]<K))
    return aq,sa,wq,sw


@triton.jit
def _buffered_kernel(A,W,Y,AD,WD,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                     BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                     LOOP_STAGES:tl.constexpr):
    im=tl.program_id(0)
    jn=tl.program_id(1)
    rm=im*BM+tl.arange(0,BM)
    rn=jn*BN+tl.arange(0,BN)
    acc=tl.full((BM,BN),0,tl.float32)
    # Prologue fills current quantized operand buffers. Loop-carried values
    # keep current and next tiles distinct; inspect lowering rather than
    # assuming source order proves hardware concurrency.
    aq,sa,wq,sw=_load_quant(A,W,AD,WD,rm,rn,0,im,jn,M,N,K,BM,BN,BK)
    for block in tl.range(1,tl.cdiv(K,BK),num_stages=LOOP_STAGES):
        nq,ns,nw,nws=_load_quant(A,W,AD,WD,rm,rn,block,im,jn,M,N,K,BM,BN,BK)
        acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
        aq,sa,wq,sw=nq,ns,nw,nws
    # Drain the final buffered tile, including a masked K tail.
    acc=tl.dot_scaled(aq,sa,'e4m3',tl.trans(wq),sw,'e4m3',acc)
    tl.store(Y+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))


def quantized_mm(a,w,bm=32,bn=64,bk=128,warps=4,stages=2,return_kernel=False,buffered=False,loop_stages=1):
    m,k=a.shape
    n=w.shape[0]
    if k%32 or not a.is_contiguous() or not w.is_contiguous():
        raise ValueError('Requires contiguous matrices and K divisible by 32')
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    ad=torch.empty_like(a,dtype=torch.bfloat16)
    wd=torch.empty_like(w,dtype=torch.bfloat16)
    grid=(triton.cdiv(m,bm),triton.cdiv(n,bn))
    if buffered:
        kernel=_buffered_kernel[grid](a,w,y,ad,wd,m,n,k,bm,bn,bk,loop_stages,
            num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    else:
        kernel=_kernel[grid](a,w,y,ad,wd,m,n,k,bm,bn,bk,
            num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (y,ad,wd,kernel) if return_kernel else (y,ad,wd)

from torch import nn
import native
from activation import activation_backward, pack_activation


class PipelinedMLPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w1,w2,stochastic,buffered):
        shape=x.shape
        if buffered:
            z,ax,bw=quantized_mm(x.reshape(-1,shape[-1]),w1,bm=32,bn=64,bk=32,stages=1,buffered=True)
        else:
            z,ax,bw=quantized_mm(x.reshape(-1,shape[-1]),w1)
        hq,hs,hx=pack_activation(z)
        cq,sc,cw=native.fp8_pack(w2,False,return_dequant=True)
        y=torch._scaled_mm(hq,cq.T,hs,sc,out_dtype=torch.bfloat16)
        ctx.save_for_backward(ax,bw,z,hx,cw)
        ctx.shape,ctx.stochastic=shape,stochastic
        return y.reshape(*shape[:-1],w2.shape[0])

    @staticmethod
    def backward(ctx,grad):
        x,w,z,h,v=ctx.saved_tensors
        g=grad.reshape(-1,grad.shape[-1])
        dh=native.mx_mm(g,v.T,ctx.stochastic)
        dv=native.mx_mm(g.T,h.T,ctx.stochastic)
        dz=activation_backward(z,dh)
        dx=native.mx_mm(dz,w.T,ctx.stochastic)
        dw=native.mx_mm(dz.T,x.T,ctx.stochastic)
        return dx.reshape(ctx.shape),dw,dv,None,None


class PipelinedMLP(nn.Module):
    def __init__(self,mlp,buffered=False):
        super().__init__()
        self.c_fc,self.c_proj=mlp.c_fc,mlp.c_proj
        self.stochastic=mlp.c_fc.stochastic
        self.buffered=buffered

    def forward(self,x):
        return PipelinedMLPFn.apply(x,self.c_fc.weight,self.c_proj.weight,self.stochastic,self.buffered)


def replace_mlps(model,buffered=False):
    for block in model.transformer.h:
        if getattr(block.mlp.c_fc,'forward_format',None)=='mxfp8':
            block.mlp=PipelinedMLP(block.mlp,buffered=buffered)
