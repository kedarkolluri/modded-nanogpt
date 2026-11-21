"""Explicit CUDA producer/consumer warps; shared FP8 activation ring buffers."""
import ctypes
from pathlib import Path
import torch
import native

CONFIGS=[(128,32,4),(256,32,4),(512,32,4),(128,64,4),(256,64,4),(512,64,4),
         (256,32,2),(512,32,2),(256,64,2),(512,64,2)]
_LIBRARY=None


def _library():
    global _LIBRARY
    if _LIBRARY is None:
        path=Path(__file__).parent/'cuda'/'producer_consumer_mxfp8.so'
        _LIBRARY=ctypes.CDLL(str(path))
        _LIBRARY.run_pc.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_void_p]
        _LIBRARY.run_pc.restype=ctypes.c_int
    return _LIBRARY


def activation_mm(a,wq,sw,config=4,serial=False,trace=False):
    if a.dtype!=torch.float32 or not a.is_cuda or not a.is_contiguous():
        raise ValueError('Activation input must be contiguous CUDA float32')
    if wq.dtype!=torch.float8_e4m3fn or not wq.is_contiguous():
        raise ValueError('Weight input must be contiguous E4M3')
    m,k=a.shape
    n=wq.shape[0]
    if wq.shape[1]!=k or k%32:
        raise ValueError('Matching K divisible by 32 is required')
    bn,bk,pw=CONFIGS[config]
    y=torch.empty((m,n),device=a.device,dtype=torch.bfloat16)
    ad=torch.empty_like(a,dtype=torch.bfloat16)
    events=torch.zeros(((k+bk-1)//bk,7),device=a.device,dtype=torch.int64) if trace else y
    status=_library().run_pc(a.data_ptr(),wq.data_ptr(),sw.data_ptr(),y.data_ptr(),ad.data_ptr(),
                            m,n,k,events.data_ptr(),config,int(serial),int(trace),torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'Producer/consumer launch failed with CUDA status {status}')
    return (y,ad,events) if trace else (y,ad)


def weight_reuse_mm(a,w,config=4,serial=False):
    wq,sw,wd=native.fp8_pack(w,False,return_dequant=True)
    y,ad=activation_mm(a,wq,sw,config=config,serial=serial)
    return y,ad,wd


@torch.library.custom_op('spark_pc::activation_mm',mutates_args=())
def compiled_activation_mm(a:torch.Tensor,wq:torch.Tensor,sw:torch.Tensor,config:int,serial:bool)->tuple[torch.Tensor,torch.Tensor]:
    return activation_mm(a,wq,sw,config=config,serial=serial)


@compiled_activation_mm.register_fake
def _fake(a,wq,sw,config,serial):
    return (torch.empty((a.shape[0],wq.shape[0]),device=a.device,dtype=torch.bfloat16),
            torch.empty_like(a,dtype=torch.bfloat16))


def training_mm(a,w):
    wq,sw,wd=native.fp8_pack(w,False,return_dequant=True)
    y,ad=compiled_activation_mm(a,wq,sw,2,False)
    return y,ad,wd
