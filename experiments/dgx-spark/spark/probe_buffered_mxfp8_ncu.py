import torch
from pipelined_mxfp8 import quantized_mm

a=torch.randn(8192,768,device='cuda',dtype=torch.float32)
w=torch.randn(3072,768,device='cuda',dtype=torch.float32)*.03
for buffered in (False,True):
    for _ in range(10):
        quantized_mm(a,w,bm=16,bn=64,bk=128,stages=1,buffered=buffered)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for buffered in (False,True):
    quantized_mm(a,w,bm=16,bn=64,bk=128,stages=1,buffered=buffered)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
