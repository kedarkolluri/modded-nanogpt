"""Validate the FP32-master adapter against TorchAO's own MXFP8Linear."""
import copy
import json
import torch
import torchao
from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
from torchao_backend import TorchAOLinear

torch.manual_seed(42)
master = torch.nn.Linear(768,3072,bias=False,device='cuda')
adapter = TorchAOLinear(master)
reference = MXFP8Linear(768,3072,bias=False,device='cuda',dtype=torch.bfloat16)
reference.weight.data.copy_(master.weight.data)
compiled = torch.compile(copy.deepcopy(adapter),fullgraph=True,dynamic=False)
x = torch.randn(128,768,device='cuda',dtype=torch.bfloat16,requires_grad=True)
grad = torch.randn(128,3072,device='cuda',dtype=torch.bfloat16)
values = []
for module in (reference, adapter, compiled):
    x.grad = None
    y = module(x)
    y.backward(grad)
    values.append((y.detach().clone(),x.grad.clone(),module.weight.grad.float().clone()))
for current in values[1:]:
    for a,b in zip(values[0], current):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
assert adapter.weight.dtype == torch.float32
assert all(torch.isfinite(v).all() for row in values for v in row)
print(json.dumps({'passed':True,'torch':torch.__version__,'torchao':torchao.__version__,
    'exact_official_adapter_compiled_outputs_input_weight_gradients':True,
    'master_weight_dtype':str(adapter.weight.dtype),
    'cuda_quantize_kernel_registered':torch._C._dispatch_has_kernel_for_dispatch_key('torchao::mxfp8_quantize','CUDA'),
    'dynamo_counters':{k:dict(v) for k,v in torch._dynamo.utils.counters.items()}}))
