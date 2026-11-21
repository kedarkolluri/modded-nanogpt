"""Validate native MMA, epilogue packing, padding and complete MLP gradients."""
import copy
import json
import torch
import native
from activation import pack_activation,FusedActivationMLP
from fused_mlp_epilogue import gemm_pack,FusedEpilogueMLP
from record_model import MLP,GPTConfig

native.set_pack_backend('triton-tiled')
torch.manual_seed(42)
errors = []
for m,n,k in [(1,128,96),(129,512,256),(256,1024,768),(8192,3072,768)]:
    a,sa,ax = native.fp8_pack(torch.randn(m,k,device='cuda',dtype=torch.bfloat16),False,True)
    b,sb,bw = native.fp8_pack(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.03,False,True)
    z,h,hs,hd,kernel = gemm_pack(a,sa,b,sb,return_kernel=True)
    ref = torch._scaled_mm(a,b.T,sa,sb,out_dtype=torch.bfloat16)
    rel = float((z.float()-ref.float()).norm()/ref.float().norm())
    assert rel < .003,(m,n,k,rel)
    expected = pack_activation(z)
    for actual,want in zip((h,hs,hd),expected):
        assert torch.equal(actual.view(torch.uint8),want.view(torch.uint8)),(m,n,k,'epilogue')
    errors.append({'shape':[m,n,k],'z_relative_l2':rel})
ptx = kernel.asm['ptx']
assert 'mma.sync.aligned.m16n8k32' in ptx and 'kind::mxf8f6f4.block_scale' in ptx
assert 'tcgen05' not in ptx
mlp = MLP(GPTConfig(n_embd=256,n_head=2)).cuda()
for name in ('c_fc','c_proj'):
    layer = getattr(mlp,name)
    layer.weight.data.normal_(0,.03)
    setattr(mlp,name,native.NativeLinear(layer,True,'mxfp8'))
old = FusedActivationMLP(copy.deepcopy(mlp))
new = FusedEpilogueMLP(copy.deepcopy(mlp))
assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
x = torch.randn(2,64,256,device='cuda',requires_grad=True)
g = torch.randn_like(x,dtype=torch.bfloat16)
state = torch.cuda.get_rng_state()
values = []
for module in (old,new):
    torch.cuda.set_rng_state(state)
    x.grad = None
    y = module(x)
    y.backward(g)
    values.append([y.detach().clone(),x.grad.clone(),*[p.grad.clone() for p in module.parameters()],torch.cuda.get_rng_state()])
assert torch.equal(values[0][-1],values[1][-1]),'RNG consumption'
grad_errors = []
for a,b in zip(values[0][:-1],values[1][:-1]):
    rel = float((a.float()-b.float()).norm()/a.float().norm())
    assert rel < .015,rel
    grad_errors.append(rel)
# Compilation must preserve the new custom kernel's forward and backward too.
nearest = copy.deepcopy(new)
nearest.stochastic = False
nearest.zero_grad(set_to_none=True)
compiled = torch.compile(copy.deepcopy(nearest),fullgraph=True,dynamic=False)
compiled_values = []
for module in (nearest,compiled):
    x.grad = None
    y = module(x)
    y.backward(g)
    compiled_values.append([y.detach().clone(),x.grad.clone(),*[p.grad.clone() for p in module.parameters()]])
compiled_errors = []
for a,b in zip(*compiled_values):
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    compiled_errors.append(float((a.float()-b.float()).abs().max()))
result = {'passed':True,'gemm_cases':errors,'exact_epilogue_packing':True,
          'whole_mlp_output_input_weight_grad_relative_l2':grad_errors,
          'rng_consumption_exact':True,'native_mxfp8_mma':True,
          'compiled_nearest_max_abs_output_input_weight_grad_errors':compiled_errors,
          'mma_instruction':next(l.strip() for l in ptx.splitlines() if 'mma.sync' in l),
          'cp_async_instructions':sum('cp.async' in l for l in ptx.splitlines()),
          'registers':kernel.n_regs,'spills':kernel.n_spills}
print(json.dumps(result))
