"""Whole-MLP gradient, RNG and fullgraph parity for operand-loop quantization."""
import copy,json,torch,native,sys
from activation import FusedActivationMLP
from pipelined_mxfp8 import PipelinedMLP
from record_model import MLP,GPTConfig
native.set_pack_backend("triton-tiled")
torch.manual_seed(42)
mlp = MLP(GPTConfig(n_embd=256,n_head=2)).cuda()
for name in ('c_fc','c_proj'):
    layer = getattr(mlp,name)
    layer.weight.data.normal_(0,.03)
    setattr(mlp,name,native.NativeLinear(layer,True,'mxfp8'))
old = FusedActivationMLP(copy.deepcopy(mlp))
new = PipelinedMLP(copy.deepcopy(mlp),buffered='--buffered' in sys.argv,reuse_weights='--reuse-weights' in sys.argv,producer_consumer='--producer-consumer' in sys.argv)
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
print(json.dumps({"passed":True,"gradient_relative_l2":grad_errors,"compiled_max_abs_errors":compiled_errors,"rng_exact":True}))
