"""BF16 control: exact output/gradient parity for common activation fusion."""
import copy
import json
import torch
from torch.nn import functional as F
from activation import BF16ReluSquareFn, FusedBF16ActivationMLP
from record_model import MLP, GPTConfig


def main():
    torch.manual_seed(31)
    for rows, cols in [(1, 32), (129, 768), (256, 3072)]:
        z = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn_like(z)
        outputs = []
        for fused in (False, True):
            z.grad = None
            y = BF16ReluSquareFn.apply(z) if fused else F.relu(z).square()
            y.backward(g)
            outputs.append((y.detach().clone(), z.grad.clone()))
        for a,b in zip(*outputs):
            assert torch.equal(a,b)
    mlp = MLP(GPTConfig(n_embd=128, n_head=2)).cuda()
    for p in mlp.parameters():
        p.data.normal_(0,.03)
    fused = FusedBF16ActivationMLP(copy.deepcopy(mlp))
    x = torch.randn(2,64,128,device='cuda',requires_grad=True)
    g = torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
    outputs = []
    for model in (mlp,fused):
        x.grad = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model(x)
        y.backward(g)
        outputs.append((y.detach().clone(),x.grad.clone(),*[p.grad.clone() for p in model.parameters()]))
    for a,b in zip(*outputs):
        assert torch.equal(a,b)
    print(json.dumps({'passed': True,'exact_bf16_activation_and_mlp_gradient_parity': True}))


if __name__ == '__main__':
    main()
