"""Thin TorchAO MXFP8 adapter retaining the experiment's FP32 master weights.

TorchAO's primitive requires BF16 operands. Casts remain differentiable; all
quantization and three GEMMs are implemented by unmodified TorchAO code.
TorchAO RCEIL uses deterministic rounding and saved high-precision operands,
which differs from our reconstructed-forward/stochastic-backward recipe.
"""
import torch
from torch import nn
from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.quantization.quantize_.common import KernelPreference


class TorchAOLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError('This experiment uses bias-free MLP projections')
        self.weight = linear.weight

    def forward(self, x):
        return _to_mxfp8_then_scaled_mm(
            x.to(torch.bfloat16), self.weight.to(torch.bfloat16),
            kernel_preference=KernelPreference.AUTO,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
            wgrad_with_hp=False)


def replace_mlps(model, tail_precision='mxfp8', fuse_activation=False):
    from activation import FusedBF16ActivationMLP
    count = len(model.transformer.h)
    first_tail = count - (count+5)//6
    converted = 0
    for index, block in enumerate(model.transformer.h):
        if index < first_tail or tail_precision == 'mxfp8':
            for name in ('c_fc', 'c_proj'):
                setattr(block.mlp, name, TorchAOLinear(getattr(block.mlp, name)))
                converted += 1
        if fuse_activation:
            # Generic pointwise fusion; no custom MXFP8 packing is used here.
            block.mlp = FusedBF16ActivationMLP(block.mlp)
    return converted
