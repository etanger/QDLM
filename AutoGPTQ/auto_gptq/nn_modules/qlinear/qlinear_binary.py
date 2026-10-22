"""Packed W1 storage with a portable dequantize-and-matmul reference backend.

This is an accuracy backend, not an accelerated binary GEMM kernel. Each
weight uses one sign bit; each output-channel/input-group uses one scale.
The two values are -alpha and +alpha. No residual weights are retained.
"""

import math

import torch
from torch import nn


class QuantLinear(nn.Module):
    QUANT_TYPE = "binary-reference"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias,
                 use_cuda_fp16=True, trainable=False, weight_dtype=torch.float16):
        super().__init__()
        if bits != 1 or trainable:
            raise ValueError("The binary reference backend supports inference-only W1.")
        if min(infeatures, outfeatures) <= 0 or (group_size != -1 and group_size <= 0):
            raise ValueError("Invalid binary linear dimensions or group size.")
        self.bits = 1
        self.maxq = 1
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.group_size = infeatures if group_size == -1 else group_size
        self.register_buffer("qweight", torch.zeros(
            (math.ceil(infeatures / 32), outfeatures), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(
            (math.ceil(infeatures / self.group_size), outfeatures), dtype=weight_dtype))
        self.register_buffer("g_idx", torch.arange(infeatures, dtype=torch.int32) // self.group_size)
        self.register_buffer("bias", torch.zeros(outfeatures, dtype=weight_dtype) if bias else None)

    @torch.no_grad()
    def pack(self, linear, scales, zeros, g_idx=None):
        if not isinstance(linear, nn.Linear):
            raise TypeError("Binary packing currently supports nn.Linear only.")
        if tuple(linear.weight.shape) != (self.outfeatures, self.infeatures):
            raise ValueError("Binary packing received an incompatible weight shape.")
        if scales.T.shape != self.scales.shape or zeros.shape != scales.shape:
            raise ValueError("Binary scale/zero shapes do not match the group layout.")
        if not torch.all(zeros == 0.5):
            raise ValueError("Binary GPTQ zero points must be 0.5.")
        if not torch.isfinite(scales).all() or not torch.all(scales > 0):
            raise ValueError("Binary GPTQ scales must be finite and positive.")
        if not torch.isfinite(linear.weight).all():
            raise ValueError("Cannot pack nonfinite binary weights.")
        groups = self.g_idx if g_idx is None else g_idx
        if groups.shape != self.g_idx.shape or groups.min() < 0 or groups.max() >= self.scales.shape[0]:
            raise ValueError("Invalid binary group indices.")
        device = self.qweight.device
        self.g_idx.copy_(groups.to(device=device, dtype=torch.int32))
        self.scales.copy_((scales.T / 2).to(device=device, dtype=self.scales.dtype))
        # Pack 32 input positions per int32. Padding bits are never evaluated.
        signs = (linear.weight.detach().T >= 0).to(device=device, dtype=torch.int32)
        self.qweight.zero_()
        for shift in range(32):
            part = signs[shift::32]
            self.qweight[:part.shape[0]].bitwise_or_(part << shift)
        if self.bias is not None:
            self.bias.copy_(linear.bias.to(self.bias))

    def dequantize_weight(self):
        """Return the [input, output] matrix using the stored sign bits."""
        shifts = torch.arange(32, device=self.qweight.device, dtype=torch.int32)
        codes = ((self.qweight[:, None, :] >> shifts[None, :, None]) & 1)
        codes = codes.reshape(-1, self.outfeatures)[:self.infeatures]
        signs = codes.to(self.scales.dtype).mul_(2).sub_(1)
        return signs * self.scales[self.g_idx.long()]

    def forward(self, x):
        weight = self.dequantize_weight().to(dtype=x.dtype)
        output = torch.matmul(x, weight)
        if self.bias is not None:
            output = output + self.bias.to(dtype=x.dtype)
        return output
