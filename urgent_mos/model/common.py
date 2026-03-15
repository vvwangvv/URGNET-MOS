from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeActivation(nn.Module):
    """
    Bounded Range Activations:
      - Two-sided: scaled sigmoid
      - Lower-only: softplus lower clamp
      - Upper-only: softplus upper clamp
      - Identity if both bounds are infinite
    """

    def __init__(self, min_value=-math.inf, max_value=math.inf):
        super().__init__()
        self.register_buffer("min_value", torch.as_tensor(float(min_value)))
        self.register_buffer("max_value", torch.as_tensor(float(max_value)))
        if math.isfinite(min_value) and math.isfinite(max_value):
            if not (max_value > min_value):
                raise ValueError("max_value must be larger than min_value")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_is_finite = torch.isfinite(self.min_value)
        max_is_finite = torch.isfinite(self.max_value)
        max_value = self.max_value.to(x.dtype)  # amp friendly
        min_value = self.min_value.to(x.dtype)

        if not min_is_finite and not max_is_finite:
            return x
        if min_is_finite and max_is_finite:
            return min_value + (max_value - min_value) * torch.sigmoid(x)
        elif min_is_finite:
            return min_value + F.softplus(x - min_value)
        else:
            return max_value - F.softplus(max_value - x)


# https://github.com/facebookresearch/fairseq2/blob/077ac04e89a4ebfdc0691ee0bdb84883391e8c2a/src/fairseq2/nn/utils/grad.py#L54
def scale_grad(x: torch.nn.Tensor, scale: float) -> torch.nn.Tensor:
    """Scale the gradient of ``x`` during backpropagation.

    This is typically used to allow one part of a model to learn at a lower rate
    than the rest.

    :param x:
        The input tensor.
    :param scale:
        The scale factor of the gradient.
    """
    return _GradScaleFunction.apply(x, scale)


class _GradScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.nn.Tensor, scale: float) -> torch.nn.Tensor:
        if not x.dtype.is_floating_point:
            raise TypeError(f"`x` must be a float tensor, but is a `{x.dtype}` tensor instead.")

        ctx.scale = scale

        return x.detach().clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.nn.Tensor) -> tuple[torch.nn.Tensor, None]:
        return grad_output * ctx.scale, None
