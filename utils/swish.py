import torch
from torch import nn
import numpy as np
# Swish
class SwishFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = torch.sigmoid(x)
        y = x * s
        ctx.save_for_backward(x, y, s)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, s = ctx.saved_tensors
        dy_dx = s + y * (1.0 - s)    # = s + x*s*(1-s)
        return grad_output * dy_dx


class SwishModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishFn.apply(x)