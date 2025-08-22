import torch
from torch import nn
import numpy as np
# Gaussian error linear unit. Uses the approximate form of GeLU
class GeluFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save for backward pass (for gradient calculation)
        k = np.sqrt(2.0 / np.pi)
        gx = k * (x + 0.044715 * x.pow(3))
        t = torch.tanh(gx)
        y = 0.5 * x * (1.0 + t)
        ctx.save_for_backward(x, t)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        x, t = ctx.saved_tensors
        k = np.sqrt(2.0 / np.pi)
        gp = k * (1.0 + 0.134145 * x.pow(2))  # g'(x)
        dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t.pow(2)) * gp
        return grad_output * dy_dx


class GeluModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GeluFn.apply(x)