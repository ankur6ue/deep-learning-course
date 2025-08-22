import torch
from torch import nn


class TanhFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save for backward pass (for gradient calculation)
        y = torch.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        (y,) = ctx.saved_tensors
        return grad_output * (1 - y.pow(2))


class TanhModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return TanhFn.apply(x)