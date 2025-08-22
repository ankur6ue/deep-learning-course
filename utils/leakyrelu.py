import torch
from torch import nn


class LeakyReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.05):
        # Save the input for the backward pass (for gradient calculation)
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.where(x >= 0, x, alpha * x)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        (x,) = ctx.saved_tensors
        # Calculate the gradient: 1 where input > 0, 0 otherwise
        # derivative mask: 1 for x>0, alpha for x<=0 (note the convention)
        grad_input = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, ctx.alpha))
        return grad_output * grad_input


class LeakyReLUModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return LeakyReLUFn.apply(x)