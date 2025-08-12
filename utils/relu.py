import torch
from torch import nn


class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save the input for the backward pass (for gradient calculation)
        ctx.save_for_backward(x)
        # Apply ReLU: max(0, input_tensor)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        x, = ctx.saved_tensors
        # Calculate the gradient: 1 where input > 0, 0 otherwise
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input


class ReLUModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ReLUFn.apply(x)