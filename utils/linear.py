import torch
from torch import nn
import numpy as np


# General patter:
# write the forward and backward calculation explicitly as static functions which are invoked using the apply function
# from the corresponding module. This is standard practice in PyTorch.
class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        # Save tensors needed for backward pass in ctx
        ctx.save_for_backward(x, W, b)
        output = torch.matmul(W, x)
        if b is not None:
            output += b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, W, b = ctx.saved_tensors
        grad_input = W.T @ grad_output
        grad_weight = grad_output @ input.T
        grad_bias = torch.clone(grad_output).sum(dim=1, keepdim=True)
        return grad_input, grad_weight, grad_bias


class LinearModule(nn.Module):
    def __init__(self, _in, _out, name):
        super().__init__()
        scale = 1  # 0.5
        self.weight = torch.randn(_out, _in) * np.sqrt(1 / _in) * scale
        self.bias = torch.randn(_out, 1) * np.sqrt(1 / _in) * scale
        # self.weight = l.weight.data
        self.weight.requires_grad = True  # _out * _in
        # self.bias = l.bias.data.unsqueeze(1) # _out * 1. The unsqueeze is to add the 2nd dimension
        self.bias.requires_grad = True
        self.name = name

    def forward(self, x):
        return LinearFn.apply(x, self.weight, self.bias)