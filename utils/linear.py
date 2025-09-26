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
    def __init__(self, _in, _out, name, scale=1):
        super().__init__()
        # Need to set a scale for class_2/simple_nn1_mlflow.py example, when Simple Optimizer is being used
        # We implement the matrix mul in a linear layer as Wx + b, so the second dimension of W must match the input
        # to the layer
        self.weight = torch.randn(_out, _in) * np.sqrt(1 / _in) * scale
        self.bias = torch.randn(_out, 1) * np.sqrt(1 / _in)
        # self.bias = torch.zeros(_out, 1)
        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self.name = name

    def forward(self, x):
        return LinearFn.apply(x, self.weight, self.bias)