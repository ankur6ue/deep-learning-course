import torch
from torch import nn


class HadamardProdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # Save tensors needed for backward pass in ctx
        ctx.save_for_backward(x, y)
        output = torch.mul(x, y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output
        # torch.mul (or *) computes element wise product, while torch.matmul computes dot product
        # this is equivalent to the below, for B = 1
        # return torch.matmul(torch.diag(y.squeeze()), grad_output), torch.matmul(torch.diag(x.squeeze()), grad_output)

    @staticmethod
    def backward(ctx, grad_output):
        sz = ctx.sz_x
        grad_x = torch.ones(sz)
        return grad_x


class HadamardProdModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return HadamardProdFn.apply(x, y)

