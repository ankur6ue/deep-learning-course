import torch
from torch import nn

class SumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # in backward, we just need the size of x, not the entire tensor. We can attach the size
        # directly on the ctx object
        # save_for_backward must be used to save tensors, while non-tensors (such as sizes) can be saved directly on the
        # context, see Sum function as an example and:
        # https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html
        ctx.sz_x = x.size()
        output = torch.sum(x) # will sum along both dimensions
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sz = ctx.sz_x
        grad_x = torch.ones(sz)
        return grad_x


class SumModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SumFn.apply(x)

