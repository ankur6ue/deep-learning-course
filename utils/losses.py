import torch
from torch import nn


class CrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, labels):
        # Save the input for the backward pass (for gradient calculation)
        exp_scores = torch.exp(y)
        probs = exp_scores / torch.sum(exp_scores, dim=0, keepdims=True)

        ctx.save_for_backward(probs, labels)
        B = y.size(1)
        return -torch.sum(torch.log(probs[(labels, range(0, B))]))/B

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        probs_, labels = ctx.saved_tensors
        probs = torch.clone(probs_)
        B = probs.size(1)
        probs[[labels, range(0, B)]] -= 1
        return probs/B, torch.zeros(labels.size())


class MSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, g):
        # y: output of the net
        # g: goundtruth
        # Save the input for the backward pass (for gradient calculation)
        ctx.save_for_backward(y, g)
        B = y.size(1)
        return torch.matmul((y-g), (y-g).T)/B

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        # If the forward function operates on n inputs the number of outputs in the corresponding backward function must
        # also be n (assuming all inputs require gradients)
        y, g = ctx.saved_tensors
        B = y.size(1)
        return 2*(y - g)/B, torch.zeros(g.size())


class CrossEntropyLossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return CrossEntropyLossFn.apply(y, labels)


class MSELossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return MSELossFn.apply(y, labels)