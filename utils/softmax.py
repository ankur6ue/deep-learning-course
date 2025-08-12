import torch
from torch import nn


# Converts logits into probabilities and then cross entropy loss
class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        exp_scores = torch.exp(x)
        # we are using columns are the batch dimension
        probs = exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)
        # Save for use in the backward pass (for gradient calculation)
        ctx.save_for_backward(probs)
        return probs


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        x, = ctx.saved_tensors
        # for each element of the batch (row vector of size m*1), the jacobian (m*m) is:
        # denoting o as output of softmax and x as input
        # do_dx = torch.diag(x) - torch.matmul(x, x.T)
        # when multiplied with grad_output, this becomes
        # do_dx = torch.mul(x, grad_output) - x @ x.T @ grad_output
        # when B > 1, we need to concatenate
        do_dx = []
        B = x.size(1)
        for i in range(0, B):
            # By using x[[i]], the [i] is interpreted as a list containing a single index 0. This tells PyTorch to
            # select the element at index i and keep it within its own dimension, resulting in a shape of n*1
            do_dx.append(torch.mul(x[:, [i]], grad_output[:,[i]]) - x[:,[i]] @ x[:,[i]].T @ grad_output[:,[i]])
        out = torch.cat(do_dx, dim=1)
        return out


class SoftmaxModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SoftmaxFn.apply(x)