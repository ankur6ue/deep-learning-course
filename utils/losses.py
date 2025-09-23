import torch
from torch import nn


class CrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, labels):
        # Save the input for the backward pass (for gradient calculation)
        sum_exp_scores = torch.sum(torch.exp(y), dim=0)
        # internally pytorch uses log probs for numerical stability.. so we'll also do that
        # probs = exp_scores / torch.sum(exp_scores, dim=0, keepdims=True)
        log_probs = y - torch.log(sum_exp_scores)
        ctx.save_for_backward(log_probs, labels)
        B = y.size(1)
        # Below assumes that labels are one hot vectors, so we are simply indexing the probs array at the non-zero index
        # and summing across the batch
        return -torch.sum(log_probs[(labels.int(), range(0, B))])/B

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        log_probs, labels = ctx.saved_tensors
        probs = torch.exp(torch.clone(log_probs))
        B = probs.size(1)
        probs[[labels.int(), range(0, B)]] -= 1
        return probs/B, torch.zeros(labels.size())


class MSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, g):
        # y: output of the net
        # g: groundtruth
        # Save the input for the backward pass (for gradient calculation)
        ctx.save_for_backward(y, g)
        return torch.mean((y-g) ** 2) # mean operates over both dimensions.. vector and batch.

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        # If the forward function operates on n inputs the number of outputs in the corresponding backward function must
        # also be n (assuming all inputs require gradients)
        y, g = ctx.saved_tensors
        sz = y.size(0)
        B = y.size(1)
        return 2*(y - g)/(sz*B), torch.zeros(g.size())

class FusedXEntropyMSELossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, labels):
        # element wise product, then sum along columns
        exp_logits = torch.exp(y)
        probs = exp_logits / torch.sum(exp_logits, dim=0, keepdims=True)
        sum_sq_probs = torch.sum(probs * probs, dim=0, keepdim=True)
        # element wise product, then sum along columns
        sum_prob_labels = torch.sum(probs * labels, dim=0, keepdim=True)
        # notice the trade-off between memory consumed and computation..
        # we can calculate probs, sum_sq_probs etc from y and labels.. but we can avoid that computation at the
        # expense of saving more tensors, and consuming more memory
        ctx.save_for_backward(y, probs, labels, sum_sq_probs, sum_prob_labels)
        ctx.sum_prob_labels = sum_prob_labels
        ctx.sum_sq_probs = sum_sq_probs
        return torch.mean((probs-labels) ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        y, probs, labels, sum_sq_probs, sum_prob_labels = ctx.saved_tensors
        B = probs.size(1)
        K = probs.size(0)
        # When probs and labels are N*B and sum_sq_probs and sum_prob_labels are 1*B, they'll be broadcasted along B
        # dimension
        return 2 * probs * (probs - labels - sum_sq_probs + sum_prob_labels)/(K * B), torch.zeros(labels.size())

class CrossEntropyLossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):

        return CrossEntropyLossFn.apply(y, labels)

class FusedXEntropyMSELossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return FusedXEntropyMSELossFn.apply(y, labels)


class MSELossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return MSELossFn.apply(y, labels)


class MSELossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return MSELossFn.apply(y, labels)

