import torch
from torch import nn


class CrossEntropyLossFn(torch.autograd.Function):
    """
    A custom cross-entropy loss (logits + index labels), matching the common
    "softmax + NLL" formulation, with numerical stability via log-sum-exp.

    Expected shapes (column-major batches):
        - y:      (C, B) float tensor of logits (C = #classes, B = batch size)
        - labels: (B,)   int64 tensor of class indices in [0, C-1]

    Returns:
        - Scalar tensor: mean cross-entropy over the batch.

    Notes:
        - We compute log-probabilities as: log_probs = y - logsumexp(y, dim=0).
        - The loss is: L = -(1/B) * sum_b log_probs[labels[b], b]
        - Backward returns dL/dy (shape (C,B)) and None for labels.
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Stable log-softmax along class dimension (rows), per column (sample).
        logsumexp = torch.logsumexp(y, dim=0, keepdim=True)  # (1, B)
        log_probs = y - logsumexp                            # (C, B)

        # Save for backward
        ctx.save_for_backward(log_probs, labels)

        B = y.size(1)
        # Gather log-probs at target indices and average (negative log-likelihood)
        loss = -log_probs[labels.long(), torch.arange(B, device=y.device)].mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        log_probs, labels = ctx.saved_tensors
        probs = log_probs.exp()                     # (C, B)
        B = probs.size(1)

        # dL/d(logits) = (probs - one_hot) / B; scale by grad_output (scalar)
        probs[labels.long(), torch.arange(B, device=probs.device)] -= 1.0
        grad_y = (probs / B) * grad_output
        return grad_y, None  # no gradient for labels


class MSELossFn(torch.autograd.Function):
    """
    Mean Squared Error (MSE) loss over all elements of (C, B) tensors.

    Expected shapes:
        - y: (C, B) predictions
        - g: (C, B) ground-truth

    Returns:
        - Scalar tensor: mean((y - g)^2) over both dimensions.

    Backward:
        - dL/dy = (2/(C*B)) * (y - g) * grad_output
        - dL/dg = None (no gradient needed by default for targets)
    """

    @staticmethod
    def forward(ctx, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(y, g)
        return torch.mean((y - g) ** 2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        y, g = ctx.saved_tensors
        C, B = y.shape
        scale = 2.0 / (C * B)
        grad_y = scale * (y - g) * grad_output
        return grad_y, None


class FusedSoftmaxMSELossFn(torch.autograd.Function):
    """
    MSE between softmax(logits) and target probabilities, with an efficient
    backward (no explicit Jacobian materialization).

    Expected shapes (column-major batches):
        - logits:       (C, B) float tensor (pre-softmax)
        - target_probs: (C, B) float tensor (each column sums to 1)

    Returns:
        - Scalar tensor: mean( (softmax(logits) - target_probs)^2 )

    Backward:
        Let p = softmax(logits), g = 2*(p - t)/(C*B). For each column b:
            dL/d(logits)_b = J_p(z_b) @ g_b
                           = g_b - p_b * (p_b^T g_b)
        Implemented as a vectorized column-wise operation.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        # Softmax across classes (rows) for each column (sample)
        exp_logits = torch.exp(logits)
        probs = exp_logits / exp_logits.sum(dim=0, keepdim=True)

        # Save tensors needed for backward
        ctx.save_for_backward(probs, target_probs)

        # Mean MSE over all elements
        loss = torch.mean((probs - target_probs) ** 2)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        probs, target_probs = ctx.saved_tensors
        C, B = probs.shape

        # Upstream gradient for MSE wrt probs: 2*(p - t)/(C*B)
        g = (2.0 / (C * B)) * (probs - target_probs)

        # Jacobian-vector product for softmax along columns:
        # J @ g = p * g - p * (sum_j p_j * g_j)  (computed per column, becomes the dot product between p_j and g_j)
        dot = (probs * g).sum(dim=0, keepdim=True)  # (1, B)
        grad_logits = probs * g - probs * dot

        # Scale by incoming scalar grad_output
        grad_logits = grad_logits * grad_output

        return grad_logits, None

class CrossEntropyLossModule(nn.Module):
    """
    nn.Module wrapper for CrossEntropyLossFn.

    Forward:
        y:      (C, B) logits
        labels: (B,)   int64 class indices
        returns scalar mean cross-entropy
    """

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return CrossEntropyLossFn.apply(y, labels)


class FusedSoftmaxMSELossModule(nn.Module):
    """
    nn.Module wrapper for FusedSoftmaxMSELossFn.

    Forward:
        y:      (C, B) logits
        labels: (C, B) target probabilities (columns sum to 1)
        returns scalar mean MSE between softmax(y) and labels
    """

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return FusedSoftmaxMSELossFn.apply(y, labels)


class MSELossModule(nn.Module):
    """
    nn.Module wrapper for MSELossFn.

    Forward:
        y:      (C, B) predictions
        labels: (C, B) targets
        returns scalar mean squared error
    """

    def __init__(self):
        super().__init__()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return MSELossFn.apply(y, labels)

