import torch
import numpy as np

# Set a deterministic random seed for reproducibility
torch.manual_seed(42)

# m = input dimension, n = output dimension, B = batch size
m = 5
n = 4
B = 1

# We will draw many random samples of (x, W, b) where:
#   x ~ N(0, 1) in R^{m x B}
#   W ~ N(0, 1) in R^{n x m}
#   b = 0 in R^{n x 1}
#
# For each trial we compute:
#   y = W x + b      (pre-activation)
#   a = ReLU(y)      (post-activation)
#
# Then we empirically estimate the mean/variance of y and a and compare to theory.

N = 10000  # number of Monte Carlo trials

# y_hist and a_hist will store all y and a vectors column-wise:
#   shape: (n, N)
y_hist = torch.empty(n, 0)
a_hist = torch.empty(n, 0)

for trials in range(0, N):
    # Sample input x: shape (m, B)
    x = torch.randn(m, B, requires_grad=False)

    # Sample weight matrix W: shape (n, m)
    W = torch.randn(n, m, requires_grad=False)

    # Bias is zero: shape (n, 1)
    b = torch.zeros(n, 1)

    # Compute pre-activation:
    #   y = W x + b, shape (n, B) = (n, 1)
    # Note: in PyTorch, "@" = matrix multiplication, "*" = elementwise (Hadamard) product
    y = W @ x + b

    # Append y as another column in y_hist
    y_hist = torch.cat([y_hist, y], dim=1)

    # Apply ReLU nonlinearity elementwise
    a = y.clamp(min=0)

    # Append a as another column in a_hist
    a_hist = torch.cat([a_hist, a], dim=1)

# Empirical mean and variance across trials, for each of the n output units
# mean_y, var_y, mean_a, var_a all have shape (n,)
mean_y = y_hist.mean(dim=1)
var_y = y_hist.var(dim=1)
mean_a = a_hist.mean(dim=1)
var_a = a_hist.var(dim=1)

# Theoretical results for ReLU applied to a Gaussian:
# If y ~ N(0, σ^2), then:
#   E[ReLU(y)] = σ / sqrt(2π)
#   Var[ReLU(y)] = σ^2 * (1/2 - 1/(2π))
#
# Here σ^2 ≈ Var(y), empirically estimated.
print(f'mean of first element of y: {mean_y[0]}')
# Compare empirical mean of first activation to theoretical expression
print(f'mean of first element of activation array: {mean_a[0]}')
print(f'mean of first element of activation array from math: {np.sqrt(var_y[0] / (2 * np.pi))}')
#
print(f'variance of first element of activation array: {var_a[0]}')
print(f'variance of first element of activation array from math: {m * 1 * (1/2 - 1/(2 * np.pi))}')

print('done')

# Sanity checks you might do (not printed here):
# - Verify mean_y (n*1 vector) is close to 0, since y has zero-mean by construction.
# - Verify var_y (n*1 vector) is close to m (because Var(y_j) ≈ sum_i Var(W_ji x_i) = m).
# - Verify mean_a and var_a match the theoretical expressions above for ReLU(y).
