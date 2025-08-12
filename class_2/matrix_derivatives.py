import torch
import numpy as np
from torch import nn

# Part 1
############################################
# For Y = WX + b, we calculate the partial derivative of Y against X and W, and compare that with
# the value calculated using first principles (definition of a derivative) and using the Pytorch backward function

# For a n*m matrix W, n*1 vector b and m*1 vector X, calculate the
# product WX and its derivative against X and W
# Set seed so that running this code repeatedly produces the same result
torch.manual_seed(42)
m = 5
n = 4
B = 1
X = torch.randn(m, B, requires_grad=True)
W = torch.randn(n, m, requires_grad=True)
b = torch.randn(n, B, requires_grad=True)

# y is n*1
# In Pytorch, @ means matrix multiplication, * means hadamard product
Y = W @ X + b
# dy_dX is a m*n matrix in denominator notation
dY_dX = W.T
# verify against value of derivative calculated using first principles
for i in range(0, m):
    for j in range(0, n):
        # 1. Change X(i, 0) by a small amount and observe change in Y(j,0), calculate derivative manually
        epsilon = 1e-4
        X_ = X.clone()
        X_.data[i,0] = X_.data[i,0] + epsilon
        Y_ = W @ X_ + b
        # first element of dY_dX
        dYj_dXi = (Y_[j,0] - Y[j,0])/(X_[i,0] - X[i,0])
        print(f"dY_dX[{i},{j}] = {dYj_dXi: >4}")

#dY_dW is a 3D matrix..so we don't worry about that now
# Note: We'd also like to calculate dY_dX using Pytorch's backward function, but that only oeprates
# on a scalar, while Y is a vector.. we'll do that in part 2
############################################

# Part 2
# Now we'll add a sum function to Y, so the output becomes scalar. That will make it possible to calculate the
# derivative against W, X and b and also use the pytorch backward function

# Now let's add a sum function to Y
Y = W @ X + b
L = torch.sum(Y)
# Now we can calculate dL_dX (n*1), dL_dW (n*m), dL_db (n*1)
# First using chain rule, as in the lecture slides
dL_dY = torch.ones(n,1) # n*1 vector
dL_dX = W.T @ dL_dY

# Verification using first principles: Tweak one element of X and let's look at the change in L
for i in range(0, m):
    # 1. Change X(i, 0) by a small amount and observe change in L, calculate derivative manually
    epsilon = 1e-4
    X_ = X.clone()
    X_.data[i,0] = X_.data[i,0] + epsilon
    Y_ = W @ X_ + b
    L_ = torch.sum(Y_)
    # first element of dY_dX
    dL_dXi = (L_ - L)/(X_[i,0] - X[i,0])
    print(f"dL_dX[{i}] = {dL_dXi: >4}")

# compare against dL_dX (visual inspection

# Now use Pytorch's backward function..This will calculate the derivative of the loss against all tensors involved
# in its computation that have requires_grad = True
L.backward()
# Compare X.grad with dL_dX
# rtol: relative tolerance
# atol: absolute tolerance
X_grad_clone = X.grad.clone()
are_equal = np.allclose(X_grad_clone.numpy(), dL_dX.detach().numpy(), rtol=1e-5, atol=1e-8)
# Assert are_equal == True
if are_equal:
    print("Our calculation of DL_DX matches that of Pytorch")

# Let's calculate DL_dW
dL_dW = dL_dY @ X.T
# Compare against first principles. For brevity, we'll just tweak one element of W, instead of all elements
epsilon = 1e-4
W_ = W.clone()
W_.data[0,0] = W_.data[0,0] + epsilon
Y_ = W_ @ X + b
L_ = torch.sum(Y_)
# first element of dY_dX
dL_dW_00 = (L_ - L)/(W_[0,0] - W[0,0])
print(f"dL_dW[0,0] = {dL_dW_00: >4}")

# Now compare dL_dW with Pytorch's grad calculation
W_grad_clone = W.grad.clone()
are_equal = np.allclose(W_grad_clone.numpy(), dL_dW.detach().numpy(), rtol=1e-5, atol=1e-8)
# Assert are_equal == True
if are_equal:
    print("Our calculation of DL_DW matches that of Pytorch")

