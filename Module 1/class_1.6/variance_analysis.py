# Copyright 2025 Ankur Mohan
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import numpy as np
torch.manual_seed(42)
m = 5
n = 4
B = 1

# Each element of X, W, b is drawn from a mean=0, var=1 normal distribution
# Draw N samples and calculate Y. Save the Ys so we can calculate the mean and variance
N = 10000
y_hist = torch.empty(n, 0)
a_hist = torch.empty(n, 0)
for trials in range(0, N):
    x = torch.randn(m, B, requires_grad=False)
    W = torch.randn(n, m, requires_grad=False)
    b = torch.zeros(n, 1)

    # y is n*1
    # In Pytorch, @ means matrix multiplication, * means hadamard product
    y = W @ x + b
    y_hist = torch.cat([y_hist, y], dim=1)
    a = y.clamp(min=0)
    a_hist = torch.cat([a_hist, a], dim=1)

mean_y = y_hist.mean(dim=1)
var_y = y_hist.var(dim=1)
mean_a = a_hist.mean(dim=1)
var_a = a_hist.var(dim=1)
# From lecture slides, we expect mean_a = sqrt(var(y)/2pi)
print(f'mean of first element of activation array: {mean_a[0]}')
print(f'mean of first element of activation array from math: {np.sqrt(var_y[0]/(2*np.pi))}')
# From lecture slides, we expect var_a approx m*var(x)*var(W)/2*(1-1/2pi)
print(f'variance of first element of activation array: {var_a[0]}')
print(f'variance of first element of activation array from math: {m * 1 * 1/2 * (1 - 1/(2*np.pi))}')
print('done')
# verify mean (n*1 vector) is close to 0
# verify variance (n*1 vector) is close to m + 1 (1 for the bias term)
