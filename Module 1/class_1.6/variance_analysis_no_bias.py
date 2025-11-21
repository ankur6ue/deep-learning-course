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
    # y is n*1
    # In Pytorch, @ means matrix multiplication, * means hadamard product
    y = W @ x
    y_hist = torch.cat([y_hist, y], dim=1)


mean_y = y_hist.mean(dim=1)
var_y = y_hist.var(dim=1)

# From lecture slides, we expect mean_y = 0
print(f'mean of first element of y: {mean_y[0]}')
# From lecture slides, we expect var_y = m*var_x*var_W = m
print(f'variance of first element of activation array: {var_y[0]}')

print('done')
