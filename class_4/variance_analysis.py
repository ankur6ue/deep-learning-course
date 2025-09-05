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
N = 1000
Y_hist = torch.empty(n, 0)
for trials in range(0, N):
    X = torch.randn(m, B, requires_grad=False)
    W = torch.randn(n, m, requires_grad=False) * 1/np.sqrt(m)
    b = torch.randn(n, B, requires_grad=False) * 1/np.sqrt(m)

    # y is n*1
    # In Pytorch, @ means matrix multiplication, * means hadamard product
    Y = W @ X + b
    Y_hist = torch.cat([Y_hist, Y], dim=1)


mean = Y_hist.mean(dim=1)
var = Y_hist.var(dim=1)
print('done')
# verify mean (n*1 vector) is close to 0
# verify variance (n*1 vector) is close to m + 1 (1 for the bias term)
