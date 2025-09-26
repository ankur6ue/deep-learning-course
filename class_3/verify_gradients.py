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
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.softmax import SoftmaxModule
from utils.losses import CrossEntropyLossModule, MSELossModule, FusedSoftmaxMSELossModule
import torch.nn.functional as F

# Verify that our calculation of derivatives for x-entropy loss and softmax match pytorch's
torch.manual_seed(42)
m = 60
n = 4
B = 3
X = torch.randn(m, B, requires_grad=True)
# Create random one hot vectors of size m, B as ground truth
random_indices = torch.randint(0, m, (B,))
one_hot_vectors = F.one_hot(random_indices, num_classes=m).T
Y = one_hot_vectors.float()
logits = X

# Compare output of our softmax against pytorch's
# our softmax calculation
softmax = SoftmaxModule()
probs = softmax(logits)
# Pytorch's. Summing along dim=0 sum's along columns
probs_pt = F.softmax(logits, dim=0)
are_equal = np.allclose(probs_pt.detach().numpy(), probs.detach().numpy(), rtol=1e-5, atol=1e-8)
# Assert are_equal == True
if are_equal:
    print("Our calculation of softmax matches that of Pytorch")

mse = MSELossModule()
loss = mse(probs, Y)
loss.backward()
X_grad_clone = X.grad.clone()
# Now use pytorch calculation
X.grad.zero_()
loss_pt = F.mse_loss(F.softmax(logits, dim=0), Y)
loss_pt.backward()
X_grad_clone_pt = X.grad.clone()
are_equal = np.allclose(X_grad_clone_pt.detach().numpy(), X_grad_clone.detach().numpy(), rtol=1e-5, atol=1e-8)
# Assert are_equal == True
if are_equal:
    print("Our calculation of derivative of softmax + MSE loss matches that of Pytorch")
# Try our fusedxentropymse loss..
X.grad.zero_()
fused_xentropy_mse = FusedSoftmaxMSELossModule()
fused_xentropy_mse_loss = fused_xentropy_mse(logits, Y)
fused_xentropy_mse_loss.backward()
X_grad_clone = X.grad.clone()
are_equal = np.allclose(X_grad_clone_pt.detach().numpy(), X_grad_clone.detach().numpy(), rtol=1e-5, atol=1e-8)
# Assert are_equal == True
if are_equal:
    print("Our calculation of derivative of softmax + MSE loss using a fused kernel matches that of Pytorch")
X.grad.zero_()
x_entropy = CrossEntropyLossModule()
loss = x_entropy((logits), random_indices)
loss.backward()
X_grad_clone = X.grad.clone()
# Now use pytorch calculation
X.grad.zero_()
# Normally, pytorch's cross_entropy inputs are shaped (B, N), where B is batch dimension and N is number of classes.
# So we need to transpose the logits array
loss_pt = F.cross_entropy(torch.transpose(logits, 1, 0), random_indices)
# The above is equivalent to:
# loss_pt = F.nll_loss(F.log_softmax(logits, dim=0).T, random_indices)
loss_pt.backward()
X_grad_clone_pt = X.grad.clone()
are_equal = np.allclose(X_grad_clone_pt.detach().numpy(), X_grad_clone.detach().numpy(), rtol=1e-5, atol=1e-8)
if are_equal:
    print("Our calculation of derivative of x-entropy matches that of Pytorch")

