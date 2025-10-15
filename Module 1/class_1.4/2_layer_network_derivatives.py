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

import numpy as np

rng = np.random.default_rng(0)

# Dimensions
m, n, k = 5, 4, 3
x = rng.normal(size=(m,))
A = rng.normal(size=(n, m))
B = rng.normal(size=(k, n))

def forward_v(A, B, x):
    y = A @ x            # (n,)
    u = B @ y            # (k,)
    v = u.sum()          # scalar
    return v, y, u

# Analytical gradient wrt B for v = sum(u)
v, y, u = forward_v(A, B, x)
gradB_analytic = np.ones((k,1)) @ y[None, :]   # 1 * y^T  -> shape (k,n)

# Finite differences on B
eps = 1e-6
gradB_fd = np.zeros_like(B)

for i in range(k):
    for j in range(n):
        Bij_plus = B.copy()
        Bij_minus = B.copy()
        Bij_plus[i, j]  += eps
        Bij_minus[i, j] -= eps
        v_plus, _, _  = forward_v(A, Bij_plus, x)
        v_minus, _, _ = forward_v(A, Bij_minus, x)
        gradB_fd[i, j] = (v_plus - v_minus) / (2*eps)

# finite differences on A
gradA_fd = np.zeros_like(A)
dv_dy = B.T @ np.ones((k,1))
gradA_analytic = dv_dy @ x[None, :]   # dv_dy * x^T  -> shape (k,n)

for i in range(n):
    for j in range(m):
        Aij_plus = A.copy()
        Aij_minus = A.copy()
        Aij_plus[i, j]  += eps
        Aij_minus[i, j] -= eps
        v_plus, _, _  = forward_v(Aij_plus, B, x)
        v_minus, _, _ = forward_v(Aij_minus, B, x)
        gradA_fd[i, j] = (v_plus - v_minus) / (2*eps)

# Report agreement
max_abs_err = np.max(np.abs(gradB_fd - gradB_analytic))
rel_err = max_abs_err / (np.max(np.abs(gradB_analytic)) + 1e-12)
print("Finite-diff check for dv/dB:")
print("  max |fd - analytic| =", max_abs_err)
print("  relative error      =", rel_err)

max_aAs_err = np.max(np.abs(gradA_fd - gradA_analytic))
rel_err = max_aAs_err / (np.max(np.abs(gradA_analytic)) + 1e-12)
print("Finite-diff check for dv/dA:")
print("  max |fd - analytic| =", max_aAs_err)
print("  relative error      =", rel_err)
