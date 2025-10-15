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

# For a function x->y->z, where y=Ax, and z=By calculates derivative of z wrt x using chain rule and verifies it using
# finite difference.
# Denominator-layout check for dz/dx = A^T B^T  (with y = A x, z = B y)
# Keeps math explicit; finite-diff uses central differences.
import numpy as np

rng = np.random.default_rng(0)

def fd_dz_dx_denom(A, B, x, h=1e-6):
    """
    Finite-difference approximation of denominator-layout dz/dx (shape n x k).
    Row i = ∂z/∂x_i (a 1 x k row vector).
    """
    n = x.size
    k = B.shape[0]
    FD = np.zeros((n, k))
    Ax = A @ x  # reuse where possible
    for i in range(n):
        e = np.zeros_like(x); e[i] = 1.0
        z_plus  = B @ (A @ (x + h*e))
        z_minus = B @ (A @ (x - h*e))
        FD[i, :] = (z_plus - z_minus) / (2*h)  # Row[i] of the finite difference matrix contains changes in each
        # dimension of z for changes in x[i]
    return FD

def test_once(n=5, m=7, k=4, h=1e-6):
    A = rng.normal(size=(m, n))
    B = rng.normal(size=(k, m))
    x = rng.normal(size=(n,))
    # Analytic (denominator layout): dz/dx = A^T B^T  (shape n x k)
    dzdx_analytic = A.T @ B.T
    # Finite-diff (denominator layout): n x k
    dzdx_fd = fd_dz_dx_denom(A, B, x, h=h)

    abs_err = np.max(np.abs(dzdx_analytic - dzdx_fd))
    rel_err = abs_err / (np.max(np.abs(dzdx_analytic)) + 1e-12)
    return abs_err, rel_err, dzdx_analytic, dzdx_fd

if __name__ == "__main__":
    for t in range(5):
        abs_err, rel_err, ana, fd = test_once(n=6, m=8, k=5, h=1e-6)
        print(f"trial {t}: max |abs err| = {abs_err:.3e}  |  rel err = {rel_err:.3e}")
