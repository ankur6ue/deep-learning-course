# ------------------------------------------------------------------------------
# Copyright 2025 Ankur Mohan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------

import numpy as np

## Part 1: for h = sin(x*y)
def h(x, y):
    return np.sin(x*y)

def dh_dx(x, y):
    return y*np.cos(x*y)

def dh_dy(x, y):
    return x*np.cos(x*y)

x, y = 1.0, 0.5
eps = 1e-5
dh_dx_num = (h(x+eps, y)-h(x-eps, y))/(2*eps)
dh_dy_num = (h(x, y+eps)-h(x, y-eps))/(2*eps)

print("∂h/∂x analytical:", dh_dx(x,y), "numerical:", dh_dx_num)
print("∂h/∂y analytical:", dh_dy(x,y), "numerical:", dh_dy_num)

## Part 2: y = a^T x
def dot_forward(a: np.ndarray, x: np.ndarray) -> float:
    """Forward: y = a^T x"""
    return float(a @ x)

def grad_wrt_x(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Analytical: dy/dx = a"""
    return a.copy()

def grad_wrt_a(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Analytical: dy/da = x"""
    return x.copy()

def finite_diff_grad_wrt_x(a: np.ndarray, x: np.ndarray, eps=1e-6) -> np.ndarray:
    """Finite-difference estimate of dy/dx."""
    g = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy();  x_plus[i]  += eps
        g[i] = (dot_forward(a, x_plus) - dot_forward(a, x)) / (eps)
    return g

def finite_diff_grad_wrt_a(a: np.ndarray, x: np.ndarray, eps=1e-6) -> np.ndarray:
    """Finite-difference estimate of dy/da."""
    g = np.zeros_like(a)
    for i in range(a.size):
        a_plus = a.copy();  a_plus[i]  += eps
        g[i] = (dot_forward(a_plus, x) - dot_forward(a, x)) / (eps)
    return g

# ----- Experiment -----
rng = np.random.default_rng(0)
N = 128
trials = 200
err_x, err_a = [], []

for _ in range(trials):
    # Sample random vectors (zero-mean, unit variance)
    a = rng.normal(size=N)
    x = rng.normal(size=N)

    y = dot_forward(a, x)

    # Analytical
    gxa = grad_wrt_x(a, x)
    gaa = grad_wrt_a(a, x)

    # Numerical
    gxn = finite_diff_grad_wrt_x(a, x, eps=1e-6)
    gan = finite_diff_grad_wrt_a(a, x, eps=1e-6)

    err_x.append(np.max(np.abs(gxa - gxn)))
    err_a.append(np.max(np.abs(gaa - gan)))

print(f"[dy/dx]  max abs error over {trials} trials: {np.max(err_x):.3e}, mean: {np.mean(err_x):.3e}")
print(f"[dy/da]  max abs error over {trials} trials: {np.max(err_a):.3e}, mean: {np.mean(err_a):.3e}")
