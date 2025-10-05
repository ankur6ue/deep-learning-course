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


# ------------------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------------------
def h(x):
    """
    Computes the function:
        h(x) = sin(f2) / log(g1)
    where:
        f1 = x^2
        g1 = 1/x
        f2 = f1 + g1
        f3 = sin(f2)
        g2 = log(g1)

    Returns:
        h, f1, g1, f2, f3, g2
    """
    f1 = x ** 2
    g1 = 1 / x

    f2 = f1 + g1
    f3 = np.sin(f2)
    g2 = np.log(g1)

    h = f3 / g2
    return h, f1, g1, f2, f3, g2


# ------------------------------------------------------------------------------
# Evaluate the forward pass at a specific value of x
# ------------------------------------------------------------------------------
x = 0.1
h_, f1, g1, f2, f3, g2 = h(x)


# ------------------------------------------------------------------------------
# Backward pass (manual differentiation via chain rule)
# ------------------------------------------------------------------------------
# Derivatives flowing backward through the computation graph

# ∂h/∂f3 and ∂h/∂g2 from h = f3 / g2
dh_df3 = 1 / g2
dh_dg2 = -f3 / (g2 ** 2)

# ∂g2/∂g1 from g2 = log(g1)
dg2_dg1 = 1 / g1

# ∂f3/∂f2 from f3 = sin(f2)
df3_df2 = np.cos(f2)

# ∂h/∂f2 (propagate through f3)
dh_df2 = dh_df3 * df3_df2

# f2 = f1 + g1 → ∂f2/∂f1 = 1, ∂f2/∂g1 = 1
df2_df1 = 1
df2_dg1 = 1

# Chain rule applications
dh_df1 = dh_df2 * df2_df1
dh_dg1 = dh_df2 * df2_dg1 + dh_dg2 * dg2_dg1

# f1 = x^2 → ∂f1/∂x = 2x
df1_dx = 2 * x

# g1 = 1/x → ∂g1/∂x = -1/x^2
dg1_dx = -1 / (x ** 2)

# Combine contributions: ∂h/∂x = ∂h/∂f1 * ∂f1/∂x + ∂h/∂g1 * ∂g1/∂x
dh_dx = dh_df1 * df1_dx + dh_dg1 * dg1_dx


# ------------------------------------------------------------------------------
# Numerical gradient check (finite differences)
# ------------------------------------------------------------------------------
epsilon = 1e-5
dh_dx_numerical = (h(x + epsilon)[0] - h(x)[0]) / epsilon

# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------
print("Analytical dh/dx:", dh_dx)
print("Numerical dh/dx :", dh_dx_numerical)
print("Difference      :", abs(dh_dx - dh_dx_numerical))
