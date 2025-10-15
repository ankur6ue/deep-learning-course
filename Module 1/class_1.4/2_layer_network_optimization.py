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

rng = np.random.default_rng(1)

# Dimensions (can reuse the earlier ones)
m, n, k = 5, 4, 3
x = rng.normal(size=(m,)) # NumPy treats it as a row or column depending on context..
A = rng.normal(scale=0.5, size=(n, m))
B = rng.normal(scale=0.5, size=(k, n))

def forward(A, B, x):
    y = A @ x          # (n,)
    u = B @ y          # (k,)
    L = 0.5 * np.dot(u, u) # In the slides we are referring to L as v
    return L, y, u

lr_A = 0.1
lr_B = 0.1
iters = 200

for t in range(iters):
    L, y, u = forward(A, B, x)
    # Gradients
    dL_du = u
    dL_dB = np.outer(dL_du, y)         # (k,n)
    dL_dy = B.T @ u
    dL_dA = np.outer(dL_dy, x)     # (n,m), also note lack of transpose.. that's because of how we defined x

    # Update
    B -= lr_B * dL_dB
    A -= lr_A * dL_dA

    if (t+1) % 2 == 0 or t == 0:
        print(f"iter {t+1:3d}: L={L:.6f}, ||u||={np.linalg.norm(u):.6f}")

# Final check: u should be near zero
L_final, y_final, u_final = forward(A, B, x)
print("\nAfter optimization:")
print("  L_final =", L_final)
print("  ||u_final|| =", np.linalg.norm(u_final))
