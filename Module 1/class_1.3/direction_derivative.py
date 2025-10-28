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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function and gradient
def f(x, y):
    return x**2 + y**2

def grad(x, y):
    return np.array([2*x, 2*y])

# Point of interest and gradient there
x0, y0 = 2.0, 2.0
g = grad(x0, y0)               # (4, 4)
g_norm = np.linalg.norm(g)     # theoretical max directional derivative

# Sample unit directions uniformly on the circle
angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
dirs = np.column_stack([np.cos(angles), np.sin(angles)])   # shape (100, 2)

# Directional derivative in each direction: g · u
dir_deriv = dirs @ g

# Use magnitude for arrow length to avoid flipping the direction
lengths = np.abs(dir_deriv)

# Normalize lengths for a nice plot scale
if lengths.max() == 0:
    lengths_scaled = lengths
else:
    lengths_scaled = 1.2 * lengths / lengths.max()   # arrows up to length ~1.2

# Split by sign for coloring
pos = dir_deriv >= 0
neg = ~pos

# Prepare arrows
U_pos = dirs[pos, 0] * lengths_scaled[pos]
V_pos = dirs[pos, 1] * lengths_scaled[pos]
U_neg = dirs[neg, 0] * lengths_scaled[neg]
V_neg = dirs[neg, 1] * lengths_scaled[neg]

plt.figure(figsize=(8, 8))

# Draw arrows from the origin with quiver (keeps direction = u)
plt.quiver(np.zeros_like(U_pos), np.zeros_like(U_pos), U_pos, V_pos,
           angles='xy', scale_units='xy', scale=1, color='tab:blue', alpha=0.85)
plt.quiver(np.zeros_like(U_neg), np.zeros_like(U_neg), U_neg, V_neg,
           angles='xy', scale_units='xy', scale=1, color='tab:red', alpha=0.85)

# Highlight the gradient direction (steepest ascent)
g_unit = g / g_norm
plt.arrow(0, 0, 1.4*g_unit[0], 1.4*g_unit[1],
          head_width=0.10, head_length=0.12, length_includes_head=True,
          color='green', linewidth=3)

# Optional: also show steepest descent
plt.arrow(0, 0, -1.0*g_unit[0], -1.0*g_unit[1],
          head_width=0.08, head_length=0.10, length_includes_head=True,
          color='black', linewidth=2, alpha=0.6)

# Formatting
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1.6, 1.6); plt.ylim(-1.6, 1.6)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title("Directional derivatives at (2,2) for f(x,y)=x^2+y^2\n"
          "Arrow length = |g·u|, color = sign (blue:+, red:−)")

plt.xlabel("x-direction"); plt.ylabel("y-direction")

# Legend proxies
legend_elems = [
    Line2D([0], [0], color='tab:blue', lw=3, label='Increase (g·u > 0)'),
    Line2D([0], [0], color='tab:red', lw=3, label='Decrease (g·u < 0)'),
    Line2D([0], [0], color='green', lw=3, label='Gradient direction (max)'),
    Line2D([0], [0], color='black', lw=2, label='Steepest descent'),
]
plt.legend(handles=legend_elems, loc='upper right')

plt.show()

print("Gradient at (2,2):", g)
print("Max directional derivative (should be ||g||):", dir_deriv.max(), "≈", g_norm)
