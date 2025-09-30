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
import matplotlib.pyplot as plt

# Define function and derivative
def f(x):
    return (x - 3)**2

def f_prime(x):
    return 2*(x - 3)

# Range for plotting
x = np.linspace(-1, 7, 400)
y = f(x)

# Pick some starting points
points = [0.5, 2, 4.5, 6]

plt.figure(figsize=(8,6))
plt.plot(x, y, label=r'$f(x) = (x-3)^2$')
plt.axvline(3, color='k', linestyle='--', label='Global minimum at x=3')

# Show derivative direction at selected points
for p in points:
    slope = f_prime(p)
    direction = -np.sign(slope)  # opposite of slope is direction to move
    plt.scatter(p, f(p), color='red')
    plt.annotate(f"f'={slope:.1f}\nmove {'→' if direction>0 else '←'}",
                 (p, f(p)),
                 textcoords="offset points", xytext=(0,15), ha='center',
                 color='blue',  fontsize=12, fontweight='bold')

plt.title("Derivative sign points toward the global minimum")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
