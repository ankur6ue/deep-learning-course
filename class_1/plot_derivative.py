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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# This code defines a function with one minimum and its derivatives
# and plots both. Intent is to show that the minimum coincides with the point
# where the derivative is zero

# Define the function (a quadratic function for simplicity, which has a clear minimum)
def f(x):
    return x**2 + 2*x + 1

# Define the derivative of the function
def df_dx(x):
    # For f(x) = x^2 + 2x + 1, the derivative is 2x + 2
    return 2*x + 2

# Generate x values
x = np.linspace(-5, 3, 400)

# Calculate y values for the function
y = f(x)

# Calculate y values for the derivative
y_prime = df_dx(x)

# Find the minimum of the function
# For a quadratic function ax^2 + bx + c, the minimum is at x = -b / (2a)
x_min = -2 / (2 * 1)
y_min = f(x_min)

# Plot the function and its derivative
plt.figure(figsize=(10, 6))

plt.plot(x, y, label='f(x) = x^2 + 2x + 1', color='blue')
plt.plot(x, y_prime, label="f'(x) = 2x + 2", color='red', linestyle='--')

# Mark the minimum point on the function
plt.plot(x_min, y_min, 'go', markersize=8, label=f'Minimum at ({x_min:.2f}, {y_min:.2f})')

# Draw a horizontal line at y=0 for the derivative to show where it crosses zero
plt.axhline(0, color='gray', linestyle=':', linewidth=0.8)

# Add a vertical line at the x-coordinate of the minimum to show where the derivative is zero
plt.axvline(x_min, color='green', linestyle=':', linewidth=0.8, label='x at minimum')

plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()