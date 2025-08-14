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

# This code defines a function with multiple local minima and its derivatives
# and plots both. Intent is to show that the local minimas coincide with points
# where the derivative is zero

def f(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.1 * x**2

def df_dx(x):
    # For f(x) = x^2 + 2x + 1, the derivative is 2x + 2
    return np.cos(x) - np.sin(2*x) + 0.2*x

# Generate x-values for the plot
x = np.linspace(-5, 5, 500)  # Adjust the range and number of points as needed

# Calculate corresponding y-values
y = f(x)
# calculate the derivative
y_prime = df_dx(x)
# Plot the function and the derivative
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x)=sin(x) + 0.5cos(2x) + 0.1x^2')
plt.plot(x, y_prime, label="f'(x) = cos(x) - sin(2x) + 0.2x", color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function with Multiple Local Minima')
plt.grid(True)
plt.legend()
plt.show()

