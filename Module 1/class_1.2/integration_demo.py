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

def f(x):
    return x**2

def integral_f(x):
    return 1/3*x**3

# 1) Riemann rectangles
a, b = 0, 2
x = np.linspace(a, b, 600)
y = f(x)

n_rects = 8
xs = np.linspace(a, b, n_rects + 1)
rect_x = xs[:-1]
rect_width = (b - a) / n_rects
rect_heights = f(rect_x)

plt.figure(figsize=(8,6))
plt.plot(x, y, label='f(x)=x^2')
plt.fill_between(x, 0, y, alpha=0.3, label='Area under f(x)')
for xi, hi in zip(rect_x, rect_heights):
    plt.bar(xi, hi, width=rect_width, align='edge', alpha=0.4, edgecolor='black')
plt.title('Integration as Area Under a Curve')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# 2) Convergence plot
x_true = np.linspace(a, b, 2000)
y_true = f(x_true)
integral_np = np.trapz(y_true, x_true)
integral_true = integral_f(b) - integral_f(a)

rect_counts = np.arange(2, 101)
errors = []
for n_rects in rect_counts:
    xs = np.linspace(a, b, n_rects + 1)
    rect_x = xs[:-1]
    rect_width = (b - a) / n_rects
    rect_heights = f(rect_x)
    integral_approx = np.sum(rect_heights * rect_width)
    errors.append(abs(integral_true - integral_approx))

plt.figure(figsize=(8,6))
plt.plot(rect_counts, errors, marker='o')
plt.title('Convergence of Riemann Sum Approximation')
plt.xlabel('Number of rectangles')
plt.ylabel('Absolute error')
plt.yscale('log')
plt.grid(True, which='both')
plt.show()
