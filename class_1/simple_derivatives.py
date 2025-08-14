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

import numpy as np # linear algebra
import matplotlib.pyplot as plt

# This program calculates the derivative of a function using first principles and compares it against the exact value of the derivative
# for different values of \delta. As delta increases, the calculated value of the derivative becomes less accurage
x = 1
n = 2
def f(x, n):
    return x**n

def df_dx(x, n):
    return n*x^(n-1)

# f(x) = x^n
# f'(x) = nx^(n-1)
dfdx = df_dx(x, n)
df_manual_list = []
for delta_x in np.arange(0.0001, 1, 0.05):
    df_dx_manual = (f(x + delta_x, n) - f(x, n))/delta_x
    df_manual_list.append(df_dx_manual)
    df_manual_list.append(df_dx_manual)
    print(f"delta_x: {delta_x: .2f}, derivative: {df_dx_manual: .2f}")


