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

# ---- Function and derivatives ----
def f(x):
    return np.sin(x) + 0.5*np.cos(2*x) + 0.1*x**2

def fprime(x):
    # d/dx [sin x] = cos x
    # d/dx [0.5 cos(2x)] = 0.5 * (-2 sin(2x)) = -sin(2x)
    # d/dx [0.1 x^2] = 0.2 x
    return np.cos(x) - np.sin(2*x) + 0.2*x

def fsecond(x):
    # d/dx f'(x) = -sin x - 2 cos(2x) + 0.2
    return -np.sin(x) - 2*np.cos(2*x) + 0.2

# ---- Gradient Descent ----
def gradient_descent(x0, lr=0.05, max_steps=5000, tol=1e-8):
    x = x0
    traj = [x]
    for t in range(max_steps):
        g = fprime(x)
        x_new = x - lr*g
        traj.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return np.array(traj)

# ---- Run experiments from multiple inits ----
starts = [-8.0, -4.0, -2.0, 0.5, 2.5, 5.0, 8.0]
lr = 0.05

# Compute trajectories
trajectories = {x0: gradient_descent(x0, lr=lr) for x0 in starts}

# ---- Plot function and trajectories ----
xmin, xmax = -10, 10
xs = np.linspace(xmin, xmax, 2000)
ys = f(xs)

plt.figure(figsize=(10,6))
plt.plot(xs, ys, linewidth=2, label='f(x)')


# Plot descent paths
for x0, traj in trajectories.items():
    plt.plot(traj, f(traj), linestyle='--', marker='o', markersize=3, linewidth=1,
             label=f'start {x0:g} → {traj[-1]:.3f}')
    # highlight start & end
    plt.scatter([traj[0]], [f(traj[0])], s=50)
    plt.scatter([traj[-1]], [f(traj[-1])], s=50)

plt.title(r"Gradient Descent on $f(x)=\sin x + 0.5\cos(2x) + 0.1x^2$: Different Starts → Different Local Minima")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc='best', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Print a concise report ----
print("Learning rate:", lr)
for x0, traj in trajectories.items():
    x_end = traj[-1]
    print(f"Start {x0:>6.2f} → Converged to x ≈ {x_end:.6f}, f(x) ≈ {f(x_end):.6f}, steps = {len(traj)-1}")
