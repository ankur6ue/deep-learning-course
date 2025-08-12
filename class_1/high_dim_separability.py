import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# This program shows how functions that are not separable in 2D can become separable by
# projecting them to higher dimensions

# 1. Create non-linearly separable 2D data (a circle)
np.random.seed(0)
n = 100
theta = 2 * np.pi * np.random.rand(n)
r_inner = 1
r_outer = 2

# Inner circle
x_inner = np.c_[r_inner * np.cos(theta[:n//2]) + 0.1*np.random.randn(n//2),
                r_inner * np.sin(theta[:n//2]) + 0.1*np.random.randn(n//2)]
y_inner = np.zeros(n//2)

# Outer circle
x_outer = np.c_[r_outer * np.cos(theta[n//2:]) + 0.1*np.random.randn(n//2),
                r_outer * np.sin(theta[n//2:]) + 0.1*np.random.randn(n//2)]
y_outer = np.ones(n//2)

X = np.vstack([x_inner, x_outer])
y = np.hstack([y_inner, y_outer])

# Plot original 2D data
plt.figure(figsize=(6,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title('Non-linearly separable in 2D')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.axis('equal')
plt.show()

# 2. Map to higher dimensional space: φ(x1, x2) = (x1, x2, x1^2 + x2^2)
def phi(x):
    x1, x2 = x[:, 0], x[:, 1]
    return np.c_[x1, x2, x1**2 + x2**2]

X_3D = phi(X)

# Plot in 3D to show linear separability
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], c=y, cmap='bwr', edgecolor='k')
ax.set_title('Linearly separable in 3D feature space')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x1² + x2²')
plt.tight_layout()
plt.show()