import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Function and derivatives ---
def f(x, y):
    return x**2 + y**2

def fx(x, y):
    return 2*x

def fy(x, y):
    return 2*y

# --- Grid and point ---
x = np.linspace(-3, 3, 120)
y = np.linspace(-3, 3, 120)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

x0, y0 = 2.0, 2.0
z0 = f(x0, y0)
fx0, fy0 = fx(x0, y0), fy(y0, y0)  # fx0 = 4, fy0 = 4

# --- 3D surface with tangent arrows ---
fig = plt.figure(figsize=(20, 12))

ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.35, linewidth=0, antialiased=True)

# Mark the point
ax.scatter(x0, y0, z0, color='red', s=80, label='(2,2)')

# Tangent in +x direction at (x0,y0): (dx, dy, dz) = (1, 0, fx0*1)
ax.quiver(x0, y0, z0, 1, 0, fx0,
          color='blue', linewidth=3, length=0.9, arrow_length_ratio=0.2, label='∂f/∂x')

# Tangent in +y direction at (x0,y0): (0, 1, fy0*1)
ax.quiver(x0, y0, z0, 0, 1, fy0,
          color='orange', linewidth=3, length=0.9, arrow_length_ratio=0.2, label='∂f/∂y')

# Gradient direction on the surface:
# Move in xy by (fx0, fy0); surface tangent is (fx0, fy0, fx0*fx0 + fy0*fy0)
ax.quiver(x0, y0, z0, fx0, fy0, fx0*fx0 + fy0*fy0,
          color='green', linewidth=3, length=0.18, arrow_length_ratio=0.25, label='∇f')

# Make geometry easier to read
ax.set_box_aspect([1, 1, 0.5])
ax.view_init(elev=28, azim=-60)  # choose a clear viewpoint
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('f(x,y)', fontsize=12)
ax.set_title(r'$f(x,y)=x^2+y^2$ — tangents & gradient at (2,2)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)

# --- 2D contour with gradient vector (no 3D perspective) ---
ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contour(X, Y, Z, levels=15)
ax2.clabel(cs, inline=True, fontsize=8)
ax2.scatter([x0], [y0], color='red', s=60)

# Partial derivative arrows in the xy-plane (projected)
ax2.quiver(x0, y0, 1, 0, color='blue', scale=8, width=0.007)
ax2.quiver(x0, y0, 0, 1, color='orange', scale=8, width=0.007)

# Gradient arrow in the xy-plane (points radially outward)
ax2.quiver(x0, y0, fx0, fy0, color='green', scale=25, width=0.010)

ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Contour view with ∇f at (2,2)')

plt.tight_layout()
plt.show()

print("At (2,2): f =", z0, ", fx =", fx0, ", fy =", fy0)
