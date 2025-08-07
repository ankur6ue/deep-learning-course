import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function that maps 2D points (x, y) to 3D points (x, y, z)
def map_2d_to_3d(x, y):
    # Example function: a paraboloid
    z = np.sin(np.sqrt(x**2 + y**2))
    return z

# Create a grid of 2D input points
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate the corresponding Z values using the function
Z = map_2d_to_3d(X, Y)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis') # 'viridis' is a colormap

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Set a title for the plot
ax.set_title('3D Plot of a Function Mapping 2D to 3D')

# Show the plot
plt.show()