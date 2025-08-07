import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

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