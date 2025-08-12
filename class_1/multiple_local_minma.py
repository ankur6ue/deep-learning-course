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

