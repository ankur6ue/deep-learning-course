"""
Gradient Descent with Learning Rate Decay Visualization

This script demonstrates the effect of learning rate decay in gradient descent optimization.
It compares two gradient descent paths: one with a constant learning rate and another with a decaying learning rate.
The visualization shows how learning rate decay can help in better convergence to the minimum.

Author: Ankur Mohan
License: MIT License (see below for full license text)
"""

# Copyright 2025 Ankur Mohan
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Gradient Descent with and without Learning Rate Decay

This script demonstrates how gradient descent performs with and without learning rate decay
on a function with local oscillations. The key difference between the two approaches:
- Fixed learning rate: May oscillate around the minimum without converging precisely
- Decaying learning rate: Gradually reduces step size, allowing for more precise convergence
"""

def f(x):
    """
    Objective function: A quadratic function with added sinusoidal oscillations
    
    The function has a global minimum at x ≈ 2 but with local oscillations that create
    multiple local minima and maxima, making optimization more challenging.
    
    Args:
        x: Input value or array
        
    Returns:
        float or ndarray: Function value at x
    """
    return (x - 2)**2 + np.sin(5 * x) * 1.0

def grad_f(x):
    """
    Gradient of the objective function f(x)
    
    This is the analytical derivative of f(x), used to compute the gradient at any point x.
    
    Args:
        x: Input value or array
        
    Returns:
        float or ndarray: Gradient at point x
    """
    return 2*(x - 2) + 5 * np.cos(5 * x)

def gradient_descent(lr, decay=False, max_iters=50):
    """
    Perform gradient descent optimization
    
    Args:
        lr: Initial learning rate
        decay: If True, apply learning rate decay (multiplicative decay of 0.85 per step)
        max_iters: Maximum number of iterations
        
    Returns:
        tuple: (history of x values, history of function values)
    """
    x = -6.0  # Initial position
    history_x = [x]  # Store all x positions during optimization
    history_f = [f(x)]  # Store all function values during optimization
    
    for i in range(max_iters):
        # Compute gradient at current position
        g = grad_f(x)
        
        # Update position using gradient descent
        x -= lr * g
        
        # Apply learning rate decay if enabled
        if decay:
            lr *= 0.85  # Reduce learning rate by 15% each step
            
        # Record history
        history_x.append(x)
        history_f.append(f(x))
        
    return np.array(history_x), np.array(history_f)

# Run gradient descent with both fixed and decaying learning rates
hx_fixed, hf_fixed = gradient_descent(lr=0.2, decay=False)  # Fixed learning rate
hx_decay, hf_decay = gradient_descent(lr=0.2, decay=True)   # Decaying learning rate

# Set up the plot for visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Generate points for plotting the function
X = np.linspace(-6, 6, 400)
Y = f(X)

# Plot the objective function
ax.plot(X, Y, 'k-', label='$f(x) = (x-2)^2 + \\sin(5x)$', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('Gradient Descent: Fixed vs Decaying Learning Rate', fontsize=14)
ax.grid(True, alpha=0.3)

# Create point markers for the animation
point_fixed, = ax.plot([], [], 'ro', label='Fixed LR', markersize=8, alpha=0.7)
point_decay, = ax.plot([], [], 'go', label='Decaying LR', markersize=10, alpha=0.7)
ax.legend(fontsize=10)

# Initialize the animation
def init():
    """Initialize the animation with empty points"""
    point_fixed.set_data([], [])
    point_decay.set_data([], [])
    return point_fixed, point_decay

def update(frame):
    """Update the animation frame
    
    Args:
        frame: Current frame number
        
    Returns:
        tuple: Updated point objects for the animation
    """
    # Update the position of both points
    point_decay.set_data([hx_decay[frame]], [hf_decay[frame]])
    point_fixed.set_data([hx_fixed[frame]], [hf_fixed[frame]])
    
    # Add a trail effect by plotting past positions with decreasing opacity
    if frame > 0:
        ax.plot(hx_decay[max(0, frame-5):frame+1], 
                hf_decay[max(0, frame-5):frame+1], 
                'g-', alpha=0.3, linewidth=1)
        ax.plot(hx_fixed[max(0, frame-5):frame+1], 
                hf_fixed[max(0, frame-5):frame+1], 
                'r-', alpha=0.3, linewidth=1)
    
    return point_fixed, point_decay

# Create the animation
ani = FuncAnimation(
    fig, 
    update, 
    frames=len(hx_fixed), 
    init_func=init,
    blit=False,  # Set to False to allow for trail effect
    repeat=False,
    interval=300  # Delay between frames in milliseconds
)

# Save the animation as a video file
print("Saving animation to 'gradient_descent_fx_vs_x.mp4'...")
ani.save("gradient_descent_fx_vs_x.mp4", 
         writer='ffmpeg', 
         fps=5,  # Frames per second
         dpi=150,  # Dots per inch
         savefig_kwargs={'facecolor': 'white'})  # Ensure white background

# Display the plot
plt.tight_layout()
plt.show()

print("Animation complete! The video shows how the decaying learning rate"
      " helps the optimization converge to the true minimum.")
