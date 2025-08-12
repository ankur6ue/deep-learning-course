import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Objective function: quadratic + oscillations
def f(x):
    return (x - 2)**2 + np.sin(5 * x) * 1.0

def grad_f(x):
    return 2*(x - 2) + 5 * np.cos(5 * x)

# Gradient descent
def gradient_descent(lr, decay=False, max_iters=50):
    x = -6.0
    history_x = [x]
    history_f = [f(x)]
    for i in range(max_iters):
        g = grad_f(x)
        x -= lr * g
        if decay:
            lr *= 0.85
        history_x.append(x)
        history_f.append(f(x))
    return np.array(history_x), np.array(history_f)

# Histories
hx_fixed, hf_fixed = gradient_descent(lr=0.2, decay=False)
hx_decay, hf_decay = gradient_descent(lr=0.2, decay=True)
steps = len(hf_fixed)

# Prepare plot
fig, ax = plt.subplots(figsize=(6, 4))
X = np.linspace(-6, 6, 400)
Y = f(X)
ax.plot(X, Y, 'k-', label='$f(x) = (x-2)^2+1$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Gradient Descent With and Without Learning Rate Decay')
ax.legend()

# Points for animation
point_fixed, = ax.plot([], [], 'ro', label='Fixed LR', markersize=4)
point_decay, = ax.plot([], [], 'go', label='Decaying LR', markersize=8)
ax.legend()

# Init function
def init():
    point_fixed.set_data([], [])
    point_decay.set_data([], [])
    return point_fixed, point_decay

# Animation function
def update(frame):
    point_decay.set_data([hx_decay[frame]], [hf_decay[frame]])
    point_fixed.set_data([hx_fixed[frame]], [hf_fixed[frame]])
    return point_fixed, point_decay

ani = FuncAnimation(fig, update, frames=len(hx_fixed), init_func=init,
                    blit=True, repeat=False, interval=500)

# Save as video
ani.save("gradient_descent_fx_vs_x.mp4", fps=2)

plt.show()
