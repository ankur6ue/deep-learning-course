import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    # Approximate GELU using tanh-based formulation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

def swish(x):
    return x / (1 + np.exp(-x))

def swiglu(x):
    # SwiGLU = Swish(x1) * x2, here approximated by setting x1 = x2 = x
    return swish(x) * x

# Generate input values
x = np.linspace(-5, 5, 400)

# Compute activations
y_relu = relu(x)
y_tanh = tanh(x)
y_leaky = leaky_relu(x)
y_gelu = gelu(x)
y_swish = swish(x)
y_swiglu = swiglu(x)

# Plot all activations
plt.figure(figsize=(10, 6))
plt.plot(x, y_relu, label="ReLU")
plt.plot(x, y_tanh, label="tanh")
plt.plot(x, y_leaky, label="Leaky ReLU (α=0.1)")
plt.plot(x, y_gelu, label="GELU")
plt.plot(x, y_swish, label="Swish")
# plt.plot(x, y_swiglu, label="SwiGLU (approx)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("Activation Functions: ReLU, tanh, Leaky ReLU, GELU, Swish")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()