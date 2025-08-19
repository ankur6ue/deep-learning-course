import torch
import numpy as np


class SimpleOptimizer:
    def __init__(self, layers, learning_rate=0.01, reg=0.001):
        self.layers = layers
        self.lr = learning_rate
        self.reg = reg

    def step(self):
        for l in self.layers:
            l.weight.data -= self.lr * (l.weight.grad + self.reg * l.weight.data)
            l.bias.data -= self.lr * l.bias.grad

    def zero_grad(self):
        for l in self.layers:
            # Zero the gradients, otherwise they'll accumulate
            l.weight.grad.zero_()
            l.bias.grad.zero_()


class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Dictionary to store first moment estimates (m) for each parameter
        self.v = {}  # Dictionary to store second moment estimates (v) for each parameter
        self.t = 0  # Timestep counter

    def step(self):
        """
        Performs a single optimization step.
        """
        self.t += 1  # Increment timestep
        for l in self.layers:
            layer_name = l.name
            # weights and biases

            param_name = layer_name + "_w"
            param_value = l.weight.grad
            if param_name not in self.m:
                self.m[param_name] = torch.zeros_like(param_value)
                self.v[param_name] = torch.zeros_like(param_value)
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * l.weight.grad
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (l.weight.grad ** 2)
            # Bias correction for first moment
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            # Bias correction for second moment
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            # Update parameters
            l.weight.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # biases
            param_name = layer_name + "_b"
            param_value = l.bias.grad
            if param_name not in self.m:
                self.m[param_name] = torch.zeros_like(param_value)
                self.v[param_name] = torch.zeros_like(param_value)
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * l.bias.grad
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (l.bias.grad ** 2)
            # Bias correction for first moment
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            # Bias correction for second moment
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            # Update parameters
            l.bias.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for l in self.layers:
            # Zero the gradients, otherwise they'll accumulate
            l.weight.grad.zero_()
            l.bias.grad.zero_()