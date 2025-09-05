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

from torch import nn
import numpy as np
import torch
import sys
import os
# The below is ugly and not recommended, but lets us run the code as a file (python -m simple_nn1.py) instead
# of converting it into a package..
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.optimizer import AdamOptimizer, SimpleOptimizer
from utils.lr_scheduler import LinearLR, ConstantLR, CosineLR
from utils.linear import LinearModule
import matplotlib.pyplot as plt

initial_lr = 0.1
final_lr = 0.001
epochs = 100

# Dummy net, not used in this example

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = LinearModule(input_size, hidden_size, "fc1")
        self.layers = []
        self.layers.append(self.fc1)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = SimpleNN(input_size=784, hidden_size=100, output_size=10)
opt = SimpleOptimizer(model.layers, initial_lr, epochs)
lr_scheduler_cosine = CosineLR(opt, epochs, initial_lr, final_lr)
lr_scheduler_linear = LinearLR(opt, epochs, initial_lr, final_lr)
lr_history_cosine = []
lr_history_linear = []

for epoch in range(epochs):
    lr_scheduler_linear.step(epoch)
    lr = lr_scheduler_linear.get_lr()
    lr_history_linear.append(lr)

    lr_scheduler_cosine.step(epoch)
    lr = lr_scheduler_cosine.get_lr()
    lr_history_cosine.append(lr)

x = range(epochs)
plt.plot(x, lr_history_cosine, label='cosine annealing', color='blue')
plt.plot(x, lr_history_linear, label="linear annealing", color='red')
plt.title('Linear and Cosine Learning Rate Schedules')
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.legend()
plt.show()