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
# The below is ugly and not recommended, but lets us run the code as a file (python -m simple_nn1_mlflow.py) instead
# of converting it into a package..
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.losses import MSELossModule
from utils.optimizer import AdamOptimizer, SimpleOptimizer, AdamWOptimizer
from utils.lr_scheduler import LinearLR, ConstantLR, CosineLR
from utils.simple_nn_sinx import SimpleNN, create_dataset, draw_movie_frame
import matplotlib.pyplot as plt
import argparse
import matplotlib

# Set the seed for reproducibility
torch.manual_seed(42)
# Here we make the following changes on top of simple_nn1.py:
# 1. Define our own Pytorch modules, subclassing pytorch nn.module classes.
#    Implement the forward/backward pass manually
# 2. Sample a mini-batch from the training dataset without replacement
# 3. Try different optimizers and learning rate schedules


if __name__ == "__main__":
    # This makes plots show up as a separate figure
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='Train a simple neural network to model a sine function')
    parser.add_argument('--N', type=int, default=600, help='Number of points in the sine wave')
    parser.add_argument('--H', type=int, default=150, help='Size of hidden layer')
    parser.add_argument('--capture_frames', action='store_true', help='If set, every other frame is'
                                                                      'captured and saved to a frames directory')
    parser.add_argument('--optimizer', choices=['simple', 'adam'], default='simple')
    parser.add_argument('--lr_scheduler', choices=['linear', 'cosine', 'constant'], default='linear')
    parser.add_argument('--lr', type=float, default=0.07, help='learning rate (default: 0.07)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=300, help='batch size (should divide N)')

    args = parser.parse_args()
    N = args.N # Number of points in the entire dataset
    H = args.H # Size of the hidden layer
    B = args.batch_size # size of each batch used to average gradients
    lr = args.lr
    X, Y = create_dataset(N)
    # lets visualize the data:
    X = X.T # 1 * B
    Y = Y.T # 1 * B
    X.requires_grad = True
    # create the model
    model = SimpleNN(1, H, 1, 0.07)
    criterion = MSELossModule()
    if args.optimizer == "adam":
        optimizer = AdamOptimizer(model.layers, learning_rate=lr)
    else:
        optimizer = SimpleOptimizer(model.layers, learning_rate=lr)

    num_iter_per_epoch = (int)(N / args.batch_size)
    num_steps = (int)(num_iter_per_epoch * args.epochs)
    if args.lr_scheduler == "cosine":
        lr_scheduler = CosineLR(optimizer, num_steps, lr, 0.001)
    if args.lr_scheduler == "linear":
        lr_scheduler = LinearLR(optimizer, num_steps, lr, 0.001)
    if args.lr_scheduler == "constant":
        lr_scheduler = ConstantLR(optimizer, num_steps, lr, 0.001)
    loss_func_history = []
    c = 0 # global iteration count
    for e in range(args.epochs):
        # Each epoch uses a different permutation of indices
        indices = torch.randperm(N)
        for iter in range(num_iter_per_epoch):
            batch_indices = indices[iter*B: (iter+1)*B]
            X_ = X[:, batch_indices]
            Y_ = Y[:, batch_indices]
            o = model(X_)    # Forward pass
            loss = criterion(o, Y_)
            if args.capture_frames:
                if c % 2 == 0:
                    draw_movie_frame(model, c/2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step(c)
            loss_func_history.append(loss.item()) # .item extracts the value from a single element tensor,
            # converting it into a python number
            c = c + 1
        # print loss after every epoch
        print(f"After Epoch {e}, Loss  = {loss.item(): .4f}")


    # generate test data
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 300)
    X = torch.tensor(x_test.unsqueeze(1), dtype=torch.float32).unsqueeze(1)
    y_pred = model(X.T).detach().squeeze().numpy()
    plt.plot(x_test, y_pred, color='red', label="NN Approximation")
    plt.plot(x_test, np.sin(x_test), color='green', linestyle='--', label="True sin(x)")
    plt.title("Neural Network Approximating sin(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(range(0,c), loss_func_history, color='red', label="Loss function progression")
    plt.title("Progression of the loss function")
    plt.legend()
    plt.show()
    print('done')
