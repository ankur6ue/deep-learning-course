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

from DL_Course.utils.tanh import TanhModule

# The below is ugly and not recommended, but lets us run the code as a file (python -m simple_nn1.py) instead
# of converting it into a package..
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.leakyrelu import LeakyReLUModule
from utils.tanh import TanhModule
from utils.gelu import GeluModule
from utils.swish import SwishModule
from utils.losses import MSELossModule
from utils.optimizer import AdamOptimizer, SimpleOptimizer
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib

# Set the seed for reproducibility
torch.manual_seed(42)

# We train a simple neural network with 1 hidden layer to learn to predict the value of a sine function.
# The user can specify the kind of non-linearity to use
# We'll use MLFlow to track training progress for every non-linearity

def create_dataset(N):
    np.random.seed(0)
    x_np = np.linspace(-2 * np.pi, 2 * np.pi, N)
    y_np = np.sin(x_np) + 0.1 * np.random.randn(*x_np.shape)  # noisy sin(x)

    X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y


def draw_movie_frame(net, frame_num):
    # create array of numbers size 100, from -1 to 1
    x = np.array([-1 + 0.02 * n for n in range(100)], dtype=np.float32)
    y = np.array([-1 + 0.02 * n for n in range(100)],  dtype=np.float32)

    # create an array with coordinates of pixels on a 100, 100 frame
    XX = np.empty((2, 0), dtype=np.float32)
    for x_ in x:
        for y_ in y:
            XX = np.hstack((XX, np.array([[x_, y_]]).T))

    # Ask the neural net to produce the probability distribution over classes for each pixel on this frame
    o = net.forward(torch.from_numpy(XX))
    # Get the predicted class and reshape into a 100, 100 frame
    predicted_class = np.argmax(o.detach().numpy(), axis=0).reshape(100, 100)
    # Show the frame and save to a file. We'll use ffmpeg to make a movie out of the frames
    plt.imshow(predicted_class)
    script_dir = os.path.dirname(__file__)
    plt.savefig(script_dir + "/frames" + "/file%04d.png" % frame_num)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = LinearModule(input_size, hidden_size, "fc1")
        self.fc2 = LinearModule(hidden_size, output_size, "fc2")
        # self.relu = ReLUModule()  # Activation function
        self.non_linear = LeakyReLUModule()
        self.non_linear = TanhModule()
        self.non_linear = GeluModule()
        self.non_linear = SwishModule()
        self.layers = []
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear(x)
        x = self.fc2(x)
        return x


def draw_movie_frame(model, frame_num):
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 300)
    X = torch.tensor(x_test.unsqueeze(1), dtype=torch.float32).unsqueeze(1)
    y_pred = model(X.T).squeeze().detach().numpy()
    plt.cla()
    # set ylim so matplotlib doesn't set it based on data. That causes jerkiness when the
    # images are converted into a video
    plt.ylim(-1.5, 1.5)
    plt.plot(x_test, y_pred, color='red', label="NN Approximation")
    plt.plot(x_test, np.sin(x_test), color='green', linestyle='--', label="True sin(x)")
    plt.title("Neural Network Approximating sin(x)")
    plt.legend()
    plt.grid(True)
    #   plt.show(block=False)
    script_dir = os.path.dirname(__file__)
    plt.savefig(script_dir + "/frames" + "/frame%04d.png" % frame_num)


if __name__ == "__main__":
    # This makes plots show up as a separate figure
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='Train a simple neural network to model a sine function')
    parser.add_argument('--N', type=int, default=200, help='Number of points in the sine wave')
    parser.add_argument('--H', type=int, default=100, help='Size of hidden layer')
    parser.add_argument('--capture_frames', action='store_true', help='If set, every other frame is'
                                                                      'captured and saved to a frames directory')
    parser.add_argument('--optimizer', choices=['simple', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size (should divide N)')

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
    model = SimpleNN(1, H, 1)
    criterion = MSELossModule()
    if args.optimizer == "adam":
        optimizer = AdamOptimizer(model.layers, learning_rate=lr)
    else:
        optimizer = SimpleOptimizer(model.layers, learning_rate=lr)

    c = 0 # global iteration count
    for e in range(args.epochs):
        # Each epoch uses a different permutation of indices
        indices = torch.randperm(N)
        num_iter_per_epoch = (int)(N/args.batch_size)
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
            c = c + 1
        # print loss after every epoch
        print(f"After Epoch {e}, Loss  = {loss.detach().squeeze().numpy(): .4f}")


    # generate test data
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 300)
    X = torch.tensor(x_test.unsqueeze(1), dtype=torch.float32).unsqueeze(1)
    y_pred = model(X.T).squeeze().detach().numpy()
    plt.plot(x_test, y_pred, color='red', label="NN Approximation")
    plt.plot(x_test, np.sin(x_test), color='green', linestyle='--', label="True sin(x)")
    plt.title("Neural Network Approximating sin(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
