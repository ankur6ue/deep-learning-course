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
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.losses import MSELossModule
from utils.optimizer import AdamOptimizer, SimpleOptimizer
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib
from torch.utils.data import Dataset, DataLoader, RandomSampler

# We train a simple neural network with 1 hidden layer to learn to predict the value of a sine function
# We use the pytorch dataloader and sampler instead of manual sampling as in simple_nn1.py

# Set the seed for reproducibility
torch.manual_seed(42)

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
        self.relu = ReLUModule()  # Activation function
        self.layers = []
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def zero_grad(self):
        for l in self.layers:
            # Zero the gradients, otherwise they'll accumulate
            l.weight.grad.zero_()
            l.bias.grad.zero_()

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


# Create a Pytorch Dataset out of our data
class MyDataset(Dataset):
    def __init__(self, X, Y):
        # Just some dummy data: features and labels
        self.X = X
        self.Y = Y

    def __len__(self):
        return X.shape[1]

    def __getitem__(self, idx):
        return self.X[:,idx], self.Y[:,idx]


def collate_columns(batch):
    # batch is a list of tuples: [(x1, y1), (x2, y2), ...]
    xs, ys = zip(*batch)
    # Now xs and ys are tuples of tensors with shape [1]
    # Convert to shape [1, 1] and then concat along dim=1
    xs = [x.unsqueeze(0) if x.dim() == 1 else x for x in xs]
    ys = [y.unsqueeze(0) if y.dim() == 1 else y for y in ys]
    # Concatenate along dim=1 instead of dim=0
    x_cat = torch.cat(xs, dim=1)
    y_cat = torch.cat(ys, dim=1)
    return x_cat, y_cat


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
    N = args.N # Number of points in the batch
    H = args.H # Size of the hidden layer
    B = args.batch_size
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
        optimizer = AdamOptimizer(learning_rate=lr)
    else:
        optimizer = SimpleOptimizer(learning_rate=lr)

    dataset = MyDataset(X, Y)
    sampler = RandomSampler(dataset)  # samples elements randomly without replacement
    dataloader = DataLoader(dataset, batch_size=B, sampler=sampler, collate_fn=collate_columns)
    c = 0 # global iteration count
    for e in range(args.epochs):
        for batch_idx, (X_, Y_) in enumerate(dataloader):
            o = model(X_)    # Forward pass
            loss = criterion(o, Y_)
            if args.capture_frames:
                if c % 2 == 0:
                    draw_movie_frame(model, c/2)
            loss.backward()
            optimizer.step(model.layers)
            model.zero_grad()
            c = c + 1
        # print loss after every epoch
        print(loss)

    print('done')

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
