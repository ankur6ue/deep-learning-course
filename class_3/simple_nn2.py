import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
# compute derivative of swiGLU of a vector x and compare gradient calculated by autograd against manual calculation
# See swiGLU section in https://arxiv.org/pdf/2410.10989 and swiglue{i}.jpg for derivation of manual derivative
# Set the seed for reproducibility
torch.manual_seed(42)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib
import sys
# The below is ugly and not recommended, but lets us run the code as a file (python -m simple_nn1.py) instead
# of converting it into a package..
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.losses import CrossEntropyLossModule
from utils.optimizer import AdamOptimizer, SimpleOptimizer
# Tensorboard related
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# MLFlow related
import mlflow

# Create a spiral dataset with N points and K arms
def create_dataset(K, N, D=2):
    np.random.seed(42)
    # Code to generate spiral dataset
    X = np.zeros((N * K, D), dtype=np.float32)  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype=np.long)  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


# Ask the network to predict the class of each point in a 2D matrix and create an image
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

# Our simple network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = LinearModule(input_size, hidden_size, "fc1")
        # Hidden layer
        self.fc2 = LinearModule(hidden_size, hidden_size, "fc2")
        self.fc3 = LinearModule(hidden_size, output_size, "fc3")
        self.relu = ReLUModule()  # Activation function
        self.layers = []
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)
        self.layers.append(self.fc3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # This makes plots show up as a separate figure
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='Create a spiral dataset and build a small neural network to identify'
                                                 'points belonging to each spiral')

    parser.add_argument('--N', type=int, default=200, help='Number of points in one arm of the spiral dataset')
    parser.add_argument('--D', type=int, default=2, help='Dimensionality of the data')
    parser.add_argument('--K', type=int, default=4, help='Number of classes')
    parser.add_argument('--H', type=int, default=150, help='Size of hidden layer')
    parser.add_argument('--capture_frames', action='store_true', help='If set, every other frame is'
                                                                      'captured and saved to a frames directory')
    parser.add_argument('--optimizer', choices=['simple', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size (should divide N)')

    args = parser.parse_args()
    N = args.N # Number of points per class. The size of the dataset then will be K*N
    D = args.D # Dimension of a point (2 for 2D)
    K = args.K # Number of classes (arms of the spiral)
    H = args.H # Size of the hidden layer
    B = args.batch_size # Size of each batch used to average gradients
    lr = args.lr # Learning rate

    X_, Y_ = create_dataset(K, N) # B * N
    # lets visualize the data:
    plt.scatter(X_[:, 0], X_[:, 1], c=Y_, s=40, cmap=plt.cm.Spectral)
    plt.show()
    X = torch.from_numpy(X_).T # D * N*K
    X.requires_grad = True
    # convert to tensors
    Y = torch.from_numpy(Y_)
    Y = Y.unsqueeze(1).T # (N*K * 1) Add an extra dimension. Needed for matrix math to work
    # create the model
    model = SimpleNN(D, H, K)
    # We use cross entropy loss, because our network is predicting the class of each point
    criterion = CrossEntropyLossModule()
    if args.optimizer == "adam":
        optimizer = AdamOptimizer(model.layers, learning_rate=lr)
    else:
        optimizer = SimpleOptimizer(model.layers, learning_rate=lr)

    c = 0  # global iteration count
    for e in range(args.epochs):
        # Each epoch uses a different permutation of indices
        indices = torch.randperm(N*K)
        num_iter_per_epoch = (int)(N*K / args.batch_size)
        for iter in range(num_iter_per_epoch):
            batch_indices = indices[iter * B: (iter + 1) * B]
            X_ = X[:, batch_indices] # K, B
            Y_ = Y[:, batch_indices] # K, B
            logits = model(X_)  # Forward pass. logits are size K, B
            loss = criterion(logits, Y_)
            predicted_class = np.argmax(logits.detach().numpy(), axis=0)
            acc = np.mean(predicted_class == Y_.detach().numpy())
            loss_np = loss.detach().numpy()
            print(f"At iteration num {c}, Loss  = {loss_np: .4f}, Accuracy = {acc: .2f}")
            # Write tensorboard logs
            writer.add_scalar("Loss/train", loss_np, global_step=c)
            writer.add_scalar("Accuracy", acc, global_step=c)
            # Calculate gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            if args.capture_frames:
                if c % 2 == 0:
                    draw_movie_frame(model, c / 2)
            optimizer.zero_grad()
            c = c + 1
        # We'll generally calculate validation loss after each epoch and logg it..
        # calculate validation loss
        # writer.add_scalar("Loss/val", loss_np, e)
        # writer.add_scalar("Accuracy", acc, e)

    # Calculate accuracy on the entire dataset
    logits = model(X)  # Forward pass
    loss = criterion(logits, Y)
    predicted_class = np.argmax(logits.detach().numpy(), axis=0)
    acc = np.mean(predicted_class == Y.detach().numpy())
    print(f"Accuracy on entire dataset = {acc: >4}")
    hparams = {
        'learning_rate': lr,
        'batch_size': B,
        'optimizer': args.optimizer
    }
    metrics = {
        'hparam/accuracy': acc,
        'hparam/loss': loss.detach().numpy()
    }
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
    # This ensures the logs are written to the disk
    writer.flush()
    print('done')