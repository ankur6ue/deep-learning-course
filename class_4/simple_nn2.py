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


# Here, we'll take each step in the calculation of the loss function using swiGLU and represent it as a graph. We'll
# write the forward and backward calculation explicitly as static functions which are invoked using the apply function
# from the corresponding module. This is standard practice in PyTorch. We learn several things from this exercise:
# 1. If the forward function operates on n inputs the number of outputs in the corresponding backward function must
# also be n (assuming all inputs require gradients)
# 2. Inputs to a layer (which are activations of the previous layer) and intermediate calculations made in the forward
# pass are generally re-used in the backward pass. This lets us trade memory for compute. We could save the inputs
# and intermediate calculations using the save_for_backward function (more memory), or recompute in the
# backward pass (more compute). This family of techniques is called activation checkpointing
# 3. save_for_backward must be used to save tensors, while non-tensors (such as sizes) can be saved directly on the
# context, see Sum function as an example and:
# https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html

# First the functions..
class ReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save the input for the backward pass (for gradient calculation)
        ctx.save_for_backward(x)
        # Apply ReLU: max(0, input_tensor)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        x, = ctx.saved_tensors
        # Calculate the gradient: 1 where input > 0, 0 otherwise
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        return grad_input

# Converts logits into probabilities and then cross entropy loss
class SoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        exp_scores = torch.exp(x)
        # we are using columns are the batch dimension
        probs = exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)
        # Save for use in the backward pass (for gradient calculation)
        ctx.save_for_backward(probs)
        return probs


    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        x, = ctx.saved_tensors
        # for each element of the batch (row vector of size m*1), the jacobian (m*m) is:
        # denoting o as output of softmax and x as input
        # do_dx = torch.diag(x) - torch.matmul(x, x.T)
        # when multiplied with grad_output, this becomes
        # do_dx = torch.mul(x, grad_output) - x @ x.T @ grad_output
        # when B > 1, we need to concatenate
        do_dx = []
        B = x.size(1)
        for i in range(0, B):
            # By using x[[i]], the [i] is interpreted as a list containing a single index 0. This tells PyTorch to
            # select the element at index i and keep it within its own dimension, resulting in a shape of n*1
            do_dx.append(torch.mul(x[:, [i]], grad_output[:,[i]]) - x[:,[i]] @ x[:,[i]].T @ grad_output[:,[i]])
        out = torch.cat(do_dx, dim=1)
        return out


class CrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, labels):
        # Save the input for the backward pass (for gradient calculation)
        exp_scores = torch.exp(y)
        probs = exp_scores / torch.sum(exp_scores, dim=0, keepdims=True)

        ctx.save_for_backward(probs, labels)
        B = y.size(1)
        return -torch.sum(torch.log(probs[(labels.T, range(0, B))]))/B

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input from the forward pass
        probs_, labels = ctx.saved_tensors
        probs = torch.clone(probs_)
        B = probs.size(1)
        probs[[labels.T, range(0, B)]] -= 1
        return probs/B, torch.zeros(labels.size())

class HadamardProdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # Save tensors needed for backward pass in ctx
        ctx.save_for_backward(x, y)
        output = torch.mul(x, y)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output
        # torch.mul (or *) computes element wise product, while torch.matmul computes dot product
        # this is equivalent to the below, for B = 1
        # return torch.matmul(torch.diag(y.squeeze()), grad_output), torch.matmul(torch.diag(x.squeeze()), grad_output)


class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        # Save tensors needed for backward pass in ctx
        ctx.save_for_backward(x, W, b)
        output = torch.matmul(W, x)
        if b is not None:
            output += b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, W, b = ctx.saved_tensors
        grad_input = W.T @ grad_output
        grad_weight = grad_output @ input.T
        grad_bias = torch.clone(grad_output).sum(dim=1, keepdim=True)
        return grad_input, grad_weight, grad_bias

class SumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # in backward, we just need the size of x, not the entire tensor. We can attach the size
        # directly on the ctx object
        ctx.sz_x = x.size()
        output = torch.sum(x) # will sum along both dimensions
        return output


    @staticmethod
    def backward(ctx, grad_output):
        sz = ctx.sz_x
        grad_x = torch.ones(sz)
        return grad_x


# Now the modules..
class ReLUModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ReLUFn.apply(x)

# Now the modules..
class SoftmaxModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SoftmaxFn.apply(x)


class CrossEntropyLossModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, labels):
        return CrossEntropyLossFn.apply(y, labels)

class LinearModule(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        scale = 100
        l = nn.Linear(_in, _out, dtype=torch.float32)
        self.weight = torch.randn(_out, _in) * np.sqrt(1/_in)
        self.bias = torch.randn(_out, 1) * np.sqrt(1/_in)
        # self.weight = l.weight.data
        self.weight.requires_grad = True# _out * _in
        # self.bias = l.bias.data.unsqueeze(1) # _out * 1. The unsqueeze is to add the 2nd dimension
        self.bias.requires_grad = True

    def forward(self, x):
        return LinearFn.apply(x, self.weight, self.bias)


class HadamardProdModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return HadamardProdFn.apply(x, y)


class SumModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SumFn.apply(x)

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
        self.fc1 = LinearModule(input_size, hidden_size)
        self.fc2 = LinearModule(hidden_size, hidden_size)
        self.fc3 = LinearModule(hidden_size, output_size)
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

    def step(self, lr, reg):
        for l in self.layers:
            l.weight.data -= lr * (l.weight.grad + reg * l.weight.data)
            l.bias.data -= lr * l.bias.grad


    def zero_grad(self):
        for l in self.layers:
            # Zero the gradients, otherwise they'll accumulate
            l.weight.grad.zero_()
            l.bias.grad.zero_()


if __name__ == "__main__":
    # This makes plots show up as a separate figure
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='Create a spiral dataset and build a small neural network to identify'
                                                 'points belonging to each spiral')

    parser.add_argument('--N', type=int, default=200, help='Number of points in one arm of the spiral dataset')
    parser.add_argument('--D', type=int, default=2, help='Dimensionality of the data')
    parser.add_argument('--K', type=int, default=4, help='Number of classes')
    parser.add_argument('--H', type=int, default=150, help='Size of hidden layer')
    args = parser.parse_args()
    N = args.N # Number of points per class
    D = args.D # Dimension of a point (2 for 2D)
    K = args.K # Number of classes (arms of the spiral)
    H = args.H # Size of the hidden layer

    X_, y_ = create_dataset(K, N) # B * N
    # lets visualize the data:
    plt.scatter(X_[:, 0], X_[:, 1], c=y_, s=40, cmap=plt.cm.Spectral)
    plt.show()
    X = torch.from_numpy(X_).T # N * B
    X.requires_grad = True
    # convert to tensors
    y = torch.from_numpy(y_)
    y = y.unsqueeze(1)
    # create the model
    model = SimpleNN(D, H, K)
    criterion = CrossEntropyLossModule()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_iter = 1000
    for iter in range(num_iter):
        # optimizer.zero_grad() # Zero the gradients
        logits = model(X)    # Forward pass
        loss = criterion(logits, y)
        print(loss)# Calculate loss
        predicted_class = np.argmax(logits.detach().numpy(), axis=0)
        acc = np.mean(predicted_class == y_)
        print(acc)
        loss.backward()
        model.step(lr=.2, reg=0.0005)
        # write every 10th frame to disk, to convert into a movie later
        # if iter % 10 == 0:
        #   draw_movie_frame(model, iter/10)

        model.zero_grad()

    print('ok')