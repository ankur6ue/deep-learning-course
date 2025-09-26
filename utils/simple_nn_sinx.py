import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.leakyrelu import LeakyReLUModule
from utils.tanh import TanhModule
from utils.gelu import GeluModule
from utils.swish import SwishModule

# Create our spiral dataset
def create_dataset(N):
    np.random.seed(0)
    x_np = np.linspace(-2 * np.pi, 2 * np.pi, N)
    y_np = np.sin(x_np) + 0.1 * np.random.randn(*x_np.shape)  # noisy sin(x)

    X = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y


class SimpleNN():
    def __init__(self, input_size, hidden_size, output_size, non_linearity, scale=1):
        self.fc1 = LinearModule(input_size, hidden_size, "fc1", scale)
        self.fc2 = LinearModule(hidden_size, output_size, "fc2", scale)

        if non_linearity == "leaky_relu":
            self.non_linearity = LeakyReLUModule()
        elif non_linearity == "tanh":
            self.non_linearity = TanhModule()
        elif non_linearity == "swish":
            self.non_linearity = SwishModule()
        else: # default is ReLU
            self.non_linearity = ReLUModule()

        self.layers = []
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)

    # Makes the class "calleable", i.e., we can call an object of this class.
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
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