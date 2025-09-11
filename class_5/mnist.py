import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os
import sys
import matplotlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.losses import CrossEntropyLossModule
from utils.optimizer import AdamOptimizer

# 1. Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# 2. Data Loading and Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. Defining the MLP Architecture
# Our simple network with a few hidden layer
class MLP():
    def __init__(self, input_size, hidden_size, output_size):
        self.relu1 = nn.ReLU()
        self.fc1 = LinearModule(input_size, hidden_size, "fc1")
        # Hidden layer
        self.fc2 = LinearModule(hidden_size, output_size, "fc3")
        self.relu = ReLUModule()  # Activation function
        self.layers = []
        self.layers.append(self.fc1)
        self.layers.append(self.fc2)

    # Makes the class "calleable", i.e., we can call an object of this class.
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # x is 28 by 28, we need to flatten the image
        x = self.fc1(x.T) # our network operates on input_dim * B
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # This makes plots show up as a separate figure
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='Create a spiral dataset and build a small neural network to identify'
                                                 'points belonging to each spiral')

    parser.add_argument('--H', type=int, default=128, help='Size of hidden layer')
    parser.add_argument('--capture_frames', action='store_true', help='If set, every other frame is'
                                                                      'captured and saved to a frames directory')
    parser.add_argument('--optimizer', choices=['simple', 'adam'], default='adam')
    parser.add_argument('--lr_scheduler', choices=['linear', 'cosine', 'constant'], default='linear')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size (should divide N)')
    args = parser.parse_args()
    # 3. Training the Model
    model = MLP(28 * 28, args.H, 10)
    criterion = CrossEntropyLossModule()
    if args.optimizer == "adam":
        optimizer = AdamOptimizer(model.layers, learning_rate=args.lr)


    # 5. Training Loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # 6. Evaluation

    correct = 0
    total = 0

    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 0)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')