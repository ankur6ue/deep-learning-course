"""
MNIST MLP training script with selectable loss functions and TensorBoard logging.

This script:
- Builds a simple MLP (column-major: inputs are shaped (features, batch)).
- Loads MNIST with torchvision.
- Supports Cross-Entropy or fused Softmax+MSE loss.
- Logs training loss and test accuracy to TensorBoard.

Usage:
    python train_mnist.py --H 128 --lr 1e-3 --epochs 2 \
        --loss_fn x-entropy --batch_size 64 --lr_scheduler linear
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Allow "utils" imports when running from this file's directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.linear import LinearModule
from utils.relu import ReLUModule
from utils.losses import CrossEntropyLossModule, FusedSoftmaxMSELossModule
from utils.optimizer import AdamOptimizer


class MLP:
    """
    A minimal MLP with one hidden linear layer and ReLU, implemented for
    column-major training (inputs: (features, batch)).

    Architecture:
        - fc1: LinearModule(input_size -> hidden_size)
        - ReLU
        - fc2: LinearModule(hidden_size -> output_size)

    Notes
    -----
    - Input images are flattened to 28*28 and then transposed to (features, B)
      because the custom LinearModule expects (in_features, batch).
    - This class is a plain Python object; it does not inherit from nn.Module
      because the provided LinearModule/AdamOptimizer manage params manually.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.fc1 = LinearModule(input_size, hidden_size, "fc1")
        self.relu = ReLUModule()
        self.fc2 = LinearModule(hidden_size, output_size, "fc2")
        # Layers in order for the custom optimizer to iterate over
        self.layers = [self.fc1, self.fc2]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Enable calling the model instance directly."""
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images, shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits with shape (C, B), where C is the number of classes.
        """
        # Flatten to (B, 784) then transpose to (784, B)
        x2d = x.view(-1, 28 * 28).T
        xh = self.fc1(x2d)
        xh = self.relu(xh)
        out = self.fc2(xh)
        return out  # (C, B)


def tsne_plot(data: np.ndarray, targets: np.ndarray, frame_num: int) -> None:
    """
    Visualize 2D t-SNE embedding and save a frame image.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, D) with features to embed.
    targets : np.ndarray
        Array of shape (N,) with integer labels for coloring.
    frame_num : int
        Frame index used in the saved filename.

    Notes
    -----
    Saves to "<script_dir>/frames/frameXXXX.png".
    """
    tsne = TSNE(n_components=2, random_state=42)
    x_transformed = tsne.fit_transform(data)

    tsne_df = pd.DataFrame(
        np.column_stack((x_transformed, targets)),
        columns=["X", "Y", "Targets"],
    )
    tsne_df.loc[:, "Targets"] = tsne_df["Targets"].astype(int)

    plt.figure(figsize=(10, 8))
    g = sns.FacetGrid(data=tsne_df, hue="Targets", height=8)
    g.map(plt.scatter, "X", "Y").add_legend()
    plt.tight_layout()

    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"frame{frame_num:04d}.png"))
    plt.close()


def unnormalize(img: torch.Tensor, mean: float = 0.1307, std: float = 0.3081) -> torch.Tensor:
    """
    Reverse MNIST normalization for visualization.

    Parameters
    ----------
    img : torch.Tensor
        Normalized image tensor with values ~N(0,1).
    mean : float
        Dataset mean used in normalization.
    std : float
        Dataset std used in normalization.

    Returns
    -------
    torch.Tensor
        De-normalized tensor with values approximately in [0, 1].
    """
    return img * std + mean


def build_loss(loss_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Construct the selected loss function.

    Parameters
    ----------
    loss_name : str
        Either "x-entropy" or "fused-softmax-mse".

    Returns
    -------
    Callable
        A callable (output, target) -> scalar loss.
    """
    if loss_name == "x-entropy":
        ce = CrossEntropyLossModule()

        def ce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # output: (C, B), target: (B,)
            return ce(output, target)

        return ce_loss

    if loss_name == "fused-softmax-mse":
        fsm = FusedSoftmaxMSELossModule()

        def fsm_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Convert integer labels to one-hot probabilities (C, B)
            one_hot = F.one_hot(target, num_classes=10).T.to(output.dtype)
            return fsm(output, one_hot)

        return fsm_loss

    raise ValueError(f"Unknown loss_fn: {loss_name}")


def visualize_batch(train_loader: DataLoader, num_show: int = 8) -> None:
    """
    Show a small batch of MNIST images with labels.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader for training set.
    num_show : int
        Number of images to display from the first batch.
    """
    it = iter(train_loader)
    example_data, example_targets = next(it)

    cols = min(num_show, example_data.size(0))
    fig, axes = plt.subplots(1, cols, figsize=(12, 3))
    for i in range(cols):
        img = unnormalize(example_data[i][0])
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Label: {example_targets[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main training entry point."""
    # Use an interactive backend if desired
    matplotlib.use("TkAgg")

    parser = argparse.ArgumentParser(description="Build an MLP to classify MNIST data.")
    parser.add_argument("--H", type=int, default=128, help="Size of hidden layer.")
    parser.add_argument("--lr_scheduler", choices=["linear", "cosine", "constant"], default="linear")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--loss_fn", choices=["x-entropy", "fused-softmax-mse"], default="x-entropy")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size (should divide N).")
    args = parser.parse_args()

    # --- Model & loss ---
    model = MLP(28 * 28, args.H, 10)
    criterion = build_loss(args.loss_fn)

    # --- Data ---
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean & std
        ]
    )
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("../data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Process the entire test dataset in one go for convenience
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Optional: visualize a few samples
    visualize_batch(train_loader, num_show=8)

    # --- Optimizer (for custom layers) ---
    optimizer = AdamOptimizer(model.layers, learning_rate=args.lr)

    # --- TensorBoard setup ---
    run_name = f"mnist_loss={args.loss_fn}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_hparams({"loss_fn": args.loss_fn}, metric_dict={}, run_name=run_name)

    # --- Training loop ---
    global_step = 0
    skip_frames = 20  # evaluate/log every N steps

    print("Steps per epoch:", len(train_loader))
    for epoch in range(args.epochs):
        for local_step, (data, target) in enumerate(train_loader):
            output = model(data)                  # (C, B)
            loss = criterion(output, target)     # scalar

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if local_step % skip_frames == 0:
                # Training loss
                print(f"Epoch {epoch}, step {local_step}, loss = {loss.item():.4f}")
                writer.add_scalar("train/loss", loss.item(), int(global_step / skip_frames))

                # Evaluate on the full test set
                correct = 0
                total = 0
                for data_test, target_test in test_loader:
                    out_test = model(data_test)              # (C, B_test)
                    # Example: first layer features if you want t-SNE later
                    # l1_out = model.layers[0](data_test.view(-1, 28 * 28).T)
                    preds = out_test.argmax(dim=0)           # (B_test,)
                    total += target_test.size(0)
                    correct += (preds == target_test).sum().item()

                accuracy = 100.0 * correct / total
                print(f"Accuracy after step {global_step}: {accuracy:.2f}%")
                writer.add_scalar("test/accuracy", accuracy, int(global_step / skip_frames))

            global_step += 1

        print(f"Epoch {epoch} completed. Last loss: {loss.item():.4f}")

    writer.close()


if __name__ == "__main__":
    main()
