# Module 2: Building and Training Simple Neural Networks

Module 2 is where the course starts to feel like practical machine learning. The mathematical ideas from Module 1 are used to build simple neural networks from scratch.

The module covers the structure of simple feedforward networks, the logic of the training loop, the role of the optimizer, and the practical concerns that shape model quality such as regularization, initialization, and variance propagation.

## Why This Module Matters

Many introductions to deep learning explain the mathematics of gradients and then jump quickly to high-level frameworks. This module connects those two layers carefully.

The material here answers questions like:

- what actually happens during an epoch of training?
- how do forward pass, loss computation, backward pass, and optimizer steps fit together?
- how do training, validation, and test sets play different roles?
- what causes underfitting and overfitting?
- why do initialization and regularization matter even in relatively small networks?

This module is where the reader starts building intuition for the entire lifecycle of supervised learning.

## What You Will Learn

### 1. The Structure of a Simple Neural Network

The module begins with the anatomy of a neural network: layers, activations, outputs, and parameters. The reader sees how these pieces combine into a mapping from input features to predictions.

This part establishes:

- how linear layers and non-linear activations are composed
- how network depth changes representational capacity
- how classification problems are framed in terms of outputs and labels

The emphasis is on making the network structure intuitive before turning to training dynamics.

### 2. The Training Loop

The core of the lecture is the training process. The reader is taken through the steps that appear in almost every neural network training pipeline:

- sample a mini-batch
- run a forward pass
- compute a loss against ground truth
- run a backward pass
- update parameters with an optimizer
- evaluate on validation data

This section gives the reader a procedural understanding of training rather than a purely symbolic one.

### 3. Regularization and Generalization

This section introduces regularization as a practical tool in stabalizing training.

The reader learns:

- what underfitting looks like
- what overfitting looks like
- why small weights are often desirable
- how weight decay shapes the optimization process
- how dropout changes the way the network relies on its hidden units

This part gives a very useful early intuition for why a model that fits the training data well is not automatically a good model.

### 5. Initialization and Variance Propagation

This section covers various techniques used to initialize the parameters of a neural network. 

Topics include:

- activation magnitude across layers
- variance growth or shrinkage
- ReLU output statistics
- Xavier and He initialization

This gives the reader a more mechanistic understanding of stable training and prepares them for deeper architectures later in the course.

### 6. Practical Neural Network Examples

The code in this module grounds the ideas in simple but meaningful experiments:

- small feedforward neural networks
- MNIST-based classification workflows
- PyTorch-based training examples
- synthetic classification problems such as spiral-style datasets
- simple visualizations of feature behavior and network outputs

These examples make the training loop tangible and show how theory turns into model behavior.

## Current Module Structure

At the moment, the repository content for Module 2 is concentrated in a single lecture directory, `class_2.1`, but that lecture already covers a substantial amount of material:

- neural network structure
- loss functions
- optimization and training dynamics
- regularization
- initialization and variance propagation
- implementation in PyTorch

The module also includes supporting data and model artifacts for MNIST-style experiments.

## Lecture Overview

| Lecture | Theme | Practical Emphasis |
| --- | --- | --- |
| `class_2.1` | Building simple neural networks and training loops | supervised training, regularization, initialization, MNIST and synthetic classification examples |

## How to Work Through This Module

A good way to use this module is:

1. go through the lecture slides to understand the training pipeline
2. run the smaller neural network examples to build intuition
3. move to the MNIST examples to see a more complete training workflow
4. revisit the initialization and variance sections after seeing training in practice

This module is especially useful if the reader pauses to connect training curves and model behavior back to the concepts of loss, gradient, and regularization introduced in the lecture.

## What You Should Be Able to Do Afterward

After Module 2, the reader should be able to:

- explain the full supervised learning training loop
- distinguish training, validation, and test usage clearly
- train a small neural network in PyTorch
- describe common causes of underfitting and overfitting
- explain the purpose of weight decay and dropout
- reason about why initialization affects optimization stability

This module is the bridge between mathematical preparation and modern neural network practice. It makes the rest of the course more concrete, because later architectures and systems are still built on the same training logic introduced here.
