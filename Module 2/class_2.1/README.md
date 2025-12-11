# Lecture 2.1 — Building Simple Neural Networks
© 2025 Ankur Mohan

## Agenda
1. Structure of a Neural Network
2. Loss Functions
3. Optimization & Training Dynamics
4. Regularization Techniques
5. Optimizers
6. Weight Initialization & Variance Propagation
7. PyTorch Implementation
8. Homework & Self‑Study

## Pre‑Read Materials
- 3Blue1Brown Episode 1: https://www.3blue1brown.com/lessons/neural-networks
- MIT Introduction to Deep Learning (gradient descent): https://youtu.be/iOh7QUZGyiU
- CS231n Neural Networks Part 1: https://cs231n.github.io/neural-networks-1/
- Khan Academy Multivariable Derivatives: https://www.khanacademy.org/math/multivariable-calculus

## Learning Objectives
- Understand forward and backward passes
- Implement gradient‑based training manually
- Explain overfitting and regularization
- Understand Xavier/He initialization  
- Compute mean/variance of ReLU outputs

## Key Math
### Linear Layer
y = Wx + b

∂E/∂W = δ xᵀ  
∂E/∂b = δ  
∂E/∂x = Wᵀ δ  

### ReLU Statistics
If y ~ N(0, σ²):

E[ReLU(y)] = σ / √(2π)  
Var(ReLU(y)) = σ² (1/2 − 1/(2π))

### He Initialization
Var(W_ij) = 2 / n_in

## Homework
### Math
1. Derive E[ReLU(Y)] and Var(ReLU(Y)].
2. Show variance explosion with naive initialization.
3. Explain exploding variance despite zero mean.

### Coding
1. Monte‑Carlo variance experiment.
2. Activation histogram visualization.
3. Dropout comparison experiment.

## License
© 2025 Ankur Mohan