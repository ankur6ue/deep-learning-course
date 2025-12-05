# README — Module 2 · Class 2.1  
Building Simple Neural Networks

© 2025 Ankur Mohan

## Pre-read
- Regularization (weight decay, dropout)
- Adam and AdamW optimizers

## Topics Covered
1. Structure of neural networks  
2. Loss functions  
3. Mini-batch gradient descent  
4. Optimizers (SGD, RMSProp, Adam, AdamW)  
5. Learning rate schedules  
6. Regularization  
7. Hyperparameters  
8. Example: Predicting sin(x)  
9. Manual forward/backward implementation  
10. PyTorch Dataset and DataLoader  

## Structure of a Neural Network
Neural networks consist of layers, nonlinearities, loss functions, and forward/backward passes. Backpropagation computes gradients layer by layer.

## Loss Functions
- Regression: MSE  
- Classification: Cross-entropy  
Ground truth may be supervised, weakly supervised, or self-supervised.

## Mini-Batch Gradient Descent
Batch size influences gradient noise and compute cost. One full pass over the training set = epoch.

## Weight Updates
Weights updated using:
θ ← θ − η ∇θ L  
Modern optimizers add momentum, adaptivity, regularization, and schedules.

## Learning Rate Schedules
Examples: step decay, linear decay, cosine annealing, warmup.

## Regularization
- Weight decay encourages small weights.  
- Dropout improves robustness.  
- Underfitting vs overfitting considerations.

## Optimizers
**SGD**: uniform treatment of parameters  
**RMSProp**: adaptive per-parameter LR  
**Adam**: momentum + RMSProp + bias correction  
**AdamW**: Adam with decoupled weight decay  

## Hyperparameters
Learning rate, hidden layer size, activation, batch size, schedule, weight decay, dropout.

## Code Examples
- simple_nn1.py — simple sin(x) predictor  
- simple_nn2.py — manual NN training loop  
- simple_nn3.py — PyTorch Dataset/DataLoader  

## Homework
1. Run simple_nn2.py; record loss & curve.  
2. Try 'simple' optimizer; compare.  
3. Use Adam + varying hidden sizes.  
4. Change activation to Swish; compare.  
5. Save loss histories & compare.  
6. Step through code in debugger.  
7. Test model on −3π to 3π; observe generalization.

## Appendix: Backpropagation
Slides cover derivatives for weights, biases, ReLU, and batched backprop.