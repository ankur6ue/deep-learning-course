# Deep Learning Module 1 — Class 5  
## Adding Non-Linearities and Constructing a Simple Neural Network

### 📚 Overview
In this lecture we introduce **non-linear activation functions** that allow neural networks to model non-linear relationships between inputs and outputs.  
Building on the previous lecture (two-layer linear network), we demonstrate why stacking purely linear layers still yields an affine function and how adding a non-linearity such as **ReLU** or **tanh** enables complex mappings like `sin(x)`.

### 🧩 Agenda
1. Recap: Linear networks are affine  
2. Why we need non-linearities  
3. Adding element-wise activations (ReLU, tanh)  
4. Two-layer network to learn sin(x)  
5. Mini-batch gradient descent (batch size B)  
6. Backpropagation through activations  
7. Other common non-linearities (Leaky ReLU, GELU, Swish)

### 📘 Pre-Read Materials
| Type | Resource | Description |
|------|-----------|-------------|
| 🎥 Video | [3Blue1Brown — *“But what is a neural network?”* (Deep Learning, Ch. 1)](https://www.youtube.com/watch?v=aircAruvnKk) | Short, visual intro to why we need non-linear activations. Episode 1 of 3Blue1Brown’s *Neural Networks* series. |
| 📗 Lecture | [MIT 6.S191 — *Intro to Deep Learning* (Lecture 1 PDF)](https://introtodeeplearning.com/2019/materials/2019_6S191_L1.pdf) | Concise slides introducing neural networks, gradient descent, and activations. |
| 📘 Article | [Stanford CS231n — *Deep Learning for Computer Vision: Neural Networks Part 1*](https://cs231n.github.io/neural-networks-1/) | Excellent written overview of common activation functions and their properties. |
| 🎓 Math Refresher | [Khan Academy — *Multivariable Derivatives + Chain Rule*](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives) | Handy review of partial derivatives and the chain rule used in backpropagation. |

### 💻 Code Examples
| Script | Description |
|--------|-------------|
| `two_linear_no_nonlinearity.py` | Two stacked linear layers — affine fit |
| `two_linear_with_tanh_denominator.py` | Two layers + tanh activation — non-linear fit |
| `non_linearities.py` | Implements and compares ReLU, tanh, Leaky ReLU, GELU |

### 🔑 Key Concepts
- Composition of linear transforms remains linear → need non-linear activations  
- Activations are **parameter-free**, applied element-wise  
- Backpropagation uses the **derivative of the activation** in each layer  
- Mini-batch gradient descent balances gradient noise vs. compute cost  

### 🧮 Homework

**Coding**
1. Modify two_linear_with_ReLU.py to replace ReLU with tanh non-linearity. Compare final losses
2. Experiment with different batch sizes B and learning rates; observe training stability.

© 2025 Ankur Mohan
