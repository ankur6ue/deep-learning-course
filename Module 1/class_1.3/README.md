# Deep Learning Module 1 — Class 3  
## Mathematical Preliminaries: Partial Derivatives and Optimization

### 📚 Overview
This lecture extends differentiation to **multiple dimensions**.  
We learn to compute **partial derivatives**, understand their geometric meaning, and generalize the **chain rule** to computational graphs.  
Finally, we apply these ideas to **linear regression** as an optimization problem.

### 🧩 Agenda
- Introduction to derivatives in multiple dimensions  
- Geometric interpretation of gradients  
- The chain rule in computational graphs  
- Using partial derivatives to solve linear regression  

### 📘 Pre-read Materials
- 📗 Paul’s Online Notes — Sections 13.2–13.7 (Partial Derivatives, Chain Rule, Directional Derivatives)  
- 🎥 [Khan Academy: Linear Algebra – Vectors and Spaces](https://www.khanacademy.org/math/linear-algebra)  
- 💻 NumPy Tutorial: *Linear Algebra Basics* (`numpy.dot`, `numpy.linalg.norm`)  
- 📘 *Matrix Calculus* Sections 1.1 – 2.1.1  

### 💻 Code Examples
| Script | Description |
|--------|-------------|
| `verify_partial_derivatives.py` | Numerical check of analytical partial derivatives |
| `direction_derivatives.py` | Visualize direction of maximum ascent |
| `chain_rule.py` | Demonstrate multi-node computational graph differentiation |
| `linear_regression.py` | Solve linear regression using gradient-based optimization |

### 🔑 Key Concepts
- Partial derivatives treat all other variables as constants.  
- The gradient vector  
  \[
  \nabla f(x,y) = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]
  \]
  points in the direction of steepest ascent.  
- **Hessian matrix** summarizes second-order curvature information.  
- **Chain rule** extends to multivariate functions and computational graphs.  
- Linear regression minimizes mean-squared error:
  \[
  L(a,b) = \frac{1}{N}\sum_i (y_i - (a x_i + b))^2
  \]

### 🧮 Homework
1. Compute and verify partial derivatives numerically.  
2. Complete *Matrix Calculus* Exercises 1.4.  
3. Fit a line to y = x sin x using `linear_regression.py`.  
4. Add a constraint on b (regularization) and derive new a,b updates.

© 2025 Ankur Mohan
