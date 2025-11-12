# Lecture 1.6 – Fundamentals of Probability for Machine Learning

### Overview
This lecture introduces only the core probability ideas directly used later in deep-learning math—random variables, expectation, variance, covariance, joint/conditional probability, Bayes’ rule, KL divergence, and cross-entropy loss.

### Agenda
1. Motivation: where probability appears in ML  
2. Random variables (discrete vs continuous)  
3. Expectation and variance  
4. Independence and covariance  
5. Joint and conditional probabilities; Bayes’ theorem  
6. Comparing distributions (KL divergence & MSE)  
7. Cross-entropy loss and its gradient

### Pre-read (verified links)
- [Khan Academy — Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)  
- [StatQuest — Discrete vs Continuous Random Variables (video)](https://www.youtube.com/watch?v=5yF3pEr6U5g)  
- [3Blue1Brown — Variance, Covariance and Correlation](https://www.youtube.com/watch?v=4ex_I68T7G8)  
- [Khan Academy — Conditional Probability and Bayes Theorem](https://www.khanacademy.org/math/statistics-probability/probability-library)  
- [Distill.pub — Visual Information Theory (KL divergence)](https://distill.pub/2016/misread-tsne/)  
- [CS231n — Neural Networks Part 1 (Softmax Classifier)](https://cs231n.github.io/linear-classify/#softmax)

### Homework
**Conceptual:** expectation/variance derivations, Bayes examples, KL distance.  
**Coding:** simulate dice rolls, Monte Carlo expectation, KL vs MSE comparison, verify cross-entropy gradient.

### Key Takeaways
- Probability distributions underlie network outputs and loss functions.  
- KL divergence measures distance between distributions; cross-entropy is its practical counterpart in classification.  
- Expectation and variance quantify mean behavior and spread of model outputs and weights.  
