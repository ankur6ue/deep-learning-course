## Module 1: Foundations

### Lecture 1.1: Introduction and history of Neural Networks
- Biological analogy
- Rise of deep learning
- Overview of course roadmap

### Lecture 1.2: Mathematical Preliminaries

- Optimization
- Univariate differential calculus
- Gradient descent 

### Lecture 1.3: Mathematical Preliminaries

- Vectors, matrices, linear transformations
- Notation conventions
- Multivariate differential calculus

### Lecture 1.4: Mathematical Preliminaries

- Random variables, probability distributions
- KL Distance
- Expectation, Variance
 

## Module 2: Neural Networks

### Lecture 2.1: Feedforward Neural Networks
- Neural network architecture
- MLPs
- Non-linearities
- Forward/Backward pass

### Lecture 2.2: Backpropagation
- Derivation of backprop equations
- Introduction to Pytorch

### Lecture 2.4: Losses
- Mean Square Error
- Cross Entropy, Log Likelihood

### Lecture 2.3: Monitoring/Observability
- Tensorboard, MLFlow, W&B

### Lecture 2.4: Regularization and Generalization
- Weight decay, dropout
- Bias-Variance trade-off

### Lecture 2.5: Advanced Optimization
- Adam, AdamW
- Learning rate schedules (linear, cosine, warmup)

### Lecture 2.6: Initialization and Normalization
- Xavier/He initialization
- Batchnorm, Layernorm, Groupnorm

## Module 3: Other Neural Network Architectures
### Lecture 3.1:
- Convolutional Neural Networks
- Skip connections
- Efficiency Tricks

## Module 4: GPUs
### Lecture 4.1:
- Why GPUs
- GPU architecture
- CUDA

### Lecture 4.2:
- Pipelining data transfer and data processing
- Profiling GPU code using NVIDIA nSight

### Lecture 4.3:
- Writing a custom CUDA Kernel using Triton
- Matrix Mul
- Softmax

## Module 5: Neural Network Training: Advanced Topics
### Lecture 5.1: Memory usage and techniques to lower usage
- Memory consumed by Neural Networks
- Lowering memory usage using Activation Checkpointing and Gradient Accumulation

### Lecture 5.2: Distributed Computing on CPUs
- Multiprocessing
- Streaming

### Lecture 5.2: Distributed Training: Distributed Data Parallel

### Lecture 5.3: Distributed Training: Fully Sharded Data Parallel

## Module 6: Language Modeling, Embeddings and VectorDB
### Lecture 6.1: Language Modeling basics
- N-grams, distributional hypothesis
- Word embeddings (Word2Vec, GloVe)
- Tokenization: Bytepair (BPE), Wordpiece

### Lecture 6.2: Vector Databases
- Indexing techniques
- Precision/Recall trade-off
- Popular VectorDB and evaluation criteria

## Module 7: Attention and Transformers
Goal: Transition into the transformer era

### Lecture 7.1: Introduction to Attention
- Motivation, soft alignment
- Self-attention mechanism

### Lecture 7.2: The Transformer Architecture (Part 1)
- Encoder-Decoder Architecture
- Multi-head self-attention
- Positional Encodings

### Lecture 7.3: Pre-Training strategies
- Causal LM vs Masked LM: GPT vs BERT strategies
- Decoding Strategies in GPT: Beam Search, Top-K, Nuclear Sampling..
- Task specific Heads: Text Classification 

### Lecture 7.4: Training Medium-Sized LMs
- Pre-training medium-sized BERT model on GPUs
- Finetuning
- Parallelism: Data, Model

### Lecture 7.5: Training Medium-Sized LMs
- Pre-training GPT2 style models

## Module 8: Reinforcement Learning

## Module 9: Interpretability/Explainability

## Module 10: Hyperparameter Optimization