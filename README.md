# Deep Learning Course

This repository is a project-based deep learning course organized as lecture modules, slide decks, runnable labs, and systems experiments. The material moves through core mathematical foundations, building and training small neural networks, information retrieval, transformers, reinforcement learning, LLM serving and multi-GPU distributed training

The course is meant to be used in two complementary ways:

- as a structured teaching sequence, where concepts build across modules
- as a hands-on lab repository, where lectures are paired with executable code

## Intended Audience

This course is aimed at:

- software engineers who want a practical introduction to deep learning
- ML engineers who want to connect model internals with systems concerns
- data scientists interested in operational aspects of deep learning models such as distributed training, internals of the vllm serving system, profiling GPU code using nVidia profiler etc. 

Python familiarity is assumed. The early modules cover the mathematical background needed for the rest of the repository.

## Repository Structure

Most modules are organized into `class_x.y/` directories. Lecture folders typically contain:

- a slide deck (`.pptx`)
- Python scripts illustrating the ideas in the lecture
- pre-read/post-read materials and suggested homework assignments and readings. 

## Module Overview

### Module 1: Foundations for Deep Learning

Introduces deep learning as function approximation and optimization. Covers derivatives, gradient descent, multivariate calculus, chain rule, linear regression, the role of non-linearities, and probability fundamentals.

### Module 2: Building and Training Simple Neural Networks

Focuses on the mechanics of neural network training: forward and backward passes, regularization, initialization, and practical classification examples such as MNIST and synthetic datasets.

### Module 3: Monitoring, Drift, and Interpretability

Explains how to observe and debug models after training. Covers experiment tracking, monitoring, drift detection, SHAP-based explanations, and Integrated Gradients.

### Module 4: Information Retrieval and Embeddings

Introduces retrieval metrics, lexical retrieval, sparse indexing, MinHash/LSH, semantic similarity, and dense retrieval using learned embeddings.

### Module 5: Neural Architectures and Transformers

Covers CNNs, normalization, residual learning, attention, tokenization, transformers, BERT/GPT-style pretraining, and decoding methods.

### Module 6: Distributed Training on Multi-GPU and Multi-Node Clusters

Reserved for future material on DDP, FSDP, communication, checkpointing, and scaling training across multiple GPUs and nodes.

### Module 7: Reinforcement Learning

Reserved for future material on MDPs, policy/value methods, actor-critic algorithms, exploration, and RL for language models.

### Module 8: LLM Inference Systems 

Covers the design principles behind LLM serving systems and how they are implemented in vLLM. Includes building a simple vllm-lite inference server, that implements the core innovations in vLLM such as prefix caching and pre-fill decode disaggregation, paged attention and others.

### Module 9: Quantization and Optimization

Covers key techniques for quantization and optimization of neural networks, such as Activation Aware Quantization (AWQ), using hardware-specific numerical formats such as NVFP4 


