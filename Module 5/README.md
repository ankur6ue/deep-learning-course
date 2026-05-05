# Module 5: Neural Architectures and Transformers

Module 5 is where the course moves from training simple models to understanding the architectures that define modern deep learning systems. The material covers the design patterns that make large neural networks expressive, trainable, and useful across vision and language tasks.

This module includes convolutional networks, normalization, residual connections, attention, tokenization, transformer pretraining objectives, and decoding methods for autoregressive models. It provides the conceptual base for the later modules on LLM serving, inference systems, and optimization.

## Why This Module Matters

Architectures determine what a neural network can represent, how efficiently it can be trained, and how it behaves at inference time. This module introduces the core building blocks that appear repeatedly in modern machine learning systems.

The material helps answer questions like:

- how do convolutional models process spatial data?
- why do normalization and residual connections improve training dynamics?
- What is inductive bias?
- what does self-attention compute?
- how do tokenization schemes shape language model behavior?
- what distinguishes encoder-style and decoder-style transformer training?
- how does a generative model produce text token by token?

This module also connects cleanly to later practical questions about memory usage, packed sequences, decoding speed, and inference-system design.

## What You Will Learn

### 1. Convolutional Architectures

The module begins with convolutional neural networks and the mechanics of convolution layers. The reader is introduced to the structure of convolution operations and the gradients needed to train them.

Topics include:

- convolution as a learned local pattern detector
- gradients with respect to kernels and inputs
- feature extraction in image-like data
- the practical value of CNNs in classification tasks

This section helps ground the idea that architecture should reflect the structure of the input domain.

### 2. Normalization and Residual Learning

The next part of the module explains two major ideas that shape modern deep networks: normalization and skip connections.

The reader learns:

- how batch normalization works
- why layer normalization is important in transformers
- how residual connections support gradient flow
- how residual blocks help deep models remain trainable

These ideas are important not only for CNNs, but also for transformer models and later systems-oriented discussions about precision and training stability.

### 3. Representation and Embedding Intuition

The module includes a smaller representation-focused section around learned embeddings and their geometry. This helps the reader build intuition for how neural networks organize information internally.

The main value of this section is conceptual:

- embeddings as learned continuous representations
- geometric structure in representation space
- how internal features can be visualized and inspected

This prepares the reader for the attention and tokenization material that follows.

### 4. Attention and Transformer Mechanics

A major part of the module is devoted to transformers. The reader is introduced to attention as a mechanism for modeling relationships across tokens and then to the transformer block as a reusable architecture for sequence modeling.

Topics include:

- token interactions through attention
- sequence padding and masking
- packed and unpacked sequence layouts
- transformer blocks and their major components
- the role of attention in language understanding and generation

This section is foundational for the rest of the course, especially the modules dealing with language models and serving systems.

### 5. Tokenization

The module then turns to tokenization, which is a core interface between text and neural models.

The reader learns:

- why tokenization matters
- how byte-pair encoding works
- how merge-based tokenization creates subword vocabularies
- how vocabulary design affects sequence length and model inputs

This section provides a practical understanding of the input pipeline behind modern NLP systems.

### 6. Transformer Pretraining Objectives

The module also introduces the objectives and model structures used for transformer pretraining and adaptation.

Topics include:

- masked language modeling in encoder-style transformers
- causal next-token prediction in decoder-style transformers
- classification heads and special tokens
- the role of output projections into vocabulary space
- fine-tuning and task adaptation

This gives the reader a clear picture of how transformer architectures are used in both understanding and generation settings.

### 7. Decoding for Generative Models

The final part of the module focuses on decoding, which is central to autoregressive text generation.

The reader learns:

- greedy decoding
- temperature scaling
- top-k sampling
- top-p (nucleus) sampling
- repetition control and generation quality trade-offs

This section is especially useful as preparation for the later modules on LLM inference and serving, where decoding behavior becomes a systems concern as well as a modeling concern.

## Lecture-by-Lecture Overview

| Lecture | Theme | Practical Emphasis |
| --- | --- | --- |
| `class_5.1` | Convolutional neural networks | convolution structure, convolution gradients, and image-model intuition |
| `class_5.2` | Normalization techniques and skip connections | batch norm, layer norm, residual learning, and training stability |
| `class_5.3` | Representation and embedding experiments | embedding geometry and internal representation intuition |
| `class_5.4` | Transformer attention mechanics | token interactions, masking, packed sequences, and attention behavior |
| `class_5.5` | Tokenization schemes | subword tokenization and byte-pair encoding |
| `class_5.6` | Transformer pretraining and adaptation | BERT-style and GPT-style objectives, masking, and classification heads |
| `class_5.7` | Decoding methods for autoregressive models | generation control through sampling and search choices |
| `class_5.8` | Additional decoding material | continued treatment of decoding strategies and generation behavior |

## How the Module Connects to the Rest of the Course

Module 5 gives the reader the architectural vocabulary needed for the rest of the repository. Later modules on retrieval, serving, quantization, and optimization all depend on the ideas introduced here.

The transformer material is especially central. It supports a better understanding of:

- language model pretraining
- KV-cache-based inference
- tokenization effects
- decoding latency and throughput
- the design of modern LLM serving stacks

## What You Should Be Able to Do Afterward

After Module 5, the reader should be able to:

- describe the main neural architecture patterns used in modern deep learning
- explain the role of normalization and residual connections
- explain self-attention and the structure of a transformer block
- understand how tokenization shapes model inputs
- distinguish masked and causal language modeling objectives
- describe the major decoding strategies used in text generation

This module is one of the main conceptual turning points in the course. It brings the reader from small trainable networks to the architecture family that underlies modern language models and many of the systems explored later in the repository.
