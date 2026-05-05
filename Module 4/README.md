# Module 4: Information Retrieval and Embeddings

Module 4 introduces an important application of neural networks--information retrieval, the family of techniques used to retrieve relevant items from a large collection. The material in this module covers the language of ranking and relevance, the classical machinery of sparse retrieval, approximate similarity search, and the role of learned vector representations in dense retrieval.

This module gives the reader a practical understanding of how modern retrieval systems are evaluated and built. It also creates a strong bridge into later topics involving embeddings, semantic similarity, vector search, and language-model-based applications.

## Why This Module Matters

Many real machine learning systems are retrieval systems in some form. Search engines, recommendation systems, question answering systems, and document lookup pipelines all depend on being able to score or rank candidates effectively.

This module helps answer questions like:

- what does it mean for a retrieved result to be relevant?
- how should a ranking system be evaluated?
- how do lexical retrieval systems work internally?
- how can similarity be approximated efficiently at scale?
- what do embeddings add to a retrieval system?

The ideas in this module are useful both on their own and as preparation for later material involving transformers, vector representations, and LLM-based retrieval pipelines.

## What You Will Learn

### 1. Relevance and Ranking Metrics

The module begins with the basics of information retrieval: a query, a collection of items, and a relevance notion that determines whether the returned results are useful.

The reader is introduced to:

- binary and graded relevance
- precision, recall, and F1
- rank-aware metrics such as Precision@k, Recall@k, Average Precision, mean Average Precision, and Mean Reciprocal Rank
- the trade-offs involved in tuning a retrieval system for different applications

This part of the module gives the reader a disciplined way to think about ranking quality and evaluation.

### 2. Lexical Retrieval and Sparse Indices

The next part of the module covers classical sparse retrieval. The reader sees how systems can represent documents and queries in terms of exact lexical units and then retrieve candidate documents efficiently with indexing structures.

Topics include:

- token-based matching
- n-grams and phrase queries
- inverted indices
- sparse document representations
- the operational logic of lexical retrieval

This section gives the reader a strong foundation in how retrieval systems work before learned embeddings enter the picture.

### 3. Similarity Search with MinHash and LSH

The module then turns to approximate similarity search. This introduces the idea that exact pairwise comparison can be too expensive at scale, and that compact randomized representations can preserve useful structure for retrieval.

The reader learns:

- set and string similarity
- Jaccard-style similarity intuition
- MinHash as a sketching method
- locality-sensitive hashing as a way to retrieve similar items efficiently
- the relationship between approximation quality and search efficiency

This part of the module is especially useful for understanding how large-scale similarity search can remain computationally practical.

### 4. Semantic Similarity and Dense Retrieval

The final part of the module introduces dense retrieval and embeddings. The focus shifts from exact lexical matching to learned vector representations that capture semantic relationships.

Topics include:

- vector-space representations of words and documents
- semantic neighborhood structure
- cosine similarity and embedding comparisons
- training and inspecting simple embedding models
- how dense retrieval supports meaning-based search

This gives the reader a concrete entry point into embedding-based systems that later connect naturally to transformers and vector databases.

## Lecture-by-Lecture Overview

| Lecture | Theme | Practical Emphasis |
| --- | --- | --- |
| `class_4.1` | Information retrieval setup and evaluation metrics | ranking quality, relevance, and retrieval metrics |
| `class_4.2` | Lexical retrieval and classical sparse indexing | phrase matching, n-grams, and inverted-index style retrieval |
| `class_4.3` | Set similarity, MinHash, and locality-sensitive hashing | approximate similarity search and scalable retrieval intuition |
| `class_4.4` | Semantic similarity and dense retrieval with embeddings | learned vector representations and embedding-space neighbors |

## How the Module Connects to the Rest of the Course

This module extends the course from prediction problems to retrieval problems. It provides the evaluation language needed for search and ranking, and it introduces embedding ideas that later become central in transformer-based systems and vector search workflows.

The material also builds intuition for why semantic representations matter. That intuition is useful for later modules involving tokenization, transformers, language models, and LLM serving systems.


## What You Should Be Able to Do Afterward

After Module 4, the reader should be able to:

- explain the core metrics used to evaluate retrieval systems
- describe how lexical retrieval systems organize and search text
- understand the purpose of MinHash and locality-sensitive hashing
- explain how embeddings support semantic similarity and dense retrieval
- reason about the trade-offs between exact matching, approximate matching, and semantic matching

This module gives the reader a strong conceptual and practical foundation for retrieval systems, which are a major component of many modern machine learning applications.
