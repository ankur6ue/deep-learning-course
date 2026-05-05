# Module 3: Monitoring, Drift, and Interpretability

Module 3 focuses on the question that becomes important as soon as a model is trained and begins interacting with real data: how do we understand what the model is doing, whether it is still behaving well, and why it made a particular prediction?

This module covers two closely related themes:

- observing model behavior over time in something closer to a production setting
- interpreting model predictions at the level of individual features and examples

Together, these topics give the reader a practical view of model reliability. The material connects metrics, data quality, distribution shift, and explanation methods in a way that is useful for both machine learning research and real-world deployment.

## Why This Module Matters

A trained model is only the beginning of the story. In practice, models operate on changing data, produce outputs that need to be monitored, and influence decisions that often need explanation.

This module helps answer questions like:

- how do we know whether a model is still behaving as expected after deployment?
- what kinds of drift can happen in inputs and outputs?
- how do we systematically track the effect of modifying hyperparameters on model performance?
- how can we explain a single prediction to a human reader?
- how do local explanations differ from global summaries of model behavior?


## What You Will Learn

### 1. Experiment Tracking and Model Monitoring

The first part of the module introduces the operational side of machine learning systems. It covers tracking model outputs over experiment tracking

Topics include:

- experiment tracking during development
- monitoring input data quality
- output distribution monitoring
- label and ground-truth feedback loops
- model performance metrics over time

The lecture and lab material also give a sense of how monitoring fits into a larger data and serving pipeline.

### 2. Drift Detection

Drift is one of the core ideas in this module. The reader is introduced to the fact that a model can become unreliable because the world around it changes, even if the model weights stay fixed.

The module explores:

- input distribution drift
- output distribution drift
- out-of-distribution behavior
- sliced and aggregate distance measures
- feature-level and class-level monitoring perspectives

This part of the module is especially useful for building intuition about what should be measured once a model leaves a notebook and starts receiving live traffic.

### 3. Explainability with SHAP

The second major theme is interpretability. The module begins with SHAP and related feature-attribution ideas for tabular-style models and structured feature spaces.

The reader learns:

- how SHAP assigns contribution values to input features
- what it means to compare a prediction against a baseline
- the difference between local explanations and global feature importance
- how SHAP values can reveal nonlinear behavior and feature interactions

This gives the reader a principled way to talk about why a prediction moved up or down relative to an expected reference point.

### 4. Gradient-Based Explanations

The final part of the module introduces gradient-based explanation methods, with a strong focus on Integrated Gradients.

Topics include:

- choosing a baseline input
- interpolating between baseline and actual input
- accumulating gradients along the interpolation path
- interpreting feature attributions in image and text settings
- understanding why raw gradients can be noisy
- appreciating the importance of sanity checks for explanation methods

This section gives the reader an explanation framework that is especially relevant for differentiable deep learning models.

## Lecture-by-Lecture Overview

| Lecture | Theme | Practical Emphasis |
| --- | --- | --- |
| `class_3.1` | Experiment tracking, monitoring, and drift detection | production-style metrics, data quality checks, output drift, and monitoring pipelines |
| `class_3.2` | Interpretability and explainability with SHAP | local and global feature attribution for structured prediction problems |
| `class_3.3` | Gradient-based explanations with Integrated Gradients | attribution for deep models, baseline selection, and explanation reliability |

## How the Module Connects to the Rest of the Course

Module 3 sits naturally after training-focused material. Earlier modules explain how to build and optimize models. This module explains how to observe them, analyze them, and justify their predictions.

It also prepares the reader for later systems-oriented modules by making model behavior measurable in operational terms. Monitoring, drift, and interpretability all become more important as models scale and move closer to production usage.

## What You Should Be Able to Do Afterward

After Module 3, the reader should be able to:

- describe what should be monitored in a deployed model pipeline
- explain the difference between data drift, output drift, and performance degradation
- interpret feature-attribution explanations using SHAP
- explain the logic of Integrated Gradients
- understand the difference between local explanations and global summaries
- think more critically about whether an explanation method is actually reflecting model behavior

This module helps turn model training into a broader engineering and analytical practice. It gives the reader the tools to check whether model output is stable, observable, and understandable.
