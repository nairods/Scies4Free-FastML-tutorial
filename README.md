# (Fast)ML Tutorial: From Introductory Classification to FPGA Deployment

This repository contains a hands-on tutorial series that introduces key machine learning concepts through progressively more advanced examples, starting with a simple medical classification task and building up to real-time particle classification and FPGA deployment.

The tutorial is designed to run in Jupyter notebooks, either locally or in Google Colab, and is aimed at students, beginners in machine learning, and researchers who want an accessible introduction to both standard ML workflows and hardware-aware inference.
---

## Motivation and Background
Machine learning has become an important tool across science, engineering, medicine, and industry. It is widely used to identify patterns in data, make predictions, and support decision-making, especially in problems where traditional rule-based approaches become difficult to scale. As datasets continue to grow in size and complexity, machine learning is also playing an increasingly important role in research and scientific discovery.

This workshop introduces machine learning in a gradual and practical way. We begin with a simple classification problem to build intuition for core concepts such as features, labels, training, evaluation, and model interpretation, and then move to more complex applications such as particle classification in high-energy physics.

A final goal of the workshop is to show that building an accurate model is often only part of the problem. In many real-world applications, models must also satisfy constraints on latency, memory usage, power consumption, or hardware resources, which makes efficient implementations increasingly important. This is where topics such as quantisation, pruning, and FPGA-oriented deployment become relevant.
---

## Tutorial Overview

This workshop is structured as a gradual introduction to machine learning in science:

### 1. Introductory classification with the Heart Disease dataset
We begin with a simple binary classification problem to introduce core ML concepts such as features, labels, training, validation, testing, and model evaluation.

### 2. Particle classification with jet data
We then move to a more realistic scientific application: classifying particle jets in high-energy physics. This part introduces more complex datasets and compares different neural network architectures.

### 3. Fast machine learning for hardware deployment
Finally, we explore how machine learning models can be adapted for low-latency inference on FPGAs using hls4ml, including techniques such as quantisation-aware training and pruning.

This progression is intended to give participants both a solid ML foundation and an appreciation for the challenges of deploying models in real-time systems.

---

## How to Run the Tutorial

The notebooks are intended to be runnable in Binder, Google Colab or in a local Jupyter environment.

Each notebook includes:
- Environment setup (package installation)
- Clear execution order
- Explanatory text and visualisations

You can either work through the notebooks in sequence or use individual notebooks as standalone examples.

---

## Intended Audience

This tutorial is aimed at:
- Students learning the basics of machine learning
- Participants in introductory ML workshops
- Researchers interested in scientific machine learning
- Practitioners curious about low-latency inference and FPGA deployment

No prior FPGA experience is required, and only basic Python familiarity is assumed.

---

## Acknowledgements

This tutorial builds on ideas and tools developed within the CMS trigger and FastML communities, including open-source projects such as **OpenML**, **hls4ml** and **QKeras**.
