# FastML Tutorial: Particle Classification and FPGA Deployment

This repository contains a hands-on tutorial on **fast machine learning (FastML)** for real-time particle classification in high-energy physics, with a focus on **low-latency inference and FPGA deployment**.  
The tutorial is designed to be run entirely in **Jupyter notebooks**, either locally or directly in **Google Colab**, and is suitable for students and researchers with basic Python and ML familiarity.

---

## Motivation and Background

At the CERN Large Hadron Collider (LHC), experiments such as **CMS** record collisions at an extremely high rate. Since it is neither technically nor economically feasible to store all collision data, a multi-level **trigger system** is used to decide, in real time, which events should be kept for further analysis. The CMS trigger system consists of two stages: the Level-1 (L1) Trigger, a custom hardware-based system that reduces the event rate by more than 99%, followed by the High-Level Trigger (HLT), a software-based system running on CPU and GPU farms that further reduces the data volume by approximately 95%.

The **Level-1 (L1) Trigger** operates under particularly strict constraints:
- Latency of order **microseconds**
- Limited on-chip memory and logic resources
- Implementation on **FPGAs**

As collision complexity increases at the **High-Luminosity LHC (HL-LHC)**, traditional cut-based algorithms become increasingly difficult to scale. **Machine-learning-based particle classification**, deployed directly on FPGAs, has therefore become a key strategy to maintain physics performance while respecting real-time constraints.

This tutorial walks through the full chain from model training to hardware-oriented synthesis, illustrating the practical trade-offs involved.

<img width="698" height="434" alt="image" src="https://github.com/user-attachments/assets/060e77c9-a470-4084-b97a-1c716608a92a" />

---

## Tutorial Overview

The tutorial follows a realistic end-to-end workflow:

### 1. Dataset and Physics Task
- Use of the **OpenML `lhc_jets` dataset**
- Multiclass particle classification task (e.g. different jet or particle types)
- Discussion of features, labels, and performance metrics
- Contextualisation within LHC trigger use cases

### 2. Machine Learning Models
We train and compare several model families commonly used in fast inference scenarios:
- Dense Neural Networks (DNNs)
- Set-based architectures (e.g. DeepSets)
- Transformer-based models 

Model performance is evaluated in terms of signal efficiency versus trigger rate, ensuring that physics performance is maximised without exceeding the limited trigger bandwidth.

### 3. Quantisation and Pruning
To make models suitable for FPGA deployment, we explore:
- Fixed-point homogeneous quantisation using **QKeras**
- Advanced heterogeneous quantisation techniques (**HGQ**)
- Model pruning and architectural simplifications
- Performance comparisons between floating-point and quantised models

The impact of quantisation on both accuracy and hardware suitability is discussed.

### 4. HLS and FPGA Synthesis
Quantised models are converted to hardware descriptions using **High-Level Synthesis (HLS)** tools:
- **hls4ml** (primary framework)
- Optional comparisons with other toolchains (e.g. da4ml)

We demonstrate:
- Model-to-HLS conversion
- Configuration and optimisation choices
- Latency and resource estimation

### 5. FPGA Perspective
Finally, we examine what a real FPGA synthesis flow looks like by:
- Inspecting example **Xilinx Vitis/Vivado synthesis reports**
- Interpreting resource usage (LUTs, DSPs, BRAM)
- Understanding latency, initiation interval, and throughput constraints
- Discussing how these metrics relate back to trigger requirements

---

## How to Run the Tutorial

All notebooks are intended to be runnable in **Google Colab** without local installation.

Each notebook includes:
- Environment setup (package installation)
- Clear execution order
- Explanatory text and visualisations

You can open any notebook directly in Colab via the provided links.

---

## Intended Audience

This tutorial is aimed at:
- Students and early-career researchers in HEP
- ML practitioners interested in low-latency inference
- Participants in FastML or trigger-related workshops

No prior FPGA experience is required.

---

## Acknowledgements

This tutorial builds on ideas and tools developed within the CMS trigger and FastML communities, including open-source projects such as **hls4ml**, **QKeras**, and **OpenML**.
