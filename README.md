# Balancing Privacy and Performance

A Comparative Study of Gradient Clipping Techniques in DP-SGD

As privacy concerns rise in the field of deep learning, Differentially Private Stochastic Gradient Descent (DP-SGD) offers a promising approach by modifying traditional SGD to include mechanisms that safeguard user data. However, a key challenge in DP-SGD is determining the optimal gradient clipping threshold that balances privacy with model performance.

This project investigates two methods of implementing differential privacy in machine learning: Automatic Clipping and Random Sparsification, within the Differentially Private Stochastic Gradient Descent (DP-SGD) framework. We aim to evaluate how these methods impact the privacy, utility, and performance of models trained on the Duke dataset using U-Net and Nested U-Net architectures, training them with [Opacus](https://github.com/pytorch/opacus). By analyzing these approaches, the project seeks to determine optimal strategies for balancing effective learning with robust data privacy in deep learning models.

Here's the essential information about the project:

---

### Objective

Evaluate differential privacy techniques, specifically **Automatic Clipping** and **Random Sparsification**.

### Key Challenge

Determining optimal gradient clipping values to balance privacy and performance.

### Methods Analyzed

- **Automatic Clipping**: Adjusts clipping threshold dynamically for optimal learning.
- **Random Sparsification**: Introduces randomness in gradient updates to enhance privacy.

| Per-sample gradient clipping\Open Source Library | Opacus     |
|--------------------------------------------------|----------- |
| Automatic Clipping                               | ✅         |
| Random Sparsification                            | WIP [ ]    |

### Machine learning task

I will evaluate the performance and Privacy of the segmentation of retinal images with the following networks.

### Models Used

- U-Net
- Nested U-Net

### Dataset

[Duke dataset](./data/DukeData/)

### Goals

- Measure the impact of each privacy method on model utility and performance.
- Compare the privacy levels achieved by each method.
- Plot the metrics and show the performance for different noise multipliers.

---