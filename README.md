# Analyzing and Improving the Training Dynamics of Diffusion Models

## Overview
This repository contains an unofficial implementation of the research paper [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696v1) authored by Tero Karras, Janne Hellsten, Miika Aittala, Timo Aila, Jaakko Lehtinen, and Samuli Laine from NVIDIA and NVIDIA Aalto University. The paper focuses on addressing challenges in the training of the ADM diffusion model architecture, specifically targeting uncontrolled magnitude changes and imbalances in network activations and weights during training.

## Key Contributions
- [X] **Modification of Network Layers:** The paper proposes a systematic redesign of network layers to preserve activation weight and update magnitudes, resulting in significantly better networks without increasing computational complexity.
- [ ] **Magnitude-preserving Mechanisms:** Several configurations like Magnitude-preserving learned layers (CONFIG D) and Magnitude-preserving fixed-function layers (CONFIG G) are introduced to control activation magnitudes and maintain a balance in the network.
- [ ] **Post-hoc EMA Technique:** A novel method for setting the exponential moving average (EMA) parameters post-training is presented. This allows for precise tuning of EMA length and reveals its interactions with network architecture, training time, and guidance.


## Usage

## Requirements

## References
```
@article{karras2023analyzing,
title={Analyzing and Improving the Training Dynamics of Diffusion Models},
author={Karras, Tero and Hellsten, Janne and Aittala, Miika and Aila, Timo and Lehtinen, Jaakko and Laine, Samuli},
journal={arXiv preprint arXiv:2312.02696},
year={2023}
}
```


