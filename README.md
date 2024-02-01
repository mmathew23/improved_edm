# Analyzing and Improving the Training Dynamics of Diffusion Models

## Overview
This repository contains an unofficial implementation of the research paper [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696v1) authored by Tero Karras, Janne Hellsten, Miika Aittala, Timo Aila, Jaakko Lehtinen, and Samuli Laine from NVIDIA and NVIDIA Aalto University. The paper focuses on addressing challenges in the training of the ADM diffusion model architecture, specifically targeting uncontrolled magnitude changes and imbalances in network activations and weights during training. This repo is a sample implementation of the paper using pixel-based diffusion and not latent diffusion.

## Key Contributions
- [X] **Modification of Network Layers:** The paper proposes a systematic redesign of network layers to preserve activation weight and update magnitudes, resulting in significantly better networks without increasing computational complexity.
- [X] **Magnitude-preserving Mechanisms:** Several configurations like Magnitude-preserving learned layers (CONFIG D) and Magnitude-preserving fixed-function layers (CONFIG G) are introduced to control activation magnitudes and maintain a balance in the network. This repo does not implement Magnitude preserving fixed functions (CONFIG G) as they seem to be tuned for latent diffusion.
- [X] **Post-hoc EMA Technique:** A novel method for setting the exponential moving average (EMA) parameters post-training is presented. This allows for precise tuning of EMA length and reveals its interactions with network architecture, training time, and guidance.


## Usage
# Train
```
accelerate launch train.py
```

Configs are located in the configs folder. By default it uses mlp style weighting from the paper, but if you want to use loss weighting from the original edm paper run

```
accelerate launch train.py loss_type='scaled'
```

To resume a failed or stopped training run
```
accelerate launch train.py +resume=CHECKPOINT
```

# Inference
The paper shows a way to find EMA weights after training, instead of retraining to sweep over different decay factors. In order to run inference you first need to calculate the weights and store them. Training saves ema_checkpoints in  config.output_dir/ema_checkpoints.

First run the following command to solve for weights, substituting the appropriate folder and sigmas.

```
python scripts/solve_posthoc_ema.py --ema_checkpoint_path=results/cifar10/ema_checkpoints "--target_sigma_rels=[0.05,0.075,0.1]"
```
This will save weights in a folder following this structure {OUTPUT_DIR}/posthoc_ema_checkpoints_{SNAPSHOT_T}/srel_{SIGMA_REL}. You can now use this folder as the CHECKPOINT to pass into generate.py

```
python generate.py CHECKPOINT
```

## Requirements
```
pip install -r requirements.txt
```

## References
```
@article{karras2023analyzing,
title={Analyzing and Improving the Training Dynamics of Diffusion Models},
author={Karras, Tero and Hellsten, Janne and Aittala, Miika and Aila, Timo and Lehtinen, Jaakko and Laine, Samuli},
journal={arXiv preprint arXiv:2312.02696},
year={2023}
}
```


