# ECG-PCG-CycleGAN-Transformer

This repository implements a **CycleGAN-based deep learning framework** for **bidirectional ECG ⇄ PCG signal translation**.

The model uses **Conv1D downsampling**, a **Transformer encoder**, and **Conv1D upsampling** to handle long biomedical time-series efficiently while preserving temporal structure.

## Architecture
- CycleGAN framework (ECG → PCG → ECG, PCG → ECG → PCG)
- 1D Convolutional Generators with Transformer bottleneck
- Patch-based 1D Discriminators
- Adversarial, cycle-consistency, and identity losses

## Key Features
- Handles long signals (48,000 samples)
- Efficient temporal modeling using Transformer encoders
- Stable training with CycleGAN losses
- Designed for cardiac signal translation and analysis

## Use Cases
- ECG → PCG signal generation
- PCG → ECG signal reconstruction
- Cardiac abnormality and heart attack research
- Multimodal biomedical signal modeling

## Status
Active research and development.
