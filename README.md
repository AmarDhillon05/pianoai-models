# PianoAI Models

This repository contains a set of machine learning models evaluated for piano note generation using a pianoroll representation of classical piano music. The primary goal of this project was to identify a model capable of accurate next-note prediction while remaining lightweight enough for serverless deployment.

## Objective

The core objective was to train and evaluate multiple generative architectures for piano note prediction and select a model that balances musical quality, inference speed, and strict memory constraints. The final target environment was AWS Lambda, which imposed a hard 128 MB memory limit and required fast cold-start performance.

## Dataset

The models were trained on a classical piano dataset represented as sparse pianorolls, sourced from a publicly available Kaggle dataset. Each piece is converted into a time-by-pitch matrix, where time steps correspond to musical frames and pitch corresponds to MIDI note indices.

The training task is framed as next-step prediction: given a fixed-length window of previous timesteps, the model predicts the next pianoroll frame.

## Evaluated Models

The following model families were implemented and evaluated:

- Generative Adversarial Network (GAN)
- Wasserstein GAN (WGAN)
- Transformer-based sequence model

Each approach was evaluated on prediction stability, output quality, training complexity, and deployability under memory constraints.

## Selected Model

The Transformer-based model was selected as the final architecture due to:

- Superior stability compared to GAN-based approaches
- Strong next-note prediction performance
- Deterministic inference behavior
- Compatibility with model compression and static graph export

The architecture uses a lightweight attention stack with a small number of blocks and heads, projecting pianoroll inputs into a fixed embedding space before applying multi-head self-attention and feedforward layers.

## Training

Training was performed using PyTorch and PyTorch Lightning. The loss function combines multiple regression-based components to encourage both numerical accuracy and structural similarity between predicted and target pianoroll frames. Learning rate scheduling and early stopping were used to control overfitting and training time.

## Deployment

To support AWS Lambda deployment under the 128 MB memory constraint, the final Transformer model was:

- Simplified to minimize parameter count
- Converted to TorchScript using `torch.jit`
- Deployed without external runtime dependencies beyond core PyTorch

TorchScript export enabled fast startup times and removed the need for Python-level model construction during inference.

## Repository Structure

- Model definitions for GAN, WGAN, and Transformer architectures
- Training and evaluation notebooks
- TorchScript-compatible Transformer implementation for deployment
- Utilities for pianoroll preprocessing and visualization

## Summary

This project demonstrates that a carefully constrained Transformer model can outperform GAN-based approaches for piano note generation when deployment constraints are a primary concern. By optimizing for memory footprint and inference speed, the final model is suitable for real-time, serverless music generation workloads.
