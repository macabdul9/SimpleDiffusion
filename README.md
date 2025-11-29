# Diffusion Model Experiments

This project contains toy experiments to visualize and understand diffusion models through various data generation and transformation scenarios.

## Overview

The experiments demonstrate how diffusion models (encoder/decoder) process data through different stages:
1. **Data Generation**: Starting with 2D data (Gaussian or GMM)
2. **Dimensionality Transformation**: Mapping 2D → 3D (linear or non-linear MLP)
3. **Diffusion Process**: Visualizing encoder (noise addition) and decoder (noise removal) steps

## Experiments

### Experiment I: 2D Gaussian → Linear 3D → Diffusion
- Generates 2D Gaussian data using eigendecomposition: `x = m + E*sqrt(lambda)*e`
- Transforms to 3D using linear transformation: `xx = A*x`
- Runs diffusion through encoder/decoder stages

### Experiment II: 2D GMM → Linear 3D → Diffusion
- Generates 2D data from Gaussian Mixture Model (3 components)
- Transforms to 3D using linear transformation
- Runs diffusion process

### Experiment III: 2D Gaussian → MLP 3D → Diffusion
- Generates 2D Gaussian data
- Transforms to 3D using a "wonky" non-linear MLP (warps the manifold)
- Runs diffusion process

### Experiment IV: 2D GMM → MLP 3D → Diffusion
- Generates 2D GMM data
- Transforms to 3D using non-linear MLP
- Runs diffusion process

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
source venv/bin/activate
python diffusion_experiments.py
```

## Outputs

All outputs are saved in the `diffusion_outputs/` directory:

- **Visualizations**: 
  - `experiment_I_diffusion.png` - 2D Gaussian with linear transform
  - `experiment_II_diffusion.png` - 2D GMM with linear transform
  - `experiment_III_diffusion.png` - 2D Gaussian with MLP transform
  - `experiment_IV_diffusion.png` - 2D GMM with MLP transform

- **Statistics**: `experiment_stats.json` - Contains mean, variance, and transformation matrices for each experiment

- **Logs**: `diffusion_experiments.log` - Detailed logging of all operations

## Visualization Structure

Each visualization shows:
- **Top row**: Forward diffusion (Encoder) - from 3D manifold to isotropic Gaussian
- **Bottom row**: Reverse diffusion (Decoder) - from isotropic Gaussian back to recovered manifold

The diffusion process uses 1000 steps, with 10 representative steps visualized.

## Results

### Experiment Summaries

**Experiment I: 2D Gaussian → Linear 3D → Diffusion**
- **Data**: 2D Gaussian using eigendecomposition (`x = m + E*sqrt(lambda)*e`)
- **Transform**: Linear 3x2 matrix (`xx = A*x`)
- **Result**: MSE 0.000103, isotropic ratio 1.068
- **Purpose**: Tests diffusion on a simple Gaussian manifold with a linear transformation

**Experiment II: 2D GMM → Linear 3D → Diffusion**
- **Data**: 2D Gaussian Mixture Model (3 components)
- **Transform**: Linear 3x2 matrix (`xx = A*x`)
- **Result**: MSE 0.000101, isotropic ratio 1.026
- **Purpose**: Tests diffusion on a multi-modal distribution with a linear transformation

**Experiment III: 2D Gaussian → MLP 3D → Diffusion**
- **Data**: 2D Gaussian using eigendecomposition
- **Transform**: Non-linear MLP (warps the manifold)
- **Result**: MSE 0.000096, isotropic ratio 1.027
- **Purpose**: Tests diffusion on a Gaussian manifold with a non-linear transformation

**Experiment IV: 2D GMM → MLP 3D → Diffusion**
- **Data**: 2D Gaussian Mixture Model (3 components)
- **Transform**: Non-linear MLP (warps the manifold)
- **Result**: MSE 0.000100, isotropic ratio 1.011
- **Purpose**: Tests diffusion on a multi-modal distribution with a non-linear transformation

### Overall Summary

These four experiments demonstrate diffusion on toy 2D data projected to 3D. They vary data type (single-mode 2D Gaussian vs multi-modal 2D GMM) and transformation (linear 3x2 matrix vs non-linear MLP). All use 1000 diffusion steps to show the forward process (curved manifold → isotropic Gaussian) and the reverse process (isotropic Gaussian → recovered manifold). Results show near-perfect reconstruction (MSE ~0.0001) and that the final step is close to isotropic (std ratio ~1.0–1.07), indicating the diffusion process can encode curved manifolds into isotropic Gaussians and decode them back, regardless of data modality or transformation type. This validates the core mechanism: gradual noise addition flattens the manifold, and the reverse process recovers the original structure.

## Key Concepts Demonstrated

1. **Gaussian Data Generation**: Using eigendecomposition to generate correlated 2D Gaussian data
2. **Linear vs Non-linear Manifolds**: Comparing linear transformations (A*x) vs non-linear MLP transformations
3. **Diffusion Process**: Visualizing how noise is added (encoder) and removed (decoder) step-by-step
4. **Manifold Warping**: The MLP creates non-linear warping, making the diffusion process more complex

## Code Structure

- `SimpleDiffusion`: Neural network-based diffusion model with encoder/decoder
- `WonkyMLP`: Non-linear transformation network for manifold warping
- Data generation functions: `generate_2d_gaussian()`, `generate_gmm_2d()`
- Transformation functions: `linear_transform_2d_to_3d()`, `mlp_transform_2d_to_3d()`
- Visualization: `visualize_diffusion_steps()` - Creates comprehensive plots

## Parameters

You can modify these in the code:
- `n_samples`: Number of data points (default: 1000)
- `num_steps`: Diffusion steps (default: 1000)
- `hidden_dim`: Neural network hidden dimension (default: 64)
- `n_components`: GMM components (default: 3)

