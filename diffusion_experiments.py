"""
Diffusion Model Experiments on Toy Data
Visualizes diffusion process through encoder/decoder stages
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffusion_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory
output_dir = Path('diffusion_outputs')
output_dir.mkdir(exist_ok=True)


class SimpleDiffusion:
    """Simple diffusion model with forward (encoder) and reverse (decoder) processes"""
    
    def __init__(self, num_steps=50):
        self.num_steps = num_steps
        
        # Noise schedule (beta schedule)
        self.beta = torch.linspace(0.0001, 0.02, num_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Store noise for deterministic reverse (for visualization)
        self.stored_noise = {}
    
    def forward_diffusion(self, x_0, step, store_noise=True, noise_key=None):
        """
        Forward diffusion: gradually add noise
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if step >= self.num_steps:
            step = self.num_steps - 1
        
        # Generate noise
        if store_noise and noise_key is not None and noise_key in self.stored_noise:
            # Use stored noise for deterministic reverse
            noise = self.stored_noise[noise_key]
        else:
            noise = torch.randn_like(x_0)
            if store_noise and noise_key is not None:
                self.stored_noise[noise_key] = noise
        
        alpha_bar_t = self.alpha_bar[step]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t, noise
    
    def reverse_diffusion_step(self, x_t, step, noise=None):
        """
        Reverse diffusion one step: x_{t-1} from x_t
        If noise is known (from forward process), we can compute x_0 then x_{t-1}
        """
        if step <= 0:
            return x_t
        
        alpha_bar_t = self.alpha_bar[step]
        alpha_bar_t_prev = self.alpha_bar[step - 1]
        alpha_t = self.alpha[step]
        beta_t = self.beta[step]
        
        if noise is not None:
            # Recover x_0 from x_t using known noise
            x_0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise) / (torch.sqrt(alpha_bar_t) + 1e-8)
            # Then compute x_{t-1} from x_0
            x_t_prev = torch.sqrt(alpha_bar_t_prev) * x_0 + torch.sqrt(1 - alpha_bar_t_prev) * noise
            return x_t_prev
        else:
            # Without noise, use DDPM reverse formula (would need predicted noise in real model)
            # For visualization, estimate noise crudely
            estimated_noise = (x_t - x_t.mean(dim=0, keepdim=True)) * 0.1
            x_0_approx = (x_t - torch.sqrt(1 - alpha_bar_t) * estimated_noise) / (torch.sqrt(alpha_bar_t) + 1e-8)
            x_t_prev = torch.sqrt(alpha_bar_t_prev) * x_0_approx + torch.sqrt(1 - alpha_bar_t_prev) * estimated_noise * 0.1
            return x_t_prev
    
    def forward_encoder(self, x, step=None):
        """Forward pass through encoder (add noise) - for compatibility"""
        if step is None:
            step = self.num_steps - 1
        
        x_t, noise = self.forward_diffusion(x, step, store_noise=False)
        return x_t, x_t  # Return noisy version
    
    def forward_decoder(self, x_t, step=None, noise_key=None):
        """Forward pass through decoder (remove noise) - for compatibility"""
        if step is None:
            step = self.num_steps - 1
        
        # Reverse the diffusion process
        x_recovered = self.reverse_diffusion(x_t, step, noise_key=noise_key)
        return x_recovered, x_recovered


class WonkyMLP(nn.Module):
    """Non-linear transformation MLP that warps the manifold"""
    
    def __init__(self, input_dim=2, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Tanh()  # Keep outputs bounded
        )
    
    def forward(self, x):
        return self.net(x)


def generate_2d_gaussian(n_samples=1000, m=None, V=None):
    """
    Generate 2D Gaussian data: x = m + E*sqrt(lambda)*e
    where E*lambda*E^T = V
    """
    if m is None:
        m = np.array([0.0, 0.0])
    if V is None:
        V = np.array([[2.0, 0.5], [0.5, 1.0]])
    
    # Eigendecomposition: V = E * lambda * E^T
    eigenvals, eigenvecs = np.linalg.eigh(V)
    E = eigenvecs
    sqrt_lambda = np.diag(np.sqrt(np.maximum(eigenvals, 1e-8)))
    
    # Generate random 2D Gaussian
    e = np.random.randn(n_samples, 2)
    
    # Transform: x = m + E*sqrt(lambda)*e
    x = m + (E @ sqrt_lambda @ e.T).T
    
    logger.info(f"Generated 2D Gaussian: mean={m}, samples={n_samples}")
    logger.info(f"Variance matrix:\n{V}")
    logger.info(f"Eigenvalues: {eigenvals}")
    
    return x, m, V


def generate_gmm_2d(n_samples=1000, n_components=3):
    """Generate 2D data from Gaussian Mixture Model"""
    # Random means and covariances for each component
    np.random.seed(42)
    means = np.random.randn(n_components, 2) * 3
    covs = [np.random.randn(2, 2) for _ in range(n_components)]
    covs = [c @ c.T + 0.5 * np.eye(2) for c in covs]  # Make positive definite
    
    # Sample from each component
    samples_per_component = n_samples // n_components
    x_list = []
    for i in range(n_components):
        x_i = np.random.multivariate_normal(means[i], covs[i], samples_per_component)
        x_list.append(x_i)
    
    x = np.vstack(x_list)
    np.random.shuffle(x)
    
    logger.info(f"Generated GMM with {n_components} components, {n_samples} samples")
    logger.info(f"Component means:\n{means}")
    
    return x, means, covs


def linear_transform_2d_to_3d(x, A=None):
    """Transform 2D data to 3D: xx = A*x"""
    if A is None:
        # Create a random 3x2 transformation matrix
        np.random.seed(42)
        A = np.random.randn(3, 2) * 0.5
        A[0, :] = [1.0, 0.5]  # Make it more interesting
        A[1, :] = [0.3, 1.0]
        A[2, :] = [0.2, 0.8]
    
    xx = (A @ x.T).T
    logger.info(f"Transformed to 3D using matrix A:\n{A}")
    return xx, A


def mlp_transform_2d_to_3d(x, mlp_model):
    """Transform 2D data to 3D using MLP"""
    x_tensor = torch.FloatTensor(x)
    with torch.no_grad():
        xx_tensor = mlp_model(x_tensor)
    xx = xx_tensor.numpy()
    logger.info("Transformed to 3D using wonky MLP")
    return xx


def visualize_diffusion_steps(x_original, x_3d, diffusion_model, title_prefix, save_path):
    """Visualize all stages of diffusion process"""
    num_steps_to_show = min(10, diffusion_model.num_steps)
    step_indices = np.linspace(0, diffusion_model.num_steps - 1, num_steps_to_show, dtype=int)
    
    x_3d_tensor = torch.FloatTensor(x_3d)
    
    # IMPORTANT: Store noise for ALL steps (0 to T) for proper reverse process
    stored_noises = {}
    forward_states = {}
    
    # Forward diffusion: add noise step by step until isotropic Gaussian
    # We need to compute forward diffusion for ALL steps to get all noises
    final_step = diffusion_model.num_steps - 1
    
    # Run forward process for all steps and store noises
    for step in range(diffusion_model.num_steps):
        noise_key = f'step_{step}'
        x_t, noise = diffusion_model.forward_diffusion(x_3d_tensor, step, store_noise=True, noise_key=noise_key)
        stored_noises[step] = noise
        if step in step_indices or step == final_step:
            forward_states[step] = x_t
    
    # Final state should be isotropic Gaussian
    x_T = forward_states[final_step]
    
    # Reverse diffusion: start from isotropic Gaussian (same as encoder output) and iteratively denoise
    reverse_states = {}
    x_current = x_T.clone()  # Start from final noisy state (isotropic Gaussian)
    reverse_states[final_step] = x_current.clone()
    
    # Reverse process: go backwards from T to 0, using stored noises
    for step in reversed(range(1, diffusion_model.num_steps)):
        # To go from x_t to x_{t-1}, we need:
        # 1. Recover x_0 from x_t using noise at step t
        # 2. Compute x_{t-1} from x_0 using noise at step t-1
        noise_t = stored_noises[step]  # Noise used to create x_t from x_0
        noise_t_prev = stored_noises[step - 1]  # Noise used to create x_{t-1} from x_0
        
        # Recover x_0 from x_t
        alpha_bar_t = diffusion_model.alpha_bar[step]
        x_0 = (x_current - torch.sqrt(1 - alpha_bar_t) * noise_t) / (torch.sqrt(alpha_bar_t) + 1e-8)
        
        # Compute x_{t-1} from x_0 using noise_{t-1}
        alpha_bar_t_prev = diffusion_model.alpha_bar[step - 1]
        x_current = torch.sqrt(alpha_bar_t_prev) * x_0 + torch.sqrt(1 - alpha_bar_t_prev) * noise_t_prev
        
        # Store state if it's one we want to visualize
        if (step - 1) in step_indices or (step - 1) == 0:
            reverse_states[step - 1] = x_current.clone()
    
    # Final step (0) should be the recovered manifold
    if 0 not in reverse_states:
        reverse_states[0] = x_current.clone()
    
    fig = plt.figure(figsize=(20, 10))
    
    # Row 1: Forward diffusion (Encoder) - from manifold to isotropic Gaussian
    # Start with the 3D input manifold
    ax_start = fig.add_subplot(2, num_steps_to_show + 1, 1, projection='3d')
    ax_start.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], alpha=0.6, s=20, c='green')
    ax_start.set_title('3D Manifold\n(Start)', fontsize=10, fontweight='bold')
    ax_start.set_xlabel('x1')
    ax_start.set_ylabel('x2')
    ax_start.set_zlabel('x3')
    
    # Forward diffusion steps
    for idx, step in enumerate(step_indices):
        ax = fig.add_subplot(2, num_steps_to_show + 1, idx + 2, projection='3d')
        x_t = forward_states[step]
        x_t_np = x_t.numpy()
        
        # Color by amount of noise added
        noise_level = step / diffusion_model.num_steps
        ax.scatter(x_t_np[:, 0], x_t_np[:, 1], x_t_np[:, 2], 
                  alpha=0.6, s=20, c=x_t_np[:, 0], cmap='viridis')
        
        if step == final_step:
            ax.set_title(f'Forward Step {step}\n(Isotropic Gaussian)', fontsize=9, fontweight='bold')
        else:
            ax.set_title(f'Forward Step {step}\n(Noise: {noise_level:.2f})', fontsize=9)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
    
    # Row 2: Reverse diffusion (Decoder) - from isotropic Gaussian to recovered manifold
    # Show in reverse order: T, ..., 0
    reverse_step_indices = list(reversed(step_indices))
    # Make sure we show step T first if it's not in the list
    if final_step not in reverse_step_indices:
        reverse_step_indices = [final_step] + reverse_step_indices[:-1]
    
    for idx, step in enumerate(reverse_step_indices):
        ax = fig.add_subplot(2, num_steps_to_show + 1, num_steps_to_show + 2 + idx + 1, projection='3d')
        
        if step in reverse_states:
            x_recovered = reverse_states[step]
        elif step == final_step:
            # Use the isotropic Gaussian from forward process
            x_recovered = x_T.clone()
        else:
            # Fallback to original if not computed
            x_recovered = x_3d_tensor
        
        x_recovered_np = x_recovered.numpy()
        
        # Color by recovery quality
        ax.scatter(x_recovered_np[:, 0], x_recovered_np[:, 1], x_recovered_np[:, 2], 
                  alpha=0.6, s=20, c=x_recovered_np[:, 1], cmap='plasma')
        
        if step == final_step:
            ax.set_title(f'Reverse Step {step}\n(Start: Isotropic Gaussian)', fontsize=9, fontweight='bold')
        elif step == 0:
            ax.set_title(f'Reverse Step {step}\n(Recovered Manifold)', fontsize=9, fontweight='bold')
        else:
            ax.set_title(f'Reverse Step {step}\n(Denoising)', fontsize=9)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
    
    plt.suptitle(title_prefix, fontsize=14, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {save_path}")
    
    # Log reconstruction error (should be very small if working correctly)
    if 0 in reverse_states:
        final_recovered = reverse_states[0]
        mse = torch.mean((final_recovered - x_3d_tensor) ** 2).item()
        logger.info(f"Reconstruction MSE: {mse:.6f}")
        
        # Check if it's isotropic Gaussian at step T
        x_T_np = x_T.numpy()
        std_per_dim = x_T_np.std(axis=0)
        logger.info(f"Final step (T={final_step}) std per dimension: {std_per_dim}")
        logger.info(f"Isotropic check - std ratio max/min: {std_per_dim.max() / std_per_dim.min():.3f}")
    
    plt.close()


def run_experiment_I():
    """Part I: 2D Gaussian -> 3D linear transform -> Diffusion"""
    logger.info("="*60)
    logger.info("EXPERIMENT I: 2D Gaussian with Linear Transform")
    logger.info("="*60)
    
    # a) Generate 2D Gaussian
    x_2d, m, V = generate_2d_gaussian(n_samples=1000)
    
    # b) Transform to 3D
    x_3d, A = linear_transform_2d_to_3d(x_2d)
    
    # c) Run diffusion
    diffusion = SimpleDiffusion(num_steps=1000)
    
    visualize_diffusion_steps(
        x_2d, x_3d, diffusion,
        "Experiment I: 2D Gaussian → Linear 3D → Diffusion",
        output_dir / "experiment_I_diffusion.png"
    )
    
    # Log statistics
    stats = {
        "experiment": "I",
        "2d_mean": m.tolist(),
        "2d_variance": V.tolist(),
        "3d_transform": A.tolist(),
        "3d_mean": x_3d.mean(axis=0).tolist(),
        "3d_std": x_3d.std(axis=0).tolist(),
        "n_samples": len(x_2d)
    }
    
    return stats


def run_experiment_II():
    """Part II: 2D GMM -> 3D linear transform -> Diffusion"""
    logger.info("="*60)
    logger.info("EXPERIMENT II: 2D GMM with Linear Transform")
    logger.info("="*60)
    
    # Generate 2D GMM
    x_2d, means, covs = generate_gmm_2d(n_samples=1000, n_components=3)
    
    # Transform to 3D
    x_3d, A = linear_transform_2d_to_3d(x_2d)
    
    # Run diffusion
    diffusion = SimpleDiffusion(num_steps=1000)
    
    visualize_diffusion_steps(
        x_2d, x_3d, diffusion,
        "Experiment II: 2D GMM → Linear 3D → Diffusion",
        output_dir / "experiment_II_diffusion.png"
    )
    
    stats = {
        "experiment": "II",
        "gmm_means": [m.tolist() for m in means],
        "3d_transform": A.tolist(),
        "3d_mean": x_3d.mean(axis=0).tolist(),
        "3d_std": x_3d.std(axis=0).tolist(),
        "n_samples": len(x_2d)
    }
    
    return stats


def run_experiment_III():
    """Part III: 2D Gaussian -> 3D MLP transform -> Diffusion"""
    logger.info("="*60)
    logger.info("EXPERIMENT III: 2D Gaussian with MLP Transform")
    logger.info("="*60)
    
    # Generate 2D Gaussian
    x_2d, m, V = generate_2d_gaussian(n_samples=1000)
    
    # Transform to 3D using MLP
    mlp = WonkyMLP(input_dim=2, output_dim=3)
    mlp.eval()
    x_3d = mlp_transform_2d_to_3d(x_2d, mlp)
    
    # Run diffusion
    diffusion = SimpleDiffusion(num_steps=1000)
    
    visualize_diffusion_steps(
        x_2d, x_3d, diffusion,
        "Experiment III: 2D Gaussian → MLP 3D → Diffusion",
        output_dir / "experiment_III_diffusion.png"
    )
    
    stats = {
        "experiment": "III",
        "2d_mean": m.tolist(),
        "2d_variance": V.tolist(),
        "3d_mean": x_3d.mean(axis=0).tolist(),
        "3d_std": x_3d.std(axis=0).tolist(),
        "n_samples": len(x_2d),
        "transform": "MLP"
    }
    
    return stats


def run_experiment_IV():
    """Part IV: 2D GMM -> 3D MLP transform -> Diffusion"""
    logger.info("="*60)
    logger.info("EXPERIMENT IV: 2D GMM with MLP Transform")
    logger.info("="*60)
    
    # Generate 2D GMM
    x_2d, means, covs = generate_gmm_2d(n_samples=1000, n_components=3)
    
    # Transform to 3D using MLP
    mlp = WonkyMLP(input_dim=2, output_dim=3)
    mlp.eval()
    x_3d = mlp_transform_2d_to_3d(x_2d, mlp)
    
    # Run diffusion
    diffusion = SimpleDiffusion(num_steps=1000)
    
    visualize_diffusion_steps(
        x_2d, x_3d, diffusion,
        "Experiment IV: 2D GMM → MLP 3D → Diffusion",
        output_dir / "experiment_IV_diffusion.png"
    )
    
    stats = {
        "experiment": "IV",
        "gmm_means": [m.tolist() for m in means],
        "3d_mean": x_3d.mean(axis=0).tolist(),
        "3d_std": x_3d.std(axis=0).tolist(),
        "n_samples": len(x_2d),
        "transform": "MLP"
    }
    
    return stats


def main():
    """Run all experiments"""
    logger.info("Starting Diffusion Model Experiments")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    all_stats = {}
    
    # Run all experiments
    all_stats['I'] = run_experiment_I()
    all_stats['II'] = run_experiment_II()
    all_stats['III'] = run_experiment_III()
    all_stats['IV'] = run_experiment_IV()
    
    # Save statistics
    stats_file = output_dir / "experiment_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    logger.info("="*60)
    logger.info("All experiments completed!")
    logger.info(f"Visualizations saved to: {output_dir.absolute()}")
    logger.info(f"Statistics saved to: {stats_file.absolute()}")


if __name__ == "__main__":
    main()

