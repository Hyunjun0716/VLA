"""
Subgoal Diffuser Module for TinyVLA.

Generates future latent subgoals g_t ≈ z_{t+Δ} from current latent z_t
using a lightweight MLP-based DDIM diffusion process.

Usage:
    extractor = GlobalLatentExtractor()
    diffuser = SubgoalDiffuserMLP(latent_dim=2048)

    # Extract z from VLM hidden states
    z_t = extractor(hidden_states)           # (B, 2048)

    # Training: compute loss from (z_t, z_future) pair
    loss = diffuser.compute_loss(z_t, z_future)

    # Inference: sample subgoal
    g_t = diffuser.sample(z_t)              # (B, 2048)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GlobalLatentExtractor(nn.Module):
    """
    Extracts a global latent vector from VLM hidden states via mean pooling + L2 norm.

    Input:  hidden_states (B, T, D)
    Output: z (B, D)  with ||z|| = 1
    """

    def forward(self, hidden_states):
        z = hidden_states.mean(dim=1)       # (B, D)
        return F.normalize(z, dim=-1)       # L2 norm


class SubgoalDiffuserMLP(nn.Module):
    """
    MLP-based latent diffusion denoiser for subgoal prediction.

    Given current latent z_t, generates a future latent g_t ≈ z_{t+Δ}
    using DDIM denoising in the VLM latent space.

    Args:
        latent_dim: Dimension of latent vectors (default: 2048 for Pythia-1.3B)
        time_emb_dim: Dimension of timestep embedding (default: 256)
        hidden_dim: Hidden dimension of MLP (default: 4096)
        num_train_timesteps: Number of diffusion training timesteps (default: 100)
    """

    def __init__(self, latent_dim=2048, time_emb_dim=256, hidden_dim=4096,
                 num_train_timesteps=100):
        super().__init__()
        self.latent_dim = latent_dim

        # Timestep encoder
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Denoiser MLP with residual: [x_k, z_t, time_emb] → eps_pred
        input_dim = latent_dim * 2 + time_emb_dim  # 2048*2 + 256 = 4352
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(4)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # DDIM noise scheduler (independent from action diffuser)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon',
        )

    def forward(self, x_k, timestep, z_t):
        """
        Predict noise from noisy latent.

        Args:
            x_k: Noisy latent at timestep k, (B, D)
            timestep: Diffusion timestep, (B,) or scalar
            z_t: Conditioning current latent, (B, D)

        Returns:
            eps_pred: Predicted noise, (B, D)
        """
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x_k.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(x_k.shape[0])

        time_emb = self.time_encoder(timestep)                  # (B, time_emb_dim)
        inp = torch.cat([x_k, z_t, time_emb], dim=-1)          # (B, 2D + time_emb_dim)
        h = self.input_proj(inp)                                # (B, hidden_dim)
        for block in self.res_blocks:
            h = h + block(h)                                    # residual connection
        return self.output_proj(h)                              # (B, D)

    def compute_loss(self, z_t, z_future):
        """
        Compute training loss: MSE between predicted and actual noise.

        Args:
            z_t: Current latent, (B, D)
            z_future: Future latent (target), (B, D)

        Returns:
            loss: Scalar MSE loss
        """
        B = z_t.shape[0]
        noise = torch.randn_like(z_future)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=z_t.device
        ).long()
        noisy = self.noise_scheduler.add_noise(z_future, noise, timesteps)
        noise_pred = self.forward(noisy, timesteps, z_t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(self, z_t, num_inference_steps=10):
        """
        Generate subgoal g_t via DDIM denoising.

        Args:
            z_t: Current latent, (B, D)
            num_inference_steps: Number of DDIM denoising steps

        Returns:
            g_t: Predicted future latent (L2-normalized), (B, D)
        """
        self.noise_scheduler.set_timesteps(num_inference_steps)
        x = torch.randn_like(z_t)
        for k in self.noise_scheduler.timesteps:
            k = k.to(z_t.device)
            noise_pred = self.forward(x, k, z_t)
            x = self.noise_scheduler.step(noise_pred, k, x).prev_sample
        return F.normalize(x, dim=-1)
