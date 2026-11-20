"""Noise schedules and distribution helpers for diffusion models."""

import math

import torch


def cosine_beta_schedule(num_steps, s=0.008):
    """Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models".

    Produces monotonically decreasing betas that avoid the sharp truncation
    of a linear schedule.

    Args:
        num_steps: Number of diffusion timesteps.
        s: Small offset controlling how close to zero the initial noise is.

    Returns:
        betas: (num_steps,) tensor of noise variances.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = ((steps / num_steps + s) / (1 + s)).cos().pow(2).clamp(min=1e-9)
    alphas = f[1:] / f[:-1]
    betas = (1 - alphas).clamp(max=0.999)
    return betas.float()


def linear_beta_schedule(num_steps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from "Denoising Diffusion Probabilistic Models".

    Args:
        num_steps: Number of diffusion timesteps.
        beta_start: Small initial variance.
        beta_end: Final variance.

    Returns:
        betas: (num_steps,) tensor of noise variances.
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def alpha_parameters(betas):
    """Convert betas to alpha-bar parameters used for forward diffusion.

    Args:
        betas: (T,) tensor of noise variances (must be within [0, 1]).

    Returns:
        alphas: cumulative product of (1 - betas).
        sqrt_alphas: square root of alphas, used for noising.
        sqrt_one_minus_alphas: square root of 1 - alphas, used for noising.
    """
    alphas = torch.cumprod(1 - betas, dim=0)
    return {
        "alphas": alphas,
        "sqrt_alphas": alphas.sqrt(),
        "sqrt_one_minus_alphas": (1 - alphas).sqrt(),
    }