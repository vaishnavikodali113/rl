from __future__ import annotations

import torch
import torch.nn as nn


class SimNorm(nn.Module):
    """Simplex normalization over latent groups with remainder support."""

    def __init__(self, simnorm_dim: int = 8, feature_dim: int | None = None):
        super().__init__()
        self.dim = simnorm_dim
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_dim is not None and x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Latent dimension {x.shape[-1]} must match feature_dim={self.feature_dim}."
            )
        if self.dim <= 0:
            raise ValueError(f"simnorm_dim must be positive, got {self.dim}.")

        feature_dim = x.shape[-1]
        full = feature_dim // self.dim
        rem = feature_dim % self.dim
        if rem == 0:
            shape = x.shape
            x = x.view(*shape[:-1], full, self.dim)
            x = x.softmax(dim=-1)
            return x.view(*shape)

        main = x[..., : full * self.dim].reshape(*x.shape[:-1], full, self.dim).softmax(dim=-1)
        remainder = x[..., full * self.dim :].softmax(dim=-1)
        return torch.cat([main.reshape(*x.shape[:-1], full * self.dim), remainder], dim=-1)

    def extra_repr(self) -> str:
        return f"simnorm_dim={self.dim}, feature_dim={self.feature_dim}"
