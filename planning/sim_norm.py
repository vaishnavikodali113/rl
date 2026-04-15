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

        # Keep a robust fallback for non-divisible latent sizes by normalizing
        # fixed-size groups and a final smaller remainder group.
        out = torch.empty_like(x)
        offset = 0
        for _ in range(full):
            out[..., offset : offset + self.dim] = x[..., offset : offset + self.dim].softmax(dim=-1)
            offset += self.dim
        if rem > 0:
            out[..., offset:] = x[..., offset:].softmax(dim=-1)
        return out

    def extra_repr(self) -> str:
        return f"simnorm_dim={self.dim}, feature_dim={self.feature_dim}"
