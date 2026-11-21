"""
VAE latent spaces supported by Steppe upscalers: channel count, per channel normalization statistics and
frame rules. Upscalers work on normalized latents; ComfyUI and diffusers hand over raw VAE latents.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentSpace:
    name: str
    display_name: str
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    temporal_compression: int

    def statistics(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            mean, std: [1, C, 1, 1, 1], dtype: torch.float32
        """
        mean = torch.tensor(self.mean, device=device)[None, :, None, None, None]
        std = torch.tensor(self.std, device=device)[None, :, None, None, None]
        return mean, std

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], raw VAE latents
        returns:
            latents: [B, C, T, H, W], normalized, dtype: torch.float32
        """
        mean, std = self.statistics(latents.device)
        return (latents.float() - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], normalized
        returns:
            latents: [B, C, T, H, W], raw VAE latents, dtype: torch.float32
        """
        mean, std = self.statistics(latents.device)
        return latents.float() * std + mean


WAN_2_1 = LatentSpace(
    name="wan_2_1",
    display_name="Wan 2.1 VAE",
    channels=16,
    mean=(
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ),
    std=(
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ),
    temporal_compression=4,
)

LATENT_SPACES: dict[str, LatentSpace] = {space.name: space for space in (WAN_2_1,)}


def get_latent_space(name: str) -> LatentSpace:
    if name not in LATENT_SPACES:
        raise ValueError(f"unknown latent space '{name}', supported: {', '.join(LATENT_SPACES)}")
    return LATENT_SPACES[name]
