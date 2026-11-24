"""
VAE latent spaces supported by Steppe upscalers: the latent layout handed over by ComfyUI / diffusers (channels,
2x2 packing), per channel normalization statistics, frame rules and the models that share the VAE. Upscalers work
on normalized, unpacked latents ("model space"); `normalize` / `denormalize` convert between the two.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentSpace:
    """
    channels: channels of the latents as handed over (after packing).
    patch_size: 2x2 packing of the handed over latents, channel order (C, 2, 2) as in ComfyUI and diffusers; 1 = none.
    mean, std: per channel statistics of the unpacked latents, empty = handed over latents are already normalized.
    video: the VAE encodes videos (a temporal axis exists).
    compatible_models: models sharing this VAE, for error messages and tooltips.
    """

    name: str
    display_name: str
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    temporal_compression: int
    patch_size: int = 1
    video: bool = True
    compatible_models: str = ""

    @property
    def model_channels(self) -> int:
        return self.channels // (self.patch_size * self.patch_size)

    def statistics(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
            mean, std: [1, C, 1, 1, 1], dtype: torch.float32
        """
        mean = torch.tensor(self.mean, device=device)[None, :, None, None, None]
        std = torch.tensor(self.std, device=device)[None, :, None, None, None]
        return mean, std

    def unpack(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C * p * p, T, H, W], channel order (C, p, p)
        returns:
            latents: [B, C, T, H * p, W * p]
        """
        patch = self.patch_size
        if patch == 1:
            return latents
        batch_size, channels, frames, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // (patch * patch), patch, patch, frames, height, width)
        latents = latents.permute(0, 1, 4, 5, 2, 6, 3)
        return latents.reshape(batch_size, channels // (patch * patch), frames, height * patch, width * patch)

    def pack(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], H and W multiples of the patch size
        returns:
            latents: [B, C * p * p, T, H / p, W / p], channel order (C, p, p)
        """
        patch = self.patch_size
        if patch == 1:
            return latents
        batch_size, channels, frames, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, frames, height // patch, patch, width // patch, patch)
        latents = latents.permute(0, 1, 4, 6, 2, 3, 5)
        return latents.reshape(batch_size, channels * patch * patch, frames, height // patch, width // patch)

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, T, H, W], VAE latents as handed over
        returns:
            latents: [B, C_model, T, H * p, W * p], normalized model space, dtype: torch.float32
        """
        latents = self.unpack(latents.float())
        if not self.mean:
            return latents
        mean, std = self.statistics(latents.device)
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C_model, T, H, W], normalized model space
        returns:
            latents: [B, C, T, H / p, W / p], VAE latents as handed over, dtype: torch.float32
        """
        latents = latents.float()
        if self.mean:
            mean, std = self.statistics(latents.device)
            latents = latents * std + mean
        return self.pack(latents)


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
    compatible_models=(
        "Wan 2.1, Wan 2.2 14B, Qwen-Image, Krea 2 and other models using this VAE; "
        "Wan 2.2 5B uses a different VAE (48 channels)"
    ),
)

# ComfyUI and the diffusers FLUX.2 pipelines hand over the VAE latents 2x2 packed (128 channels, 1/16 of the image)
# and batch norm normalized with the VAE's own running statistics, the space the models were trained in.
FLUX_2 = LatentSpace(
    name="flux_2",
    display_name="FLUX.2 VAE",
    channels=128,
    mean=(),
    std=(),
    temporal_compression=1,
    patch_size=2,
    video=False,
    compatible_models="FLUX.2 [dev], FLUX.2 [klein], MOSS-Image and other models using this VAE; not FLUX.1",
)

LATENT_SPACES: dict[str, LatentSpace] = {space.name: space for space in (WAN_2_1, FLUX_2)}


def get_latent_space(name: str) -> LatentSpace:
    if name not in LATENT_SPACES:
        raise ValueError(f"unknown latent space '{name}', supported: {', '.join(LATENT_SPACES)}")
    return LATENT_SPACES[name]
