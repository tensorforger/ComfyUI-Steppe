"""
Steppe: generative latent upscalers. 2x spatial and / or 2x temporal upscaling of VAE latents with a small
generative model; the output is decoded directly by the VAE, no extra denoising pass of the base model.
Latent spaces: Wan 2.1 VAE (images and videos) and FLUX.2 VAE (images), one model family each.

    from steppe import UpscaleSettings, load_upscaler

    upscaler = load_upscaler("steppe_wan_2_1_200M_1step.safetensors")
    upscaler.model.to("cuda")
    upscaled = upscaler.upscale(raw_wan_latents, UpscaleSettings(spatial=True, hff=0.9, guidance=1.0))

    flux_upscaler = load_upscaler("steppe_flux_2_200M_1step.safetensors")  # packed, normalized FLUX.2 latents
"""

from .catalog import CATALOG, CatalogEntry
from .hff import estimate_hff
from .latent_spaces import LATENT_SPACES, LatentSpace
from .model import ModelInfo, SteppeUpscaler, load_upscaler, load_upscaler_file
from .sampling import UpscaleSettings

__version__ = "1.1.0"

__all__ = [
    "CATALOG",
    "LATENT_SPACES",
    "CatalogEntry",
    "LatentSpace",
    "ModelInfo",
    "SteppeUpscaler",
    "UpscaleSettings",
    "estimate_hff",
    "load_upscaler",
    "load_upscaler_file",
]
