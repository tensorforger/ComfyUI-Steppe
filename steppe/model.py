"""
Steppe model files and the upscaler object: a safetensors file whose metadata names the latent space, the model
kind (multi step flow or single step student) and the network configuration. Development checkpoints of the
STDev research code (metadata keys `vae`, `architecture`, `model_config`) are read as well.
"""

import json
import os
from dataclasses import dataclass, replace

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from . import catalog
from .flux_transformer import FluxTransformerConfig, SteppeFluxTransformer
from .latent_spaces import LatentSpace, get_latent_space
from .sampling import UpscalerNetwork, UpscaleSettings, upscale_latents
from .wan_transformer import SteppeWanTransformer, WanTransformerConfig

FORMAT_KEY = "steppe.format"
FORMAT_VERSION = "1"
KIND_FLOW = "flow"
KIND_SINGLE_STEP = "single_step"
# STDev architecture name -> (latent space, kind); Wan files without `architecture` are flow models
DEVELOPMENT_ARCHITECTURES = {
    "WanUpscalerTransformer": ("wan_2_1", KIND_FLOW),
    "WanStudentTransformer": ("wan_2_1", KIND_SINGLE_STEP),
    "FluxUpscalerTransformer": ("flux_2", KIND_FLOW),
    "FluxStudentTransformer": ("flux_2", KIND_SINGLE_STEP),
}
DEFAULT_DEVELOPMENT_ARCHITECTURE = {"wan_2_1": "WanUpscalerTransformer"}
# recommended (steps, shift) of the multi step flow models per latent space
RECOMMENDED_SAMPLERS = {"wan_2_1": (20, 4.0), "flux_2": (30, 2.0)}


@dataclass(frozen=True)
class ModelInfo:
    name: str
    latent_space: str
    kind: str
    version: str | None
    config: dict


def read_metadata(path: str) -> dict[str, str]:
    with safe_open(path, framework="pt") as file:
        return dict(file.metadata() or {})


def parse_model_info(path: str, metadata: dict[str, str]) -> ModelInfo | None:
    """
    returns:
        model description from the file metadata, None if the file is not a Steppe model
    """
    name = os.path.splitext(os.path.basename(path))[0]
    if FORMAT_KEY in metadata:
        return ModelInfo(
            name=metadata.get("steppe.name", name),
            latent_space=metadata["steppe.latent_space"],
            kind=metadata["steppe.kind"],
            version=metadata.get("steppe.version"),
            config=json.loads(metadata["steppe.model_config"]),
        )
    vae = metadata.get("vae")
    if "model_config" in metadata:
        architecture = metadata.get("architecture", DEFAULT_DEVELOPMENT_ARCHITECTURE.get(vae))
        if architecture not in DEVELOPMENT_ARCHITECTURES:
            return None
        latent_space, kind = DEVELOPMENT_ARCHITECTURES[architecture]
        if vae != latent_space:
            return None
        return ModelInfo(
            name=name,
            latent_space=latent_space,
            kind=kind,
            version=None,
            config=json.loads(metadata["model_config"]),
        )
    return None


def make_release_metadata(info: ModelInfo, title: str, description: str) -> dict[str, str]:
    """
    Metadata of a released model file: Steppe keys plus the common `modelspec` keys shown by model managers.
    """
    return {
        FORMAT_KEY: FORMAT_VERSION,
        "steppe.name": info.name,
        "steppe.latent_space": info.latent_space,
        "steppe.kind": info.kind,
        "steppe.version": info.version or "1.0",
        "steppe.model_config": json.dumps(info.config),
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": f"steppe-{info.latent_space.replace('_', '-')}/latent-upscaler",
        "modelspec.implementation": "https://github.com/TensorForger/ComfyUI-Steppe",
        "modelspec.title": title,
        "modelspec.description": description,
        "modelspec.author": "TensorForger",
        "modelspec.license": "MIT",
    }


class SteppeUpscaler:
    """
    A loaded Steppe model: 2x spatial and / or 2x temporal upscaling of VAE latents.
    """

    def __init__(self, model: UpscalerNetwork, info: ModelInfo) -> None:
        self.model = model
        self.info = info
        self.latent_space: LatentSpace = get_latent_space(info.latent_space)

    @property
    def is_single_step(self) -> bool:
        return self.info.kind == KIND_SINGLE_STEP

    @property
    def supports_temporal(self) -> bool:
        return self.model.supports_temporal

    def resolve_settings(self, settings: UpscaleSettings) -> UpscaleSettings:
        """
        returns:
            settings with steps / shift None replaced by the recommended values of the model and temporal off for
            spatial only models
        """
        steps, shift = RECOMMENDED_SAMPLERS[self.info.latent_space]
        return replace(
            settings,
            steps=steps if settings.steps is None else settings.steps,
            shift=shift if settings.shift is None else settings.shift,
            temporal=settings.temporal and self.supports_temporal,
        )

    def upscale(
        self,
        latents: torch.Tensor,
        settings: UpscaleSettings = UpscaleSettings(),
        device: torch.device | str | None = None,
        on_pass=lambda: None,
    ) -> torch.Tensor:
        """
        Runs on `device` (default: the device of the model weights); the model is not moved. Temporal upscaling is
        ignored by spatial only models.

        args:
            latents: [B, C, T, H, W] or [B, C, H, W], VAE latents of the model's latent space as ComfyUI and
                diffusers hand them over (Wan 2.1: raw; FLUX.2: 2x2 packed and batch norm normalized)
        returns:
            latents: [B, C, T_hr, H_hr, W_hr] or [B, C, H_hr, W_hr], same layout, dtype: torch.float32, on the CPU
        """
        device = torch.device(device) if device is not None else self.model.input_conv.weight.device
        image_layout = latents.ndim == 4
        if image_layout:
            latents = latents[:, :, None]
        settings = self.resolve_settings(settings)
        upscaled = upscale_latents(self.model, self.latent_space, latents, settings, device, on_pass)
        return upscaled[:, :, 0] if image_layout else upscaled


def build_model(info: ModelInfo) -> UpscalerNetwork:
    guided = info.kind == KIND_SINGLE_STEP
    if info.latent_space == "wan_2_1":
        return SteppeWanTransformer(WanTransformerConfig(**info.config), guided=guided)
    if info.latent_space == "flux_2":
        return SteppeFluxTransformer(FluxTransformerConfig(**info.config), guided=guided)
    raise ValueError(f"no network for latent space '{info.latent_space}'")


def load_upscaler_file(path: str, dtype: torch.dtype = torch.bfloat16) -> SteppeUpscaler:
    """
    returns:
        upscaler on the CPU in `dtype`, evaluation mode
    """
    info = parse_model_info(path, read_metadata(path))
    if info is None:
        raise ValueError(f"{path} is not a Steppe model file (no Steppe metadata)")
    model = build_model(info)
    model.load_state_dict(load_file(path))
    model = model.to(dtype).eval().requires_grad_(False)
    return SteppeUpscaler(model, info)


def load_upscaler(name_or_path: str, directory: str | None = None, dtype: torch.dtype = torch.bfloat16) -> SteppeUpscaler:
    """
    Loads a local model file, or a released model by file name (downloaded from Hugging Face into `directory`,
    default: the huggingface_hub cache).
    """
    if os.path.isfile(name_or_path):
        return load_upscaler_file(name_or_path, dtype)
    entry = catalog.CATALOG_BY_FILE_NAME.get(name_or_path) or catalog.CATALOG_BY_FILE_NAME.get(
        f"{name_or_path}.safetensors"
    )
    if entry is None:
        known = ", ".join(item.file_name for item in catalog.CATALOG)
        raise FileNotFoundError(f"{name_or_path} is neither a file nor a released model ({known})")
    if directory is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=entry.repository, filename=entry.file_name)
    else:
        path = os.path.join(directory, entry.file_name)
        if not os.path.isfile(path):
            path = catalog.download_model(entry, directory)
    upscaler = load_upscaler_file(path, dtype)
    catalog.report_updates(entry.repository, entry.file_name, upscaler.info.version)
    return upscaler


__all__ = ["ModelInfo", "SteppeUpscaler", "load_upscaler", "load_upscaler_file", "make_release_metadata"]
