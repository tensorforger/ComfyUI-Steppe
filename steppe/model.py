"""
Steppe model files and the upscaler object: a safetensors file whose metadata names the latent space, the model
kind (multi step flow or single step student) and the network configuration. Development checkpoints of the
STDev research code (metadata keys `vae`, `architecture`, `model_config`) are read as well.
"""

import json
import os
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from . import catalog
from .latent_spaces import LatentSpace, get_latent_space
from .sampling import UpscaleSettings, upscale_latents
from .wan_transformer import SteppeWanTransformer, WanTransformerConfig

FORMAT_KEY = "steppe.format"
FORMAT_VERSION = "1"
KIND_FLOW = "flow"
KIND_SINGLE_STEP = "single_step"
DEVELOPMENT_KINDS = {"WanUpscalerTransformer": KIND_FLOW, "WanStudentTransformer": KIND_SINGLE_STEP}


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
    if metadata.get("vae") == "wan_2_1" and "model_config" in metadata:
        architecture = metadata.get("architecture", "WanUpscalerTransformer")
        if architecture not in DEVELOPMENT_KINDS:
            return None
        return ModelInfo(
            name=name,
            latent_space="wan_2_1",
            kind=DEVELOPMENT_KINDS[architecture],
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
    A loaded Steppe model: 2x spatial and / or 2x temporal upscaling of raw VAE latents.
    """

    def __init__(self, model: SteppeWanTransformer, info: ModelInfo) -> None:
        self.model = model
        self.info = info
        self.latent_space: LatentSpace = get_latent_space(info.latent_space)

    @property
    def is_single_step(self) -> bool:
        return self.info.kind == KIND_SINGLE_STEP

    def upscale(
        self,
        latents: torch.Tensor,
        settings: UpscaleSettings = UpscaleSettings(),
        device: torch.device | str | None = None,
        on_pass=lambda: None,
    ) -> torch.Tensor:
        """
        Runs on `device` (default: the device of the model weights); the model is not moved.

        args:
            latents: [B, C, T, H, W] or [B, C, H, W], raw VAE latents of the model's latent space
        returns:
            latents: [B, C, T_hr, H_hr, W_hr] or [B, C, H_hr, W_hr], raw VAE latents, dtype: torch.float32,
                on the CPU
        """
        device = torch.device(device) if device is not None else self.model.input_conv.weight.device
        image_layout = latents.ndim == 4
        if image_layout:
            latents = latents[:, :, None]
        upscaled = upscale_latents(self.model, self.latent_space, latents, settings, device, on_pass)
        return upscaled[:, :, 0] if image_layout else upscaled


def build_model(info: ModelInfo) -> SteppeWanTransformer:
    if info.latent_space != "wan_2_1":
        raise ValueError(f"no network for latent space '{info.latent_space}'")
    config = WanTransformerConfig(**info.config)
    return SteppeWanTransformer(config, guided=info.kind == KIND_SINGLE_STEP)


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
