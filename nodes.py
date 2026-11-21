"""
ComfyUI nodes of Steppe generative latent upscalers. Models live in models/latent_upscale_models; released models
are listed even when missing and are downloaded from Hugging Face on first use.
"""

import logging
import os
from dataclasses import replace

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from .steppe import catalog
from .steppe.hff import compute_raw_immerkaer, quantile_normalize
from .steppe.model import SteppeUpscaler, load_upscaler_file, parse_model_info, read_metadata
from .steppe.sampling import UpscaleSettings, count_model_passes, count_temporal_windows, target_size_of

logger = logging.getLogger("steppe")

MODEL_FOLDER_NAME = "latent_upscale_models"
CATEGORY = "Steppe"
UPSCALER_TYPE = "STEPPE_UPSCALER"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
BYTES_PER_TOKEN_FACTOR = 10
CHUNK_BYTES_FACTOR = 6

HFF_TOOLTIP = (
    "High frequency feature: target sharpness / fine detail of the upscaled result, [0...1] "
    "(Immerkaer estimate, ReLAION calibrated). -1 = no hff, the model decides itself. "
    "Use 'Steppe Measure HFF' to take it from a reference image."
)
GUIDANCE_TOOLTIP = (
    "1-step models: distilled guidance (trained on 1-3, no extra cost); 1 = off (default, safe and stable), 2 also works well. "
    "Base model: classifier free guidance between hff and negative hff (null by default); "
    "the hff effect is subtle, so larger values (4-8) are needed for a visible change. 1 = off (2x faster)."
)


def register_model_folder() -> None:
    if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
            [os.path.join(folder_paths.models_dir, MODEL_FOLDER_NAME)],
            {".safetensors"},
        )


def model_directory() -> str:
    return folder_paths.get_folder_paths(MODEL_FOLDER_NAME)[0]


def list_models() -> list[str]:
    """
    returns:
        released models (present or downloadable) first, then other local Steppe model files
    """
    register_model_folder()
    local = []
    for file_name in folder_paths.get_filename_list(MODEL_FOLDER_NAME):
        if file_name in catalog.CATALOG_BY_FILE_NAME:
            continue
        try:
            path = folder_paths.get_full_path(MODEL_FOLDER_NAME, file_name)
            if parse_model_info(path, read_metadata(path)) is not None:
                local.append(file_name)
        except Exception:
            continue
    return [entry.file_name for entry in catalog.CATALOG] + local


def resolve_model_path(model_name: str) -> str:
    register_model_folder()
    path = folder_paths.get_full_path(MODEL_FOLDER_NAME, model_name)
    if path is not None:
        return path
    entry = catalog.CATALOG_BY_FILE_NAME.get(model_name)
    if entry is None:
        raise FileNotFoundError(f"{model_name} not found in models/{MODEL_FOLDER_NAME}")
    return catalog.download_model(entry, model_directory())


class SteppeModel:
    """
    Loaded upscaler plus the ComfyUI model patcher that moves it between devices under memory management.
    """

    def __init__(self, upscaler: SteppeUpscaler) -> None:
        self.upscaler = upscaler
        self.patcher = comfy.model_patcher.ModelPatcher(
            upscaler.model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name != "auto":
        return DTYPES[dtype_name]
    device = comfy.model_management.get_torch_device()
    try:
        use_bf16 = comfy.model_management.should_use_bf16(device)
    except Exception:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if use_bf16 else torch.float32


class SteppeLoadUpscaler:
    CATEGORY = CATEGORY
    RETURN_TYPES = (UPSCALER_TYPE,)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "load"
    DESCRIPTION = "Loads a Steppe latent upscaler. Released models are downloaded from Hugging Face on first use."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        descriptions = "; ".join(f"{entry.file_name}: {entry.description}" for entry in catalog.CATALOG)
        return {
            "required": {
                "model_name": (list_models(), {"tooltip": f"Missing released models are downloaded. {descriptions}"}),
                "dtype": (["auto", *DTYPES], {"default": "auto", "tooltip": "auto = bfloat16 where supported."}),
            }
        }

    def load(self, model_name: str, dtype: str) -> tuple[SteppeModel]:
        path = resolve_model_path(model_name)
        upscaler = load_upscaler_file(path, resolve_dtype(dtype))
        entry = catalog.CATALOG_BY_FILE_NAME.get(model_name)
        if entry is not None:
            catalog.report_updates(entry.repository, entry.file_name, upscaler.info.version)
        return (SteppeModel(upscaler),)


def estimate_memory(upscaler: SteppeUpscaler, samples_shape: tuple[int, ...], settings: UpscaleSettings) -> float:
    """
    returns:
        rough peak activation memory in bytes of one sample and temporal window
    """
    source_frames, height, width = samples_shape[2:]
    if settings.temporal_window > 0:
        source_frames = min(source_frames, settings.temporal_window + settings.temporal_context)
    frames, target_height, target_width = target_size_of((source_frames, height, width), settings.spatial, settings.temporal)
    config = upscaler.model.config
    element_size = torch.finfo(upscaler.model.dtype).bits // 8
    tokens = frames * target_height * target_width
    chunk_tokens = max(target_height * target_width, 65536)
    return element_size * (
        BYTES_PER_TOKEN_FACTOR * tokens * config.width + CHUNK_BYTES_FACTOR * chunk_tokens * config.mlp_hidden
    )


def run_upscale(model: SteppeModel, latent: dict, settings: UpscaleSettings) -> tuple[dict]:
    upscaler = model.upscaler
    samples = latent["samples"]
    space = upscaler.latent_space
    if samples.ndim not in (4, 5) or samples.shape[1] != space.channels:
        raise ValueError(
            f"{upscaler.info.name} upscales {space.display_name} latents ({space.channels} channels: Wan 2.1, "
            f"Wan 2.2 14B, Qwen-Image and other models using this VAE), got shape {tuple(samples.shape)}. "
            "Wan 2.2 5B uses a different VAE (48 channels) and is not supported."
        )
    frames = samples.shape[2] if samples.ndim == 5 else 1
    temporal = settings.temporal and frames > 1
    if not settings.spatial and not temporal:
        logger.warning("Steppe: nothing to upscale (spatial off, temporal off or single frame)")
        return (latent,)
    settings = replace(settings, temporal=temporal)
    layout_shape = samples.shape if samples.ndim == 5 else (samples.shape[0], samples.shape[1], 1, *samples.shape[2:])
    memory_required = estimate_memory(upscaler, tuple(layout_shape), settings)
    try:
        comfy.model_management.load_models_gpu([model.patcher], memory_required=memory_required, force_full_load=True)
    except TypeError:
        comfy.model_management.load_models_gpu([model.patcher], memory_required=memory_required)
    device = model.patcher.load_device
    total = samples.shape[0] * count_temporal_windows(frames, settings) * count_model_passes(upscaler.model, settings)
    progress_bar = comfy.utils.ProgressBar(total)

    def on_pass() -> None:
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress_bar.update(1)

    upscaled = upscaler.upscale(samples, settings, device, on_pass)
    output = {key: value for key, value in latent.items() if key != "noise_mask"}
    output["samples"] = upscaled.to(comfy.model_management.intermediate_device())
    return (output,)


def optional_hff(value: float) -> float | None:
    return None if value < 0 else value


def common_inputs() -> dict:
    return {
        "upscaler": (UPSCALER_TYPE,),
        "latent": ("LATENT",),
        "spatial": ("BOOLEAN", {"default": True, "tooltip": "2x height and width."}),
        "temporal": (
            "BOOLEAN",
            {"default": False, "tooltip": "2x frame rate of videos: T -> 1 + 2 (T - 1) latent frames (4T - 3 -> 8T - 7 pixel frames). Ignored for images."},
        ),
        "hff": ("FLOAT", {"default": 0.9, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": HFF_TOOLTIP}),
        "guidance": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": GUIDANCE_TOOLTIP}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "Base model only; 1-step models ignore it."}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
    }


class SteppeUpscaleLatent:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    DESCRIPTION = (
        "2x latent upscaling (spatial) and / or 2x frame rate (temporal) with a Steppe model. "
        "The result is decoded directly with the VAE, no extra sampling pass needed."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": common_inputs()}

    def upscale(
        self,
        upscaler: SteppeModel,
        latent: dict,
        spatial: bool,
        temporal: bool,
        hff: float,
        guidance: float,
        steps: int,
        seed: int,
    ) -> tuple[dict]:
        settings = UpscaleSettings(
            spatial=spatial, temporal=temporal, hff=optional_hff(hff), guidance=guidance, steps=steps, seed=seed
        )
        return run_upscale(upscaler, latent, settings)


class SteppeUpscaleLatentAdvanced:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    DESCRIPTION = "Steppe latent upscaling with sampler settings and temporal windows for long videos."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        inputs = common_inputs()
        inputs.update(
            {
                "negative_hff": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Base model: hff of the negative guidance branch. -1 = no hff (null)."},
                ),
                "shift": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Base model: sigma shift of the sampling schedule."}),
                "noise_scale": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip": "Initial noise temperature. -1 = trained default (1.0 with spatial, 0.8 temporal only)."},
                ),
                "temporal_window": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "tooltip": "Source latent frames per window for long videos, 0 = whole clip. Memory is handled without windows too; windows keep long clips closer to the training length (up to 17 source frames)."},
                ),
                "temporal_context": (
                    "INT",
                    {"default": 2, "min": 1, "max": 16, "tooltip": "Overlap frames before every window."},
                ),
            }
        )
        return {"required": inputs}

    def upscale(
        self,
        upscaler: SteppeModel,
        latent: dict,
        spatial: bool,
        temporal: bool,
        hff: float,
        guidance: float,
        steps: int,
        seed: int,
        negative_hff: float,
        shift: float,
        noise_scale: float,
        temporal_window: int,
        temporal_context: int,
    ) -> tuple[dict]:
        settings = UpscaleSettings(
            spatial=spatial,
            temporal=temporal,
            hff=optional_hff(hff),
            guidance=guidance,
            negative_hff=optional_hff(negative_hff),
            steps=steps,
            shift=shift,
            noise_scale=None if noise_scale < 0 else noise_scale,
            seed=seed,
            temporal_window=temporal_window,
            temporal_context=temporal_context,
        )
        return run_upscale(upscaler, latent, settings)


class SteppeMeasureHFF:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("hff",)
    FUNCTION = "measure"
    DESCRIPTION = (
        "High frequency feature of an image or video frames ([0...1], averaged over the batch), "
        "e.g. to match the sharpness of a reference. Measure at the target resolution."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"image": ("IMAGE",)}}

    def measure(self, image: torch.Tensor) -> tuple[float]:
        frames = image[..., :3].movedim(-1, 1).float()
        raw = compute_raw_immerkaer(frames).mean(dim=0, keepdim=True)
        value = quantile_normalize(raw).item()
        return (round(value, 3),)


NODE_CLASS_MAPPINGS = {
    "SteppeLoadUpscaler": SteppeLoadUpscaler,
    "SteppeUpscaleLatent": SteppeUpscaleLatent,
    "SteppeUpscaleLatentAdvanced": SteppeUpscaleLatentAdvanced,
    "SteppeMeasureHFF": SteppeMeasureHFF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SteppeLoadUpscaler": "Load Steppe Upscaler",
    "SteppeUpscaleLatent": "Steppe Upscale Latent",
    "SteppeUpscaleLatentAdvanced": "Steppe Upscale Latent (Advanced)",
    "SteppeMeasureHFF": "Steppe Measure HFF",
}
